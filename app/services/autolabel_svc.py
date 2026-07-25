"""已有模型预标注 + 类别名归一化映射。"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Optional

from app import config, db
from app.infer import engine
from app.services import dataset_svc

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[^a-z0-9]+")
_ALIASES = {
    "rider": ("multirider", "rider"),
    "multirider": ("multirider", "rider"),
    "multipeople": ("multiperson", "person"),
    "wheelie": ("wheelie",),
    "nohelmet": ("nohelmet", "withouthelmet"),
    "helmet": ("helmet", "withhelmet"),
}


def normalize_token(value: str) -> str:
    return _TOKEN_RE.sub("", str(value or "").strip().lower())


def parse_class_mapping(value, dataset_classes: list[str]) -> dict[str, int]:
    dataset_lookup = {
        normalize_token(name): index for index, name in enumerate(dataset_classes)
    }
    mapping: dict[str, int] = {}

    if isinstance(value, dict):
        pairs: Iterable[tuple[str, object]] = value.items()
    else:
        pairs = []

    for raw_source, raw_target in pairs:
        source_key = normalize_token(str(raw_source))
        if not source_key and str(raw_source).strip().isdigit():
            source_key = str(int(raw_source))
        # 也支持 "0" 这种数字键
        if str(raw_source).strip().isdigit():
            mapping[str(int(raw_source))] = (
                int(raw_target)
                if isinstance(raw_target, int) or str(raw_target).isdigit()
                else dataset_lookup.get(normalize_token(str(raw_target)), -1)
            )
            if mapping.get(str(int(raw_source)), -1) < 0:
                mapping.pop(str(int(raw_source)), None)
            continue

        target_index = None
        if isinstance(raw_target, int) or str(raw_target).strip().isdigit():
            candidate = int(str(raw_target).strip())
            if 0 <= candidate < len(dataset_classes):
                target_index = candidate
        else:
            target_key = normalize_token(str(raw_target))
            if target_key in dataset_lookup:
                target_index = dataset_lookup[target_key]
        if target_index is not None and source_key:
            mapping[source_key] = target_index
    return mapping


def resolve_class_index(
    pred_index: int,
    pred_name: str,
    dataset_classes: list[str],
    class_mapping: dict[str, int],
) -> Optional[int]:
    name_key = normalize_token(pred_name)
    index_key = str(int(pred_index))

    if index_key in class_mapping:
        return class_mapping[index_key]
    if name_key and name_key in class_mapping:
        return class_mapping[name_key]

    dataset_lookup = {
        normalize_token(name): index for index, name in enumerate(dataset_classes)
    }
    if name_key and name_key in dataset_lookup:
        return dataset_lookup[name_key]
    for alias in _ALIASES.get(name_key, ()):
        if alias in dataset_lookup:
            return dataset_lookup[alias]
    # 同名序
    if 0 <= pred_index < len(dataset_classes) and not class_mapping:
        return pred_index
    return None


def uncertainty_of_boxes(boxes: list[dict]) -> float:
    if not boxes:
        return 0.9
    return 1.0 - max(float(b.get("conf", 0)) for b in boxes)


def run_autolabel(task: dict) -> None:
    from app import tasks as task_mod

    params = task.get("params") or {}
    dataset_id = task.get("dataset_id") or params.get("dataset_id")
    task_id = task["id"]
    ds = dataset_svc.get_dataset(dataset_id)
    classes = ds["classes"]
    model_path = config.resolve_weight_path(params.get("model", ""))
    conf = float(params.get("conf", 0.25))
    iou = float(params.get("iou", 0.5))
    imgsz = int(params.get("imgsz", 640))
    scope = params.get("scope") or "unlabeled"
    overwrite = bool(params.get("overwrite", False))
    class_map = parse_class_mapping(params.get("class_map") or {}, classes)

    # 选图
    page = 1
    items: list[dict] = []
    while True:
        chunk = dataset_svc.list_images(
            dataset_id, page=page, page_size=200, sort="created"
        )
        if not chunk["items"]:
            break
        for it in chunk["items"]:
            if scope == "unlabeled" and it["review_status"] not in (
                "unlabeled",
                "auto",
            ):
                if not overwrite:
                    continue
            if not overwrite and it["box_count"] > 0 and it["review_status"] in (
                "reviewed",
                "confirmed",
            ):
                continue
            items.append(it)
        if page * 200 >= chunk["total"]:
            break
        page += 1

    total = len(items)
    if total == 0:
        task_mod.set_progress(task_id, 100, "没有需要预标注的图片")
        return

    labeled = 0
    batch = config.INFER_BATCH_SIZE
    for i in range(0, total, batch):
        part = items[i : i + batch]
        paths = []
        valid = []
        for it in part:
            p = dataset_svc.image_file_path(it)
            if p.is_file():
                paths.append(str(p))
                valid.append(it)
        if not paths:
            continue
        from app.services import model_svc as _ms

        prefer_ov = _ms.prefer_openvino_for_path(model_path)
        results = engine.predict_boxes_batch(
            model_path,
            paths,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            prefer_openvino=prefer_ov,
        )
        for it, boxes in zip(valid, results):
            mapped = []
            for b in boxes:
                ci = resolve_class_index(
                    b["class_idx"], b["class_name"], classes, class_map
                )
                if ci is None:
                    continue
                mapped.append(
                    {
                        "class_idx": ci,
                        "cx": b["cx"],
                        "cy": b["cy"],
                        "w": b["w"],
                        "h": b["h"],
                        "conf": b["conf"],
                        "source": "auto",
                    }
                )
            unc = uncertainty_of_boxes(mapped)
            status = "auto" if mapped else "unlabeled"
            dataset_svc.put_annotations(it["id"], mapped, review_status=status)
            with db._lock, db.connect() as conn:
                conn.execute(
                    "UPDATE image SET uncertainty=? WHERE id=?", (unc, it["id"])
                )
            labeled += 1
        pct = 100.0 * min(i + batch, total) / total
        task_mod.set_progress(
            task_id, pct, f"预标注进度 {min(i + batch, total)}/{total}"
        )

    task_mod.update_task(
        task_id,
        message=f"完成：处理 {labeled} 张，模型 {model_path.name}",
    )
