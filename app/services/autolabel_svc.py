"""已有模型预标注 + 类别名归一化映射。"""

from __future__ import annotations

import logging
import re
from pathlib import Path
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
    # 禁止「索引碰巧在范围内就硬套」——模型 wheelie=0 不能当 helmet=0
    # 仅当类别名归一化后完全一致时才按序对齐（由 check_autolabel_compat 保障）
    if class_mapping:
        return None
    if 0 <= pred_index < len(dataset_classes):
        ds_name = normalize_token(dataset_classes[pred_index])
        if name_key and name_key == ds_name:
            return pred_index
        # 模型与数据集类别列表完全一致（数量相同且逐项同名）时才按索引
        # 这里只做单框映射；整模兼容在 run_autolabel 入口校验
    return None


def check_autolabel_compat(
    model_names: dict | list,
    dataset_classes: list[str],
    class_map: dict,
) -> list[str]:
    """模型类别与数据集类别不匹配时拒绝静默硬套。

    返回 warnings（部分匹配时的提示）；一个都对不上则抛 ValueError。
    用户显式传了 class_map 时不拦。
    """
    if class_map:
        return []
    if isinstance(model_names, dict):
        model_labels = [str(model_names[k]) for k in sorted(model_names.keys())]
    else:
        model_labels = [str(x) for x in (model_names or [])]
    ds_norm = {normalize_token(c) for c in dataset_classes}
    hits = [m for m in model_labels if normalize_token(m) in ds_norm]
    if not hits:
        raise ValueError(
            f"这个模型认识的是 {model_labels}，和数据集的 {dataset_classes} 对不上，"
            f"不能用来预标注。请改用在本数据集上训练的模型，"
            f"或在高级参数里手工指定类别映射（class_map）。"
        )
    warnings: list[str] = []
    miss_model = [m for m in model_labels if normalize_token(m) not in ds_norm]
    miss_ds = [
        c
        for c in dataset_classes
        if normalize_token(c) not in {normalize_token(m) for m in model_labels}
    ]
    if miss_model or miss_ds:
        warnings.append(
            f"类别仅部分匹配：模型有而数据集没有 {miss_model or '无'}；"
            f"数据集有而模型没有 {miss_ds or '无'}。未匹配的检测框会被丢弃。"
        )
    return warnings


def uncertainty_of_boxes(boxes: list[dict]) -> float:
    if not boxes:
        return 0.9
    return 1.0 - max(float(b.get("conf", 0)) for b in boxes)


def run_autolabel(task: dict) -> None:
    from app import tasks as task_mod
    from app.services import model_svc as _ms

    params = task.get("params") or {}
    dataset_id = task.get("dataset_id") or params.get("dataset_id")
    task_id = task["id"]
    ds = dataset_svc.get_dataset(dataset_id)
    classes = ds["classes"]
    model_path = config.resolve_weight_path(params.get("model", ""))
    conf = float(params.get("conf", 0.15))
    iou = float(params.get("iou", 0.5))
    imgsz = int(params.get("imgsz", 640))
    scope = params.get("scope") or "unlabeled"
    overwrite = bool(params.get("overwrite", False))
    class_map = parse_class_mapping(params.get("class_map") or {}, classes)

    # 类别兼容：禁止 wheelie→helmet 静默硬套
    model_names = engine.get_model_class_names(model_path)
    # get_model_class_names 返回 list；check 也接受 dict
    names_dict = {i: n for i, n in enumerate(model_names)}
    compat_warnings = check_autolabel_compat(names_dict, classes, class_map)

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

    # OpenVINO：非动态形状时强制对齐导出 imgsz 与 batch=1
    prefer_ov = _ms.prefer_openvino_for_path(model_path)
    export_meta = _ms.get_export_meta_for_path(model_path)
    batch = config.INFER_BATCH_SIZE
    align_notes: list[str] = list(compat_warnings)
    if prefer_ov and export_meta and not export_meta.get("export_dynamic"):
        exp_imgsz = export_meta.get("export_imgsz")
        if exp_imgsz and int(exp_imgsz) != imgsz:
            align_notes.append(
                f"已按模型要求把尺寸从 {imgsz} 调整为 {exp_imgsz}（OpenVINO 静态导出）"
            )
            imgsz = int(exp_imgsz)
        if batch != 1:
            align_notes.append("已按模型要求把 batch 调整为 1（OpenVINO 静态导出）")
            batch = 1
    if align_notes:
        task_mod.update_task(task_id, message="；".join(align_notes))

    labeled = 0
    used_pt_fallback = False
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
        try:
            results = engine.predict_boxes_batch(
                model_path,
                paths,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                prefer_openvino=prefer_ov and not used_pt_fallback,
            )
        except Exception as exc:
            if prefer_ov and not used_pt_fallback:
                logger.warning("OpenVINO 推理失败，降级用 .pt：%s", exc)
                used_pt_fallback = True
                task_mod.update_task(
                    task_id,
                    message="OpenVINO 不兼容，已自动改用 .pt 继续",
                )
                results = engine.predict_boxes_batch(
                    model_path,
                    paths,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    prefer_openvino=False,
                )
            else:
                raise
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
        msg = f"预标注进度 {min(i + batch, total)}/{total}"
        if used_pt_fallback:
            msg += "（.pt）"
        task_mod.set_progress(task_id, pct, msg)

    tail = f"完成：处理 {labeled} 张，模型 {Path(model_path).name}"
    if used_pt_fallback:
        tail += "；本次 OpenVINO 已降级为 .pt"
    if align_notes:
        tail += "；" + "；".join(align_notes)
    task_mod.update_task(task_id, message=tail)
