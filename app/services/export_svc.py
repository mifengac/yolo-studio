"""导出 YOLO 格式目录 + data.yaml（按 group_key 划分防泄漏）。"""

from __future__ import annotations

import logging
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional
from uuid import uuid4

from app import config, db
from app.services import dataset_svc

logger = logging.getLogger(__name__)


def _write_label_file(path: Path, boxes: list[dict]) -> None:
    lines = []
    for b in boxes:
        lines.append(
            f"{int(b['class_idx'])} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def collect_trainable_images(
    dataset_id: str, *, only_confirmed: bool = True
) -> list[dict]:
    page = 1
    items: list[dict] = []
    while True:
        chunk = dataset_svc.list_images(
            dataset_id, page=page, page_size=200, sort="created"
        )
        if not chunk["items"]:
            break
        for it in chunk["items"]:
            if it.get("review_status") == "skipped":
                continue
            if only_confirmed and it.get("review_status") != "confirmed":
                continue
            if not only_confirmed and it.get("box_count", 0) <= 0:
                if it.get("review_status") not in ("reviewed", "confirmed", "auto"):
                    continue
            if it.get("box_count", 0) <= 0:
                continue
            items.append(it)
        if page * 200 >= chunk["total"]:
            break
        page += 1
    return items


def split_by_group(
    items: list[dict], val_ratio: float = 0.2, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """按 group_key 分组划分，避免近重复/同视频泄漏。"""
    groups: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        gk = it.get("group_key") or it["id"]
        groups[gk].append(it)

    keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    if len(keys) == 1:
        # 单组：按图划分
        imgs = groups[keys[0]]
        if len(imgs) == 1:
            return imgs, imgs
        n_val = max(1, min(len(imgs) - 1, int(round(len(imgs) * val_ratio))))
        return imgs[n_val:], imgs[:n_val]

    n_val_groups = max(1, min(len(keys) - 1, int(round(len(keys) * val_ratio))))
    val_keys = set(keys[:n_val_groups])
    train, val = [], []
    for k, imgs in groups.items():
        if k in val_keys:
            val.extend(imgs)
        else:
            train.extend(imgs)
    if not train:
        train, val = val[:-1], val[-1:]
    if not val:
        val = train[-1:]
        train = train[:-1]
    return train, val


def export_yolo_dataset(
    dataset_id: str,
    *,
    only_confirmed: bool = True,
    val_ratio: float = 0.2,
    out_name: Optional[str] = None,
) -> dict:
    ds = dataset_svc.get_dataset(dataset_id)
    items = collect_trainable_images(dataset_id, only_confirmed=only_confirmed)
    if not items:
        raise ValueError(
            "没有可用于训练的图片。"
            + ("请先确认一些标注（状态=已确认）。" if only_confirmed else "请先标注一些图片。")
        )

    # 每类至少 1 个框检查在 train 侧做；这里先统计
    class_counts = [0] * len(ds["classes"])
    for it in items:
        for b in dataset_svc.get_annotations(it["id"]):
            ci = int(b["class_idx"])
            if 0 <= ci < len(class_counts):
                class_counts[ci] += 1

    train_items, val_items = split_by_group(items, val_ratio=val_ratio)
    stamp = out_name or time.strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:4]
    export_root = dataset_svc.dataset_root(dataset_id) / "exports" / stamp
    for split in ("train", "val"):
        (export_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (export_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    def _copy_split(split_name: str, subset: list[dict]) -> int:
        n = 0
        for it in subset:
            src = dataset_svc.image_file_path(it)
            if not src.is_file():
                continue
            dst_img = export_root / "images" / split_name / it["filename"]
            shutil.copy2(src, dst_img)
            boxes = dataset_svc.get_annotations(it["id"])
            stem = Path(it["filename"]).stem
            _write_label_file(
                export_root / "labels" / split_name / f"{stem}.txt", boxes
            )
            with db._lock, db.connect() as conn:
                conn.execute(
                    "UPDATE image SET split=? WHERE id=?", (split_name, it["id"])
                )
            n += 1
        return n

    n_train = _copy_split("train", train_items)
    n_val = _copy_split("val", val_items)
    if n_train == 0 or n_val == 0:
        raise ValueError("train/val 划分后有一侧为空，请增加更多标注图片")

    names_yaml = "\n".join(
        f"  {i}: {name}" for i, name in enumerate(ds["classes"])
    )
    # 用相对 path 便于移植
    yaml_text = (
        f"path: {export_root.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n{names_yaml}\n"
    )
    yaml_path = export_root / "data.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")

    return {
        "export_dir": str(export_root),
        "data_yaml": str(yaml_path),
        "train_count": n_train,
        "val_count": n_val,
        "class_counts": class_counts,
        "classes": ds["classes"],
    }


def validate_for_train(
    dataset_id: str, *, only_confirmed: bool = True, min_boxes_per_class: int = 10
) -> dict:
    ds = dataset_svc.get_dataset(dataset_id)
    if not ds["classes"]:
        raise ValueError("数据集没有任何类别，请先添加类别")
    items = collect_trainable_images(dataset_id, only_confirmed=only_confirmed)
    if len(items) < 2:
        raise ValueError("可用于训练的图片不足 2 张，无法划分 train/val")
    class_counts = [0] * len(ds["classes"])
    for it in items:
        for b in dataset_svc.get_annotations(it["id"]):
            ci = int(b["class_idx"])
            if 0 <= ci < len(class_counts):
                class_counts[ci] += 1
    bad = []
    for i, c in enumerate(class_counts):
        if c < min_boxes_per_class:
            bad.append(
                f"「{ds['classes'][i]}」只有 {c} 个框（至少需要 {min_boxes_per_class}）"
            )
    return {
        "image_count": len(items),
        "class_counts": class_counts,
        "warnings": bad,
        "ok": len(items) >= 2 and all(c >= min_boxes_per_class for c in class_counts),
        "min_boxes_per_class": min_boxes_per_class,
    }
