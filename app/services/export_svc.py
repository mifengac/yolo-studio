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

# 人工把关过的状态：reviewed（画过/改过）与 confirmed（按 Enter 确认）都算
# 真正要排除的是 auto（模型预标注、无人审核）
HUMAN_VERIFIED = ("reviewed", "confirmed")


def _write_label_file(path: Path, boxes: list[dict]) -> None:
    lines = []
    for b in boxes:
        lines.append(
            f"{int(b['class_idx'])} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _iter_images_with_boxes(dataset_id: str):
    """遍历有标注框的图片（跳过 skipped）。"""
    page = 1
    while True:
        chunk = dataset_svc.list_images(
            dataset_id, page=page, page_size=200, sort="created"
        )
        if not chunk["items"]:
            break
        for it in chunk["items"]:
            if it.get("review_status") == "skipped":
                continue
            if int(it.get("box_count") or 0) <= 0:
                continue
            yield it
        if page * 200 >= chunk["total"]:
            break
        page += 1


def pool_composition(dataset_id: str) -> dict:
    """统计有框图片的状态构成，供预估文案使用。"""
    n_reviewed = n_confirmed = n_auto = n_other = 0
    for it in _iter_images_with_boxes(dataset_id):
        st = it.get("review_status") or ""
        if st == "reviewed":
            n_reviewed += 1
        elif st == "confirmed":
            n_confirmed += 1
        elif st == "auto":
            n_auto += 1
        else:
            n_other += 1
    human = n_reviewed + n_confirmed
    return {
        "reviewed": n_reviewed,
        "confirmed": n_confirmed,
        "auto": n_auto,
        "other": n_other,
        "human": human,
        "total_with_boxes": human + n_auto + n_other,
    }


def composition_summary(comp: dict, *, only_confirmed: bool) -> str:
    """生成「可用于训练 N 张（…）」说明。"""
    human = int(comp.get("human") or 0)
    reviewed = int(comp.get("reviewed") or 0)
    confirmed = int(comp.get("confirmed") or 0)
    auto = int(comp.get("auto") or 0)
    if only_confirmed:
        return (
            f"可用于训练 {human} 张"
            f"（人工标注 {reviewed} + 已确认 {confirmed}），"
            f"已排除未审核预标注 {auto} 张"
        )
    usable = human + auto
    return (
        f"可用于训练 {usable} 张"
        f"（人工 {human} + 未审核预标注 {auto}）"
    )


def collect_trainable_images(
    dataset_id: str, *, only_confirmed: bool = True
) -> list[dict]:
    items: list[dict] = []
    for it in _iter_images_with_boxes(dataset_id):
        st = it.get("review_status") or ""
        if only_confirmed:
            # 只用人把关过的：reviewed + confirmed，排除 auto
            if st not in HUMAN_VERIFIED:
                continue
        else:
            # 放开：有框即可（含 auto）
            if st not in HUMAN_VERIFIED and st != "auto":
                # 其它状态（若有）仍允许只要有框
                pass
        items.append(it)
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
            + (
                "请先人工标注或审核一些图片（模型自动预标注的需要先审核）。"
                if only_confirmed
                else "请先标注一些图片。"
            )
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
    comp = pool_composition(dataset_id)
    items = collect_trainable_images(dataset_id, only_confirmed=only_confirmed)
    if len(items) < 2:
        raise ValueError(
            "可用于训练的图片不足 2 张，无法划分 train/val。"
            + (
                "请先人工标注或审核一些图片（模型自动预标注的需要先审核）。"
                if only_confirmed
                else ""
            )
        )
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
        "composition": comp,
        "composition_text": composition_summary(comp, only_confirmed=only_confirmed),
    }
