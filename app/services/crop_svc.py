"""智能切图导入：大图检测 person（等）→ 上半身切图入库。

用于头盔等小目标：原图 4K 全景缩小后头部只剩几个像素，必须先切再标。
"""

from __future__ import annotations

import base64
import io
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import numpy as np
from PIL import Image

from app import config, db
from app.infer import engine
from app.services import dataset_svc

logger = logging.getLogger(__name__)

# 默认参数（实测标定，勿随意改）
# ★ bottom_ratio 必须与 bczj-classifier 的 CROP_PARAMS 保持一致。
# 训练用 1.3 而推理用 0.55（或反之），模型准确率会大幅下降且难以排查。
DEFAULTS = {
    "model": "yolo26n.pt",
    "target_class": "person",
    "conf": 0.25,
    # 检测 NMS 阈值：ultralytics 默认 0.7 太松，同一个人易拆成多框
    "det_iou": 0.45,
    "imgsz": 1280,
    "min_box_h": 100,
    "pad_ratio": 0.25,
    "top_ratio": -0.15,
    # 1.30：必须包含车身，否则模型学不会区分骑手与行人（0.55 只到腰部）
    "bottom_ratio": 1.30,
    # 切图前 person 框几何去重（兜底 NMS 漏网）
    # 实测标定：0.50/0.80 会把一车载两人的后座乘客误删（骑手与后座小孩 IoU 0.42/contain 0.82）
    "dedup_iou": 0.60,  # 两框 IoU 超过此值判为同一个人
    "dedup_contain": 0.99,  # 小框被大框覆盖超过此比例才可能判重复
    "dedup_min_area_ratio": 0.80,  # 且两框面积须接近，避免「大人套小孩」被误删
    "max_crops": 8000,
    "preview_limit": 20,
    # 质量过滤（保守阈值：宁可多放进来几张暗图，也不误删有效骑手）
    "min_brightness": 25,  # 平均亮度低于此值丢弃（0~255）
    "min_aspect": 0.5,  # 宽高比下限
    "max_aspect": 3.0,  # 宽高比上限
}


def _params_from(raw: Optional[dict]) -> dict:
    p = dict(DEFAULTS)
    if raw:
        for k in DEFAULTS:
            if k in raw and raw[k] is not None:
                p[k] = raw[k]
    p["conf"] = float(p["conf"])
    p["det_iou"] = float(p["det_iou"])
    p["imgsz"] = int(p["imgsz"])
    p["min_box_h"] = int(p["min_box_h"])
    p["pad_ratio"] = float(p["pad_ratio"])
    p["top_ratio"] = float(p["top_ratio"])
    p["bottom_ratio"] = float(p["bottom_ratio"])
    p["dedup_iou"] = float(p["dedup_iou"])
    p["dedup_contain"] = float(p["dedup_contain"])
    p["dedup_min_area_ratio"] = float(p["dedup_min_area_ratio"])
    p["max_crops"] = int(p["max_crops"])
    p["preview_limit"] = int(p.get("preview_limit") or DEFAULTS["preview_limit"])
    p["min_brightness"] = float(p["min_brightness"])
    p["min_aspect"] = float(p["min_aspect"])
    p["max_aspect"] = float(p["max_aspect"])
    return p


def _crop_quality_reject(crop: Image.Image, p: dict) -> Optional[str]:
    """切图质量过滤。返回丢弃原因键名，通过则返回 None。"""
    w, h = crop.size
    if h <= 0 or w <= 0:
        return "skipped_small"
    aspect = w / h
    if aspect < p["min_aspect"] or aspect > p["max_aspect"]:
        return "skipped_aspect"
    try:
        brightness = float(np.asarray(crop.convert("L")).mean())
    except Exception:
        brightness = 255.0
    if brightness < p["min_brightness"]:
        return "skipped_dark"
    return None


def _match_class_name(name: str, target: str) -> bool:
    return str(name).strip().lower() == str(target).strip().lower()


def _box_iou(a, b) -> float:
    """两框 IoU，a/b 均为 (x1,y1,x2,y2)。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


def _contain_ratio(small, big) -> float:
    """small 被 big 覆盖的面积比例——处理「一个框套在另一个里面」的情况。"""
    ix1, iy1 = max(small[0], big[0]), max(small[1], big[1])
    ix2, iy2 = min(small[2], big[2]), min(small[3], big[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / max((small[2] - small[0]) * (small[3] - small[1]), 1e-6)


def dedup_person_boxes(
    boxes,
    iou_thr: float = 0.60,
    contain_thr: float = 0.99,
    min_area_ratio: float = 0.80,
):
    """同一个人的多个重叠框只保留面积最大的那个。

    boxes: 每项为 (x1,y1,x2,y2,...) 或 [x1,y1,x2,y2,...]
    大框优先保留：切图 bottom_ratio=1.3 需要完整车身上下文。

    ★ 为什么「包含判据」要额外加面积比限制（min_area_ratio）：
      一车载两人时，后座乘客（尤其小孩）身体大部分落在骑手框范围内，
      contain_ratio 很容易超过 0.8，若仅凭这一条就判重复，会把后座乘客整个删掉。
      实测案例：骑手框与后座小孩框 IoU 仅 0.42、contain 0.82 —— 明显是两个人。
      加上「两框面积接近」的要求后，大人套小孩的情况就不会被误判。
      而「多人骑车」本身是要查处的违法行为，漏标代价很大。
    """

    def _xyxy(b):
        return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))

    def _area(b) -> float:
        x1, y1, x2, y2 = _xyxy(b)
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    ordered = sorted(boxes, key=_area, reverse=True)
    kept = []
    for b in ordered:
        bb = _xyxy(b)
        area_b = _area(b)
        is_dup = False
        for k in kept:
            kk = _xyxy(k)
            # k 先入且面积更大
            if _box_iou(bb, kk) > iou_thr:
                is_dup = True
                break
            # 包含判据：仅在两框大小接近时才认定为同一目标
            if (
                _contain_ratio(bb, kk) > contain_thr
                and area_b / max(_area(k), 1e-6) >= min_area_ratio
            ):
                is_dup = True
                break
        if not is_dup:
            kept.append(b)
    return kept


def crop_box_region(
    W: int,
    H: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    pad_ratio: float,
    top_ratio: float,
    bottom_ratio: float,
) -> tuple[int, int, int, int]:
    """按标定规则从 person 框算切图区域（像素，含边界 clamp）。

    bottom_ratio=1.3 时尽量包住人+车身，便于模型用车身上下文区分骑手与行人。
    """
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    pad = bw * pad_ratio
    cx1 = max(0, int(x1 - pad))
    cx2 = min(W, int(x2 + pad))
    cy1 = max(0, int(y1 + bh * top_ratio))  # top_ratio 为负 = 向上扩
    cy2 = min(H, int(y1 + bh * bottom_ratio))
    if cx2 <= cx1 or cy2 <= cy1:
        return 0, 0, 0, 0
    return cx1, cy1, cx2, cy2


def _detect_xyxy(
    model_path: str,
    image_path: Path,
    *,
    conf: float,
    imgsz: int,
    target_class: str,
    det_iou: float = 0.45,
) -> list[tuple[float, float, float, float, float, str]]:
    """返回 [(x1,y1,x2,y2,conf,class_name), ...] 像素坐标。"""
    model = engine.load_model(model_path, prefer_openvino=False)
    names = model.names if isinstance(model.names, dict) else {
        i: n for i, n in enumerate(model.names or [])
    }
    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=det_iou,  # 收紧 NMS，减少同一个人多框
        imgsz=imgsz,
        device="cpu",
        half=False,
        verbose=False,
    )
    out: list[tuple[float, float, float, float, float, str]] = []
    if not results:
        return out
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return out
    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(int)
    for i in range(len(xyxy)):
        ci = int(clss[i])
        cname = str(names.get(ci, str(ci)))
        if not _match_class_name(cname, target_class):
            continue
        x1, y1, x2, y2 = map(float, xyxy[i])
        out.append((x1, y1, x2, y2, float(confs[i]), cname))
    return out


def _stem_safe(filename: str) -> str:
    stem = Path(filename).stem
    # 去掉路径穿越成分
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)[:80] or "img"


def process_one_image(
    image_path: Path,
    src_filename: str,
    *,
    model_path: str,
    p: dict,
    max_remaining: int,
) -> dict[str, Any]:
    """处理一张大图，返回切图 list[(name, bytes)] 与统计。"""
    stats = {
        "detected": 0,
        "skipped_small": 0,
        "skipped_dup_box": 0,  # 同一个人多框（检测重叠），非 sha1 内容重复
        "skipped_dark": 0,
        "skipped_aspect": 0,
        "crops": [],  # list of (filename, bytes)
        "crop_sizes": [],  # list of (w, h)，用于汇总中位尺寸
        "no_target": False,
        # 调试：被几何去重掉的框 [(x1,y1,x2,y2,conf), ...]，供验收抽查
        "dropped_dup_boxes": [],
    }
    if max_remaining <= 0:
        return stats

    boxes = _detect_xyxy(
        model_path,
        image_path,
        conf=p["conf"],
        imgsz=p["imgsz"],
        target_class=p["target_class"],
        det_iou=float(p.get("det_iou", DEFAULTS["det_iou"])),
    )
    stats["detected"] = len(boxes)
    if not boxes:
        stats["no_target"] = True
        return stats

    # 1) 过小框先剔除  2) 几何去重只保留大框  3) 再切图
    tall_enough = []
    for b in boxes:
        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
        if (y2 - y1) < p["min_box_h"]:
            stats["skipped_small"] += 1
        else:
            tall_enough.append(b)

    before_dedup = len(tall_enough)
    kept = dedup_person_boxes(
        tall_enough,
        iou_thr=float(p.get("dedup_iou", DEFAULTS["dedup_iou"])),
        contain_thr=float(p.get("dedup_contain", DEFAULTS["dedup_contain"])),
        min_area_ratio=float(
            p.get("dedup_min_area_ratio", DEFAULTS["dedup_min_area_ratio"])
        ),
    )
    stats["skipped_dup_box"] = before_dedup - len(kept)
    if stats["skipped_dup_box"] > 0:
        kept_ids = {id(b) for b in kept}
        for b in tall_enough:
            if id(b) not in kept_ids:
                stats["dropped_dup_boxes"].append(
                    (float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4]))
                )

    with Image.open(image_path) as im:
        im = im.convert("RGB")
        W, H = im.size
        stem = _stem_safe(src_filename)
        idx = 0
        for x1, y1, x2, y2, conf, _cn in kept:
            cx1, cy1, cx2, cy2 = crop_box_region(
                W,
                H,
                x1,
                y1,
                x2,
                y2,
                pad_ratio=p["pad_ratio"],
                top_ratio=p["top_ratio"],
                bottom_ratio=p["bottom_ratio"],
            )
            if cx2 <= cx1 or cy2 <= cy1:
                stats["skipped_small"] += 1
                continue
            crop = im.crop((cx1, cy1, cx2, cy2))
            reason = _crop_quality_reject(crop, p)
            if reason:
                stats[reason] = stats.get(reason, 0) + 1
                logger.debug(
                    "切图过滤 %s reason=%s size=%sx%s",
                    src_filename,
                    reason,
                    crop.size[0],
                    crop.size[1],
                )
                continue
            idx += 1
            name = f"{stem}_p{idx:02d}.jpg"
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=92)
            stats["crops"].append((name, buf.getvalue()))
            stats["crop_sizes"].append((int(crop.size[0]), int(crop.size[1])))
            if len(stats["crops"]) >= max_remaining:
                break
    return stats


def _iter_zip_images(zip_path: Path, extract_dir: Path) -> list[tuple[str, Path]]:
    """解压 ZIP，返回 [(原文件名, 路径), ...]，防目录穿越。"""
    out: list[tuple[str, Path]] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if not name or name.startswith("."):
                continue
            ext = Path(name).suffix.lower()
            if ext not in config.IMAGE_EXTS:
                continue
            # 防穿越
            target = (extract_dir / name).resolve()
            if not str(target).startswith(str(extract_dir.resolve())):
                continue
            # 重名
            if target.exists():
                target = extract_dir / f"{uuid4().hex[:6]}_{name}"
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            out.append((name, target))
    return out


def _list_dataset_images(dataset_id: str) -> list[dict]:
    items: list[dict] = []
    page = 1
    while True:
        chunk = dataset_svc.list_images(
            dataset_id, page=page, page_size=200, sort="filename"
        )
        if not chunk["items"]:
            break
        items.extend(chunk["items"])
        if page * 200 >= chunk["total"]:
            break
        page += 1
    return items


def _task_canceled(task_id: str) -> bool:
    from app import tasks as task_mod

    t = task_mod.get_task(task_id)
    if not t:
        return True
    return t.get("status") in ("canceled", "failed")


def run_crop_import(task: dict) -> None:
    """后台任务：ZIP 或源数据集 → 切图入库。"""
    from app import tasks as task_mod

    params = task.get("params") or {}
    task_id = task["id"]
    dest_id = task.get("dataset_id") or params.get("dataset_id")
    if not dest_id:
        raise ValueError("缺少目标数据集 id")
    dataset_svc.get_dataset(dest_id)
    p = _params_from(params)
    model_path = str(config.resolve_weight_path(p["model"]))

    source = params.get("source") or "zip"  # zip | dataset
    tmp_root = Path(params.get("tmp_dir") or "")
    zip_path = Path(params.get("zip_path") or "")
    src_dataset_id = params.get("src_dataset_id")

    work_dir = None
    image_list: list[tuple[str, Path]] = []

    try:
        if source == "zip":
            if not zip_path.is_file():
                raise FileNotFoundError("上传的 ZIP 已丢失，请重新上传")
            work_dir = Path(tempfile.mkdtemp(prefix="crop_zip_", dir=str(config.DATA_DIR / "logs")))
            image_list = _iter_zip_images(zip_path, work_dir)
        elif source == "dataset":
            if not src_dataset_id:
                raise ValueError("缺少源数据集 id")
            if src_dataset_id == dest_id:
                raise ValueError("源数据集不能与目标数据集相同，请新建一个「切图」数据集再导入")
            dataset_svc.get_dataset(src_dataset_id)
            for it in _list_dataset_images(src_dataset_id):
                path = dataset_svc.image_file_path(it)
                if path.is_file():
                    image_list.append((it.get("filename") or path.name, path))
        else:
            raise ValueError(f"未知来源: {source}")

        total = len(image_list)
        if total == 0:
            task_mod.set_progress(task_id, 100, "没有找到可处理的图片")
            return

        imported_total = 0
        detected_total = 0
        skipped_small = 0
        skipped_dup_box_total = 0  # 重叠重复（同人多框）
        skipped_dark = 0
        skipped_aspect = 0
        skipped_dup_total = 0  # 内容重复（sha1）
        skipped_bad_total = 0
        no_target_files: list[str] = []
        done_imgs = 0
        all_crop_ws: list[int] = []
        all_crop_hs: list[int] = []

        for src_name, src_path in image_list:
            if _task_canceled(task_id):
                task_mod.update_task(
                    task_id,
                    status="canceled",
                    message=f"已取消。此前已切图入库 {imported_total} 张",
                    finished_at=db.utcnow(),
                )
                return

            remain = p["max_crops"] - imported_total
            if remain <= 0:
                break

            try:
                st = process_one_image(
                    src_path,
                    src_name,
                    model_path=model_path,
                    p=p,
                    max_remaining=remain,
                )
            except Exception as exc:
                logger.warning("切图失败 %s: %s", src_name, exc)
                done_imgs += 1
                task_mod.set_progress(
                    task_id,
                    100.0 * done_imgs / total,
                    f"处理大图 {done_imgs}/{total}，已切 {imported_total} 张（{src_name} 失败）",
                )
                continue

            detected_total += st["detected"]
            skipped_small += st["skipped_small"]
            skipped_dup_box_total += int(st.get("skipped_dup_box") or 0)
            skipped_dark += int(st.get("skipped_dark") or 0)
            skipped_aspect += int(st.get("skipped_aspect") or 0)
            if st["no_target"]:
                no_target_files.append(src_name)

            crops = st["crops"]
            if crops:
                # group_key = 原图文件名，保证 train/val 按原图分组
                r = dataset_svc.import_files(
                    dest_id,
                    crops,
                    source="crop",
                    group_key=src_name,
                )
                imported_total += int(r.get("imported") or 0)
                skipped_dup_total += int(r.get("skipped_dup") or 0)
                skipped_bad_total += int(r.get("skipped_bad") or 0)
                for cw, ch in st.get("crop_sizes") or []:
                    all_crop_ws.append(int(cw))
                    all_crop_hs.append(int(ch))

            done_imgs += 1
            task_mod.set_progress(
                task_id,
                100.0 * done_imgs / total,
                f"处理大图 {done_imgs}/{total}，已切出 {imported_total} 张",
            )

        # 未检出列表截断，避免 params 过大
        no_target_show = no_target_files[:200]
        median_size_str = ""
        median_w = median_h = None
        if all_crop_ws and all_crop_hs:
            sw = sorted(all_crop_ws)
            sh = sorted(all_crop_hs)
            mid = len(sw) // 2
            if len(sw) % 2:
                median_w, median_h = sw[mid], sh[mid]
            else:
                median_w = (sw[mid - 1] + sw[mid]) // 2
                median_h = (sh[mid - 1] + sh[mid]) // 2
            median_size_str = f"，中位尺寸 {median_w}×{median_h}"
        # 闭合：检出 = 重叠重复 + 过小 + 过暗 + 比例异常 + 内容重复 + 损坏 + 入库
        summary = (
            f"处理大图 {done_imgs} 张，检出目标 {detected_total} 个，"
            f"跳过重叠重复 {skipped_dup_box_total} 个，"
            f"跳过过小 {skipped_small} 个，跳过过暗 {skipped_dark} 个，"
            f"跳过比例异常 {skipped_aspect} 个，跳过内容重复 {skipped_dup_total} 个"
            + (f"，跳过损坏 {skipped_bad_total} 个" if skipped_bad_total else "")
            + f"，切图入库 {imported_total} 张{median_size_str}，"
            f"其中 {len(no_target_files)} 张大图未检出任何目标"
        )
        accounted = (
            skipped_dup_box_total
            + skipped_small
            + skipped_dark
            + skipped_aspect
            + skipped_dup_total
            + skipped_bad_total
            + imported_total
        )
        if detected_total != accounted:
            logger.warning(
                "切图统计未闭合：检出=%s 合计=%s"
                "（重叠=%s 过小=%s 过暗=%s 比例=%s 内容重复=%s 损坏=%s 入库=%s）",
                detected_total,
                accounted,
                skipped_dup_box_total,
                skipped_small,
                skipped_dark,
                skipped_aspect,
                skipped_dup_total,
                skipped_bad_total,
                imported_total,
            )
            summary += f"（统计核对：合计 {accounted}/{detected_total}）"
        result = {
            "source_images": done_imgs,
            "detected": detected_total,
            "skipped_dup_box": skipped_dup_box_total,
            "skipped_small": skipped_small,
            "skipped_dark": skipped_dark,
            "skipped_aspect": skipped_aspect,
            "skipped_dup": skipped_dup_total,
            "skipped_bad": skipped_bad_total,
            "imported": imported_total,
            "no_target_count": len(no_target_files),
            "no_target_files": no_target_show,
            "median_w": median_w,
            "median_h": median_h,
        }
        # 直接标 success，避免 tasks 框架用默认「完成」盖掉总结文案
        task_mod.update_task(
            task_id,
            status="success",
            progress=100,
            message=summary,
            params={**params, "result": result},
            finished_at=db.utcnow(),
        )
    finally:
        # 清理临时 ZIP 与解压目录
        if work_dir and work_dir.is_dir():
            shutil.rmtree(work_dir, ignore_errors=True)
        if zip_path and zip_path.is_file() and "crop_" in zip_path.name:
            zip_path.unlink(missing_ok=True)
        if tmp_root and Path(tmp_root).is_dir() and "crop_" in str(tmp_root):
            shutil.rmtree(tmp_root, ignore_errors=True)


def preview_crops(
    *,
    source: str,
    zip_bytes: Optional[bytes] = None,
    src_dataset_id: Optional[str] = None,
    params: Optional[dict] = None,
) -> dict:
    """同步预览：最多处理 preview_limit 张大图，返回 base64 缩略图，不入库。"""
    p = _params_from(params)
    model_path = str(config.resolve_weight_path(p["model"]))
    limit = p["preview_limit"]
    previews: list[dict] = []
    no_target: list[str] = []
    tmp = None

    try:
        image_list: list[tuple[str, Path]] = []
        if source == "zip":
            if not zip_bytes:
                raise ValueError("请上传 ZIP 文件")
            tmp = Path(tempfile.mkdtemp(prefix="crop_prev_"))
            zpath = tmp / "in.zip"
            zpath.write_bytes(zip_bytes)
            image_list = _iter_zip_images(zpath, tmp)[:limit]
        elif source == "dataset":
            if not src_dataset_id:
                raise ValueError("请选择源数据集")
            for it in _list_dataset_images(src_dataset_id)[:limit]:
                path = dataset_svc.image_file_path(it)
                if path.is_file():
                    image_list.append((it.get("filename") or path.name, path))
        else:
            raise ValueError("来源只能是 zip 或 dataset")

        for src_name, src_path in image_list:
            st = process_one_image(
                src_path,
                src_name,
                model_path=model_path,
                p=p,
                max_remaining=50,  # 单张大图预览最多 50 个切图
            )
            if st["no_target"]:
                no_target.append(src_name)
            for name, content in st["crops"]:
                with Image.open(io.BytesIO(content)) as full:
                    fw, fh = full.size
                    thumb = full.copy()
                    thumb.thumbnail((240, 240))
                    buf = io.BytesIO()
                    thumb.save(buf, format="JPEG", quality=80)
                    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                previews.append(
                    {
                        "filename": name,
                        "source": src_name,
                        "thumb_b64": b64,
                        "w": fw,
                        "h": fh,
                    }
                )

        return {
            "source_images": len(image_list),
            "crop_count": len(previews),
            "no_target_files": no_target,
            "items": previews,
            "params": p,
        }
    finally:
        if tmp and tmp.is_dir():
            shutil.rmtree(tmp, ignore_errors=True)


def save_zip_temp(dataset_id: str, data: bytes, filename: str = "upload.zip") -> dict:
    """ZIP 落盘供后台任务使用。"""
    dataset_svc.get_dataset(dataset_id)
    root = config.DATA_DIR / "logs" / "crop_uploads"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"crop_{dataset_id}_{uuid4().hex[:8]}.zip"
    path.write_bytes(data)
    return {"zip_path": str(path), "filename": filename}
