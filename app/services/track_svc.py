"""视频抽帧 + ByteTrack 传播标注。"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from uuid import uuid4

from app import config, db
from app.services import dataset_svc
from app.services.autolabel_svc import uncertainty_of_boxes

logger = logging.getLogger(__name__)


def import_video(
    dataset_id: str,
    video_bytes: bytes,
    filename: str,
    fps: float = 2.0,
) -> dict:
    """抽帧入库，同一视频写同一 group_key。"""
    import cv2
    import tempfile
    import os

    dataset_svc.get_dataset(dataset_id)
    group_key = f"video_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
    root = dataset_svc.ensure_dataset_dirs(dataset_id)
    tmp = root / "exports" / f"_tmp_{group_key}{Path(filename).suffix or '.mp4'}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(video_bytes)

    cap = cv2.VideoCapture(str(tmp))
    if not cap.isOpened():
        tmp.unlink(missing_ok=True)
        raise ValueError("无法打开视频文件")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(round(video_fps / max(fps, 0.1))))
    frame_idx = 0
    saved = 0
    files: list[tuple[str, bytes]] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % interval == 0:
                name = f"{group_key}_f{frame_idx:06d}.jpg"
                ok2, buf = cv2.imencode(".jpg", frame)
                if ok2:
                    files.append((name, buf.tobytes()))
                    saved += 1
            frame_idx += 1
            if saved >= 5000:
                break
    finally:
        cap.release()
        tmp.unlink(missing_ok=True)

    result = dataset_svc.import_files(
        dataset_id, files, source="video", group_key=group_key
    )
    result["group_key"] = group_key
    result["frames"] = saved
    return result


def run_track(task: dict) -> None:
    from app import tasks as task_mod
    from app.infer import engine

    params = task.get("params") or {}
    dataset_id = task.get("dataset_id") or params.get("dataset_id")
    task_id = task["id"]
    start_id = params.get("start_image_id")
    max_frames = min(int(params.get("max_frames", 300)), config.TRACK_MAX_FRAMES)
    conf = float(params.get("conf", 0.25))
    model_path = config.resolve_weight_path(
        params.get("model") or "0517_yolo26s_wheelie_multi-rider.pt"
    )

    start = dataset_svc.get_image(start_id)
    if start["dataset_id"] != dataset_id:
        raise ValueError("起始图片不属于该数据集")
    group_key = start.get("group_key") or ""

    # 取同 group 或按文件名排序的后续图
    all_items = []
    page = 1
    while True:
        chunk = dataset_svc.list_images(
            dataset_id, page=page, page_size=200, sort="filename"
        )
        if not chunk["items"]:
            break
        all_items.extend(chunk["items"])
        if page * 200 >= chunk["total"]:
            break
        page += 1

    if group_key:
        sequence = [x for x in all_items if x.get("group_key") == group_key]
    else:
        sequence = all_items

    # 从 start 开始
    ids = [x["id"] for x in sequence]
    if start_id not in ids:
        sequence = [start] + [x for x in sequence if x["id"] != start_id]
    else:
        idx = ids.index(start_id)
        sequence = sequence[idx : idx + max_frames]

    sequence = sequence[:max_frames]
    total = len(sequence)
    if total == 0:
        task_mod.set_progress(task_id, 100, "没有可跟踪的帧")
        return

    model = engine.load_model(model_path, prefer_openvino=False)
    ds = dataset_svc.get_dataset(dataset_id)
    names = model.names if isinstance(model.names, dict) else {
        i: n for i, n in enumerate(model.names or [])
    }

    processed = 0
    for it in sequence:
        path = dataset_svc.image_file_path(it)
        if not path.is_file():
            continue
        try:
            results = model.track(
                source=str(path),
                persist=True,
                tracker="bytetrack.yaml",
                conf=conf,
                device="cpu",
                verbose=False,
            )
        except Exception as exc:
            logger.warning("track 失败 %s: %s，退回 predict", it["id"], exc)
            results = model.predict(
                source=str(path), conf=conf, device="cpu", verbose=False
            )

        mapped = []
        if results:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                h, w = r.orig_shape[:2]
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                for j in range(len(xyxy)):
                    ci = int(clss[j])
                    # 简单按索引映射
                    if ci >= len(ds["classes"]):
                        # 尝试按名
                        from app.services.autolabel_svc import (
                            normalize_token,
                            resolve_class_index,
                        )

                        mapped_ci = resolve_class_index(
                            ci, str(names.get(ci, "")), ds["classes"], {}
                        )
                        if mapped_ci is None:
                            continue
                        ci = mapped_ci
                    x1, y1, x2, y2 = map(float, xyxy[j])
                    mapped.append(
                        {
                            "class_idx": ci,
                            "cx": (x1 + x2) / 2.0 / max(w, 1),
                            "cy": (y1 + y2) / 2.0 / max(h, 1),
                            "w": max(0.0, x2 - x1) / max(w, 1),
                            "h": max(0.0, y2 - y1) / max(h, 1),
                            "conf": float(confs[j]),
                            "source": "track",
                        }
                    )
        status = "auto" if mapped else it.get("review_status") or "unlabeled"
        # 不覆盖已确认
        if it.get("review_status") in ("confirmed", "reviewed") and it.get("box_count", 0) > 0:
            pass
        else:
            dataset_svc.put_annotations(it["id"], mapped, review_status=status)
            with db._lock, db.connect() as conn:
                conn.execute(
                    "UPDATE image SET uncertainty=? WHERE id=?",
                    (uncertainty_of_boxes(mapped), it["id"]),
                )
        processed += 1
        task_mod.set_progress(
            task_id, 100.0 * processed / total, f"跟踪传播 {processed}/{total}"
        )

    task_mod.update_task(task_id, message=f"跟踪完成 {processed} 帧")
