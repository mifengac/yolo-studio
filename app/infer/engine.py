"""YOLO 模型缓存加载与批量推理（CPU / OpenVINO 优先）。"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

from app import config

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()
_LOAD_LOCKS: dict[str, threading.Lock] = {}


def _cache_key(path: Path, prefer_openvino: bool) -> str:
    return f"{path.resolve()}|ov={int(prefer_openvino)}"


def _get_load_lock(key: str) -> threading.Lock:
    with _CACHE_LOCK:
        if key not in _LOAD_LOCKS:
            _LOAD_LOCKS[key] = threading.Lock()
        return _LOAD_LOCKS[key]


def _find_openvino_dir(pt_path: Path) -> Optional[Path]:
    """同目录或 data/models 下找 *_openvino_model。"""
    name = pt_path.stem
    candidates = [
        pt_path.parent / f"{name}_openvino_model",
        config.MODELS_DIR / f"{name}_openvino_model",
        config.MODELS_DIR / f"{pt_path.stem}_openvino_model",
    ]
    for c in candidates:
        if c.is_dir() and any(c.iterdir()):
            return c
    return None


def load_model(path: str | Path, *, prefer_openvino: bool = True):
    """加载模型；prefer_openvino 时优先 OpenVINO 导出目录。"""
    config.apply_torch_threads()
    pt = Path(path).resolve()
    if not pt.exists() and not (prefer_openvino and _find_openvino_dir(pt)):
        raise FileNotFoundError(f"模型不存在: {pt}")

    ov_dir = _find_openvino_dir(pt) if prefer_openvino else None
    load_path = ov_dir if ov_dir else pt
    key = _cache_key(pt, prefer_openvino and ov_dir is not None)

    with _CACHE_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]

    lock = _get_load_lock(key)
    with lock:
        with _CACHE_LOCK:
            if key in _MODEL_CACHE:
                return _MODEL_CACHE[key]

        from ultralytics import YOLO

        # 关闭在线同步
        try:
            from ultralytics.utils import SETTINGS

            SETTINGS.update({"sync": False})
        except Exception:
            pass

        logger.info("加载模型 %s (openvino=%s)", load_path, ov_dir is not None)
        model = YOLO(str(load_path))
        with _CACHE_LOCK:
            _MODEL_CACHE[key] = model
        return model


def predict_boxes_batch(
    model_path: str | Path,
    image_paths: list[str],
    *,
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 640,
    prefer_openvino: bool = True,
) -> list[list[dict]]:
    """批量推理，返回每张图的框列表。

    每个框: {class_idx, class_name, conf, cx, cy, w, h} 归一化坐标。
    """
    if not image_paths:
        return []

    model = load_model(model_path, prefer_openvino=prefer_openvino)
    names = model.names if isinstance(model.names, dict) else {
        i: n for i, n in enumerate(model.names or [])
    }

    results = model.predict(
        source=image_paths,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device="cpu",
        half=False,
        verbose=False,
        stream=False,
    )

    all_boxes: list[list[dict]] = []
    for r in results:
        boxes: list[dict] = []
        if r.boxes is None or len(r.boxes) == 0:
            all_boxes.append(boxes)
            continue
        h, w = r.orig_shape[:2]
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(float, xyxy[i])
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            cx = (x1 + x2) / 2.0 / max(w, 1)
            cy = (y1 + y2) / 2.0 / max(h, 1)
            nw = bw / max(w, 1)
            nh = bh / max(h, 1)
            ci = int(clss[i])
            boxes.append(
                {
                    "class_idx": ci,
                    "class_name": str(names.get(ci, str(ci))),
                    "conf": float(confs[i]),
                    "cx": cx,
                    "cy": cy,
                    "w": nw,
                    "h": nh,
                }
            )
        all_boxes.append(boxes)
    return all_boxes


def export_openvino(pt_path: str | Path, imgsz: int = 416) -> Optional[Path]:
    """导出 OpenVINO，失败返回 None（优雅降级）。"""
    try:
        pt = Path(pt_path).resolve()
        if not pt.is_file():
            return None
        model = load_model(pt, prefer_openvino=False)
        out = model.export(format="openvino", imgsz=imgsz, half=False, device="cpu")
        out_path = Path(str(out))
        # 复制/移动到 models 目录规范位置
        target = config.MODELS_DIR / f"{pt.stem}_openvino_model"
        if out_path.is_dir() and out_path.resolve() != target.resolve():
            import shutil

            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(out_path, target)
            return target
        if out_path.is_dir():
            return out_path
        return None
    except Exception as exc:
        logger.warning("OpenVINO 导出失败（将继续用 .pt）: %s", exc)
        return None


def get_model_class_names(path: str | Path) -> list[str]:
    model = load_model(path, prefer_openvino=False)
    names = model.names
    if isinstance(names, dict):
        return [str(names[i]) for i in sorted(names.keys())]
    return [str(n) for n in names]
