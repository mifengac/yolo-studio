"""MobileSAM 点选辅助（LRU 缓存 embedding）。"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from app import config
from app.services import dataset_svc

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_predictor = None
# image_id -> numpy RGB array（CPU 上编码贵，至少复用原图数组）
_embed_cache: OrderedDict[str, Any] = OrderedDict()
_MAX = config.SAM_EMBED_CACHE_SIZE


def _get_predictor():
    global _predictor
    if _predictor is not None:
        return _predictor
    with _lock:
        if _predictor is not None:
            return _predictor
        config.apply_torch_threads()
        path = config.resolve_weight_path("mobile_sam.pt")
        try:
            from ultralytics import SAM

            _predictor = SAM(str(path))
            logger.info("MobileSAM 已加载: %s", path)
            return _predictor
        except Exception as exc:
            raise RuntimeError(
                f"加载 MobileSAM 失败: {exc}。请确认 weights/mobile_sam.pt 已放置。"
            ) from exc


def _cache_get(image_id: str):
    if image_id in _embed_cache:
        _embed_cache.move_to_end(image_id)
        return _embed_cache[image_id]
    return None


def _cache_put(image_id: str, value: Any) -> None:
    _embed_cache[image_id] = value
    _embed_cache.move_to_end(image_id)
    while len(_embed_cache) > _MAX:
        _embed_cache.popitem(last=False)


def _load_rgb(image_id: str) -> tuple[np.ndarray, int, int]:
    cached = _cache_get(image_id)
    if cached is not None:
        arr, w, h = cached
        return arr, w, h
    img = dataset_svc.get_image(image_id)
    path = dataset_svc.image_file_path(img)
    if not path.is_file():
        raise FileNotFoundError("原图丢失")
    with Image.open(path) as im:
        w, h = im.size
        arr = np.array(im.convert("RGB"))
    _cache_put(image_id, (arr, w, h))
    return arr, w, h


def predict_box_from_points(
    image_id: str,
    points: list[list[float]],
    labels: list[int],
) -> dict:
    """点选返回归一化外接框。"""
    if not points:
        raise ValueError("至少点一个前景点")

    arr, w, h = _load_rgb(image_id)
    model = _get_predictor()
    pts = [[float(p[0]), float(p[1])] for p in points]
    labs = [int(x) for x in (labels or [1] * len(pts))]
    if len(labs) < len(pts):
        labs = labs + [1] * (len(pts) - len(labs))

    results = model.predict(
        source=arr,
        points=pts,
        labels=labs,
        device="cpu",
        verbose=False,
    )
    if not results:
        raise RuntimeError("SAM 未返回结果")
    r = results[0]
    score = 0.9
    if r.masks is not None and len(r.masks.data) > 0:
        mask = r.masks.data[0].cpu().numpy()
        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0:
            raise RuntimeError("未分割到目标，请换个位置再点")
        mh, mw = mask.shape[-2:]
        scale_x = w / max(mw, 1)
        scale_y = h / max(mh, 1)
        x1 = float(xs.min()) * scale_x
        x2 = float(xs.max()) * scale_x
        y1 = float(ys.min()) * scale_y
        y2 = float(ys.max()) * scale_y
    elif r.boxes is not None and len(r.boxes) > 0:
        x1, y1, x2, y2 = map(float, r.boxes.xyxy[0].cpu().numpy())
        if r.boxes.conf is not None:
            score = float(r.boxes.conf[0].cpu().numpy())
    else:
        raise RuntimeError("未分割到目标，请换个位置再点")

    cx = ((x1 + x2) / 2.0) / max(w, 1)
    cy = ((y1 + y2) / 2.0) / max(h, 1)
    bw = max(0.0, x2 - x1) / max(w, 1)
    bh = max(0.0, y2 - y1) / max(h, 1)
    return {
        "box": {
            "cx": cx,
            "cy": cy,
            "w": bw,
            "h": bh,
            "conf": score,
            "source": "sam",
        },
        "score": score,
    }


def prefetch_next(image_id: str) -> None:
    """后台预热下一张图的 RGB 缓存，失败忽略。"""
    try:
        _load_rgb(image_id)
    except Exception as exc:
        logger.debug("SAM prefetch 跳过 %s: %s", image_id, exc)
