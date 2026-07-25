"""MobileSAM 点选辅助：缓存 image embedding（非 RGB），二次点选复用 encoder。"""

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
# image_id -> {features, orig_hw, im_tensor 引用所需状态}
# 每条约数十～上百 MB，默认最多 4 条（config.SAM_EMBED_CACHE_SIZE）
_embed_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
_MAX = config.SAM_EMBED_CACHE_SIZE
_current_image_id: Optional[str] = None
_prefetch_lock = threading.Lock()


def _get_predictor():
    """单例 SAMPredictor（ultralytics 8.4+）。"""
    global _predictor
    if _predictor is not None:
        return _predictor
    with _lock:
        if _predictor is not None:
            return _predictor
        config.apply_torch_threads()
        path = config.resolve_weight_path("mobile_sam.pt")
        try:
            from ultralytics.models.sam import Predictor as SAMPredictor

            overrides = {
                "model": str(path),
                "task": "segment",
                "mode": "predict",
                "imgsz": 1024,
                "device": "cpu",
                "verbose": False,
                "save": False,
            }
            pred = SAMPredictor(overrides=overrides)
            # model=None 时内部 get_model() 按 overrides['model'] 加载权重
            pred.setup_model(verbose=False)
            _predictor = pred
            logger.info("MobileSAM Predictor 已加载: %s", path)
            return _predictor
        except Exception as exc:
            raise RuntimeError(
                f"加载 MobileSAM 失败: {exc}。请确认 weights/mobile_sam.pt 已放置。"
            ) from exc


def _cache_get(image_id: str) -> Optional[dict[str, Any]]:
    if image_id in _embed_cache:
        _embed_cache.move_to_end(image_id)
        return _embed_cache[image_id]
    return None


def _cache_put(image_id: str, value: dict[str, Any]) -> None:
    _embed_cache[image_id] = value
    _embed_cache.move_to_end(image_id)
    while len(_embed_cache) > _MAX:
        _embed_cache.popitem(last=False)


def _load_bgr(path: Path) -> tuple[np.ndarray, int, int]:
    """返回 BGR（cv2/ultralytics set_image 约定）与宽高。"""
    with Image.open(path) as im:
        w, h = im.size
        rgb = np.array(im.convert("RGB"))
    # RGB -> BGR
    bgr = rgb[:, :, ::-1].copy()
    return bgr, w, h


def _ensure_image_features(image_id: str) -> tuple[Any, int, int]:
    """确保 predictor 上已有该图的 embedding；返回 (predictor, w, h)。"""
    global _current_image_id
    pred = _get_predictor()
    img = dataset_svc.get_image(image_id)
    path = dataset_svc.image_file_path(img)
    if not path.is_file():
        raise FileNotFoundError("原图丢失")

    with _lock:
        cached = _cache_get(image_id)
        if cached is not None and cached.get("features") is not None:
            # 恢复 embedding，跳过昂贵 encoder
            pred.features = cached["features"]
            if cached.get("batch") is not None:
                pred.batch = cached["batch"]
            _current_image_id = image_id
            return pred, cached["w"], cached["h"]

        if _current_image_id == image_id and getattr(pred, "features", None) is not None:
            # 当前图已 set，从 batch 取尺寸
            try:
                oh, ow = pred.batch[1][0].shape[:2]
                return pred, int(ow), int(oh)
            except Exception:
                pass

        bgr, w, h = _load_bgr(path)
        pred.set_image(bgr)
        _current_image_id = image_id
        # 缓存 features（真正的瓶颈产物）
        try:
            feat = pred.features
            batch = getattr(pred, "batch", None)
            _cache_put(
                image_id,
                {
                    "features": feat,
                    "batch": batch,
                    "w": w,
                    "h": h,
                },
            )
        except Exception as exc:
            logger.warning("缓存 SAM embedding 失败（不影响本次点选）: %s", exc)
        return pred, w, h


def predict_box_from_points(
    image_id: str,
    points: list[list[float]],
    labels: list[int],
) -> dict:
    """点选返回归一化外接框。同一张图第二次起复用 embedding。"""
    if not points:
        raise ValueError("至少点一个前景点")

    pred, w, h = _ensure_image_features(image_id)
    pts = [[float(p[0]), float(p[1])] for p in points]
    labs = [int(x) for x in (labels or [1] * len(pts))]
    if len(labs) < len(pts):
        labs = labs + [1] * (len(pts) - len(labs))

    with _lock:
        # 确保 features 仍在（多线程切换图时可能被换掉）
        cached = _cache_get(image_id)
        if cached and cached.get("features") is not None:
            pred.features = cached["features"]
            if cached.get("batch") is not None:
                pred.batch = cached["batch"]
        # ultralytics：set_image 后用 points/labels 调用，内部用 self.features
        results = pred(points=np.array(pts), labels=np.array(labs))

    if not results:
        raise RuntimeError("SAM 未返回结果")
    r = results[0] if isinstance(results, (list, tuple)) else results
    # Results 可能是 list
    if isinstance(r, list):
        if not r:
            raise RuntimeError("SAM 未返回结果")
        r = r[0]

    score = 0.9
    if getattr(r, "masks", None) is not None and len(r.masks.data) > 0:
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
    elif getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
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
    """后台线程预热下一张 embedding，不阻塞当前请求。"""

    def _job() -> None:
        if not _prefetch_lock.acquire(blocking=False):
            return
        try:
            if not (config.WEIGHTS_DIR / "mobile_sam.pt").is_file():
                return
            _ensure_image_features(image_id)
        except Exception as exc:
            logger.debug("SAM prefetch 跳过 %s: %s", image_id, exc)
        finally:
            _prefetch_lock.release()

    t = threading.Thread(target=_job, name="sam-prefetch", daemon=True)
    t.start()
