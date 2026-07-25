"""YOLO-World 开放词表零样本预标注。"""

from __future__ import annotations

import logging

from app import config, db
from app.infer import engine
from app.services import dataset_svc
from app.services.autolabel_svc import uncertainty_of_boxes

logger = logging.getLogger(__name__)


def run_openvocab(task: dict) -> None:
    from app import tasks as task_mod

    params = task.get("params") or {}
    dataset_id = task.get("dataset_id") or params.get("dataset_id")
    task_id = task["id"]
    ds = dataset_svc.get_dataset(dataset_id)
    prompts = params.get("prompts") or []
    if not prompts:
        raise ValueError("至少填写一个开放词表提示词")
    conf = float(params.get("conf", 0.1))
    imgsz = int(params.get("imgsz", 640))
    scope = params.get("scope") or "unlabeled"
    overwrite = bool(params.get("overwrite", False))
    preview_limit = params.get("preview_limit")
    class_map = params.get("class_map") or {}
    # class_map: {0: 0, 1: 1} prompt index -> dataset class index
    class_map = {int(k): int(v) for k, v in class_map.items()}

    model_path = config.resolve_weight_path(
        params.get("model") or "yolov8s-worldv2.pt"
    )

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

    if preview_limit:
        items = items[: int(preview_limit)]
    if len(items) > config.OPENVOCAB_MAX_IMAGES and not preview_limit:
        items = items[: config.OPENVOCAB_MAX_IMAGES]
        logger.warning(
            "开放词表单次限制 %s 张，已截断", config.OPENVOCAB_MAX_IMAGES
        )

    total = len(items)
    if total == 0:
        task_mod.set_progress(task_id, 100, "没有需要处理的图片")
        return

    # 加载 YOLO-World 并设置类别（依赖 clip + ~/.cache/clip/ViT-B-32.pt）
    model = engine.load_model(model_path, prefer_openvino=False)
    if not hasattr(model, "set_classes"):
        raise RuntimeError(
            "当前模型不支持 set_classes，请确认 weights/yolov8s-worldv2.pt 是 YOLO-World 权重"
        )
    try:
        model.set_classes(list(prompts))
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "开放词表需要 CLIP 库。有网时执行："
            "`pip install git+https://github.com/ultralytics/CLIP.git`，"
            "并把 ViT-B-32.pt（约 338MB）放到 ~/.cache/clip/ "
            "（或 weights/clip/ 后在镜像里拷到该路径）。"
            f" 原始错误: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"设置开放词表提示词失败: {e}。"
            "请确认已离线放置 CLIP 权重 ViT-B-32.pt（见 docs/20260725_离线权重清单.md）。"
        ) from e

    labeled = 0
    batch = max(1, min(4, config.INFER_BATCH_SIZE))  # 开放词表更重
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
        results = model.predict(
            source=paths,
            conf=conf,
            imgsz=imgsz,
            device="cpu",
            half=False,
            verbose=False,
        )
        for it, r in zip(valid, results):
            mapped = []
            if r.boxes is not None and len(r.boxes) > 0:
                h, w = r.orig_shape[:2]
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                for j in range(len(xyxy)):
                    pi = int(clss[j])
                    di = class_map.get(pi, pi if pi < len(ds["classes"]) else None)
                    if di is None or di < 0 or di >= len(ds["classes"]):
                        continue
                    x1, y1, x2, y2 = map(float, xyxy[j])
                    cx = (x1 + x2) / 2.0 / max(w, 1)
                    cy = (y1 + y2) / 2.0 / max(h, 1)
                    nw = max(0.0, x2 - x1) / max(w, 1)
                    nh = max(0.0, y2 - y1) / max(h, 1)
                    mapped.append(
                        {
                            "class_idx": int(di),
                            "cx": cx,
                            "cy": cy,
                            "w": nw,
                            "h": nh,
                            "conf": float(confs[j]),
                            "source": "openvocab",
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
            task_id, pct, f"开放词表预标注 {min(i + batch, total)}/{total}"
        )

    task_mod.update_task(task_id, message=f"完成：处理 {labeled} 张")
