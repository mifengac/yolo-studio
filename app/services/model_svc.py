"""模型仓库：注册、默认预标注、OpenVINO 导出、评估。"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from app import config, db
from app.infer import engine
from app.services import dataset_svc
from app.services.autolabel_svc import resolve_class_index

logger = logging.getLogger(__name__)


def _new_id() -> str:
    return f"mdl_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"


def list_models() -> list[dict]:
    with db._lock, db.connect() as conn:
        rows = conn.execute(
            "SELECT * FROM model ORDER BY created_at DESC"
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["classes"] = db.loads_json(d.get("classes"), [])
        d["metrics"] = db.loads_json(d.get("metrics"), {})
        d["is_default_autolabel"] = bool(d.get("is_default_autolabel"))
        out.append(d)
    return out


def get_model(model_id: str) -> dict:
    with db._lock, db.connect() as conn:
        row = conn.execute("SELECT * FROM model WHERE id=?", (model_id,)).fetchone()
    if not row:
        raise LookupError("模型不存在")
    d = dict(row)
    d["classes"] = db.loads_json(d.get("classes"), [])
    d["metrics"] = db.loads_json(d.get("metrics"), {})
    d["is_default_autolabel"] = bool(d.get("is_default_autolabel"))
    return d


def get_default_autolabel_path() -> Optional[str]:
    with db._lock, db.connect() as conn:
        row = conn.execute(
            "SELECT path FROM model WHERE is_default_autolabel=1 LIMIT 1"
        ).fetchone()
    if row:
        return row["path"]
    # 回退预置
    for key in (
        "0517_yolo26s_wheelie_multi-rider.pt",
        "0517_yolo26n_wheelie_multi-rider.pt",
    ):
        p = config.WEIGHTS_DIR / key
        if p.is_file():
            return str(p)
    return None


def register_model(
    *,
    path: str,
    name: str,
    classes: list[str],
    metrics: Optional[dict] = None,
    from_job: Optional[str] = None,
    notes: str = "",
    export_ov: bool = False,
    schedule_openvino: bool = True,
    imgsz: int = 416,
) -> dict:
    """注册模型。默认不同步导出 OpenVINO（CPU 上很慢），由后台任务异步导出。"""
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"模型文件不存在: {path}")
    mid = _new_id()
    dest = config.MODELS_DIR / f"{mid}.pt"
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    ov_path = None
    if export_ov:
        ov_path = _do_export_openvino(mid, dest, imgsz=imgsz)

    now = db.utcnow()
    with db._lock, db.connect() as conn:
        conn.execute(
            """INSERT INTO model
               (id, name, path, classes, metrics, from_job, notes,
                is_default_autolabel, openvino_path, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                mid,
                name,
                str(dest),
                db.dumps_json(classes),
                db.dumps_json(metrics or {}),
                from_job,
                notes,
                0,
                ov_path,
                now,
            ),
        )
    if schedule_openvino and not ov_path:
        try:
            from app import tasks as task_mod

            task_mod.create_task(
                "openvino_export",
                {"model_id": mid, "imgsz": imgsz},
                submit=True,
            )
        except Exception as exc:
            logger.warning("提交 OpenVINO 导出任务失败（模型仍可用）: %s", exc)
    return get_model(mid)


def _do_export_openvino(model_id: str, pt_path: Path, imgsz: int = 416) -> Optional[str]:
    ov = engine.export_openvino(pt_path, imgsz=imgsz)
    if not ov:
        return None
    target = config.MODELS_DIR / f"{model_id}_openvino_model"
    if ov.resolve() != target.resolve():
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        if ov.is_dir():
            shutil.copytree(ov, target)
            return str(target)
        return None
    return str(target)


def run_openvino_export(task: dict) -> None:
    """后台任务：为已注册模型导出 OpenVINO，失败只记日志不让模型不可用。"""
    from app import tasks as task_mod

    params = task.get("params") or {}
    model_id = params.get("model_id")
    imgsz = int(params.get("imgsz", 416))
    task_id = task["id"]
    if not model_id:
        raise ValueError("缺少 model_id")
    m = get_model(model_id)
    task_mod.set_progress(task_id, 10, f"正在导出 OpenVINO：{m['name']}")
    ov_path = _do_export_openvino(model_id, Path(m["path"]), imgsz=imgsz)
    if ov_path:
        with db._lock, db.connect() as conn:
            conn.execute(
                "UPDATE model SET openvino_path=? WHERE id=?",
                (ov_path, model_id),
            )
        task_mod.set_progress(task_id, 100, f"OpenVINO 导出完成：{ov_path}")
    else:
        task_mod.update_task(
            task_id,
            message="OpenVINO 导出失败，继续使用 .pt 推理",
        )
        logger.warning("模型 %s OpenVINO 导出失败，已降级为 .pt", model_id)


def set_default_autolabel(model_id: str) -> dict:
    get_model(model_id)
    with db._lock, db.connect() as conn:
        conn.execute("UPDATE model SET is_default_autolabel=0")
        conn.execute(
            "UPDATE model SET is_default_autolabel=1 WHERE id=?", (model_id,)
        )
    return get_model(model_id)


def delete_model(model_id: str) -> None:
    m = get_model(model_id)
    with db._lock, db.connect() as conn:
        conn.execute("DELETE FROM model WHERE id=?", (model_id,))
    p = Path(m["path"])
    if p.is_file() and config.MODELS_DIR in p.parents:
        p.unlink(missing_ok=True)
    ov = m.get("openvino_path")
    if ov and Path(ov).is_dir():
        shutil.rmtree(ov, ignore_errors=True)


def evaluate_model(model_id: str, dataset_id: str, conf: float = 0.25, imgsz: int = 416) -> dict:
    """简易评估：在 confirmed 图上统计召回粗指标（框数级）。"""
    m = get_model(model_id)
    ds = dataset_svc.get_dataset(dataset_id)
    page = 1
    items = []
    while True:
        chunk = dataset_svc.list_images(
            dataset_id, status="confirmed", page=page, page_size=200
        )
        if not chunk["items"]:
            break
        items.extend(chunk["items"])
        if page * 200 >= chunk["total"]:
            break
        page += 1
    if not items:
        raise ValueError("该数据集没有已确认图片，无法评估")

    gt_total = 0
    pred_total = 0
    matched = 0
    for it in items:
        gts = dataset_svc.get_annotations(it["id"])
        gt_total += len(gts)
        path = dataset_svc.image_file_path(it)
        if not path.is_file():
            continue
        preds = engine.predict_boxes_batch(
            m["path"], [str(path)], conf=conf, imgsz=imgsz, prefer_openvino=True
        )[0]
        mapped = []
        for b in preds:
            ci = resolve_class_index(
                b["class_idx"], b["class_name"], ds["classes"], {}
            )
            if ci is not None:
                mapped.append(b)
        pred_total += len(mapped)
        # 极简：按类别数量取 min 作为匹配近似
        from collections import Counter

        gc = Counter(int(g["class_idx"]) for g in gts)
        pc = Counter()
        for b in mapped:
            ci = resolve_class_index(
                b["class_idx"], b["class_name"], ds["classes"], {}
            )
            if ci is not None:
                pc[ci] += 1
        for k in gc:
            matched += min(gc[k], pc.get(k, 0))

    precision = matched / pred_total if pred_total else 0.0
    recall = matched / gt_total if gt_total else 0.0
    result = {
        "images": len(items),
        "gt_boxes": gt_total,
        "pred_boxes": pred_total,
        "matched_approx": matched,
        "precision_approx": round(precision, 4),
        "recall_approx": round(recall, 4),
        "note": "粗略框数级评估，非标准 mAP；正式指标以训练 val 为准",
    }
    return result


def list_base_models() -> dict:
    """前端底模选择：从零开始 + 已有模型微调。"""
    scratch = []
    for name in ("yolo26n.pt", "yolo26s.pt"):
        p = config.WEIGHTS_DIR / name
        scratch.append(
            {
                "id": name,
                "name": name,
                "path": str(p) if p.is_file() else None,
                "available": p.is_file(),
                "group": "scratch",
                "label": f"从零开始 · {name}",
            }
        )
    finetune = []
    for name in (
        "0517_yolo26n_wheelie_multi-rider.pt",
        "0517_yolo26s_wheelie_multi-rider.pt",
    ):
        p = config.WEIGHTS_DIR / name
        if p.is_file():
            finetune.append(
                {
                    "id": name,
                    "name": name,
                    "path": str(p),
                    "available": True,
                    "group": "finetune",
                    "label": f"已有模型微调 · {name}",
                    "classes": ["wheelie", "multi rider"],
                }
            )
    for m in list_models():
        finetune.append(
            {
                "id": m["id"],
                "name": m["name"],
                "path": m["path"],
                "available": True,
                "group": "finetune",
                "label": f"已有模型微调 · {m['name']}",
                "classes": m["classes"],
            }
        )
    return {"scratch": scratch, "finetune": finetune, "default_group": "finetune"}
