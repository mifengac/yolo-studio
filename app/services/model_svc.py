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
    export_ov: bool = True,
    imgsz: int = 416,
) -> dict:
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"模型文件不存在: {path}")
    mid = _new_id()
    dest = config.MODELS_DIR / f"{mid}.pt"
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    ov_path = None
    if export_ov:
        ov = engine.export_openvino(dest, imgsz=imgsz)
        if ov:
            # 规范名
            target = config.MODELS_DIR / f"{mid}_openvino_model"
            if ov.resolve() != target.resolve():
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                if ov.is_dir():
                    shutil.copytree(ov, target)
                    ov_path = str(target)
            else:
                ov_path = str(target)

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
    return get_model(mid)


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
