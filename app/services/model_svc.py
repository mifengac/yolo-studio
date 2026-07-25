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


def _normalize_ov_status(d: dict) -> dict:
    """旧数据兼容：有路径当 ready，否则 pending。"""
    st = d.get("ov_status")
    if st in ("pending", "exporting", "ready", "failed"):
        return d
    if d.get("openvino_path"):
        d["ov_status"] = "ready"
    else:
        d["ov_status"] = "pending"
    return d


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
        out.append(_normalize_ov_status(d))
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
    return _normalize_ov_status(d)


def get_default_autolabel_path() -> Optional[str]:
    """仅返回模型仓库里显式设为默认的路径；不再回退 0517 等业务权重。

    错误兜底会把翘车头模型套到头盔等新场景上，静默污染标注，危害大于无模型。
    """
    with db._lock, db.connect() as conn:
        row = conn.execute(
            "SELECT path FROM model WHERE is_default_autolabel=1 LIMIT 1"
        ).fetchone()
    if row:
        return row["path"]
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
    ov_status = "pending"
    export_dynamic = 1
    if export_ov:
        ov_path = _do_export_openvino(mid, dest, imgsz=imgsz, dynamic=True)
        ov_status = "ready" if ov_path else "failed"

    now = db.utcnow()
    with db._lock, db.connect() as conn:
        conn.execute(
            """INSERT INTO model
               (id, name, path, classes, metrics, from_job, notes,
                is_default_autolabel, openvino_path, ov_status,
                export_imgsz, export_dynamic, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                ov_status,
                imgsz if ov_path or schedule_openvino else None,
                export_dynamic if ov_path or schedule_openvino else None,
                now,
            ),
        )
    if schedule_openvino and ov_status == "pending":
        try:
            from app import tasks as task_mod

            task_mod.create_task(
                "openvino_export",
                {"model_id": mid, "imgsz": imgsz, "dynamic": True},
                submit=True,
            )
        except Exception as exc:
            logger.warning("提交 OpenVINO 导出任务失败（模型仍可用）: %s", exc)
            with db._lock, db.connect() as conn:
                conn.execute(
                    "UPDATE model SET ov_status='failed', notes=? WHERE id=?",
                    ((notes or "") + f"\nOpenVINO 任务提交失败: {exc}", mid),
                )
    return get_model(mid)


def _do_export_openvino(
    model_id: str, pt_path: Path, imgsz: int = 416, *, dynamic: bool = True
) -> Optional[str]:
    ov = engine.export_openvino(pt_path, imgsz=imgsz, dynamic=dynamic)
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
    dynamic = bool(params.get("dynamic", True))
    task_id = task["id"]
    if not model_id:
        raise ValueError("缺少 model_id")
    m = get_model(model_id)
    with db._lock, db.connect() as conn:
        conn.execute(
            "UPDATE model SET ov_status='exporting', export_imgsz=?, export_dynamic=? WHERE id=?",
            (imgsz, 1 if dynamic else 0, model_id),
        )
    task_mod.set_progress(task_id, 10, f"正在导出 OpenVINO：{m['name']}（dynamic={dynamic}）")
    ov_path = _do_export_openvino(
        model_id, Path(m["path"]), imgsz=imgsz, dynamic=dynamic
    )
    if ov_path:
        with db._lock, db.connect() as conn:
            conn.execute(
                """UPDATE model SET openvino_path=?, ov_status='ready',
                   export_imgsz=?, export_dynamic=? WHERE id=?""",
                (ov_path, imgsz, 1 if dynamic else 0, model_id),
            )
        task_mod.set_progress(task_id, 100, f"OpenVINO 导出完成：{ov_path}")
    else:
        with db._lock, db.connect() as conn:
            conn.execute(
                """UPDATE model SET ov_status='failed',
                   notes=COALESCE(notes,'') || ? WHERE id=?""",
                ("\nOpenVINO 导出失败，继续用 .pt", model_id),
            )
        task_mod.update_task(
            task_id,
            message="OpenVINO 导出失败，继续使用 .pt 推理",
        )
        logger.warning("模型 %s OpenVINO 导出失败，已降级为 .pt", model_id)


def prefer_openvino_for_path(pt_path: str | Path) -> bool:
    """仅当注册模型 ov_status=ready（或未注册但目录存在）时优先 OpenVINO。"""
    pt = Path(pt_path).resolve()
    with db._lock, db.connect() as conn:
        row = conn.execute(
            "SELECT ov_status, openvino_path FROM model WHERE path=?",
            (str(pt),),
        ).fetchone()
    if row:
        st = row["ov_status"]
        if st is None or st == "":
            return bool(row["openvino_path"])
        return st == "ready"
    # 未注册的预置权重：有导出目录才用
    return engine._find_openvino_dir(pt) is not None


def _parse_ov_metadata(ov_dir: str | Path) -> dict:
    """从 OpenVINO 导出目录的 metadata.yaml 读取 imgsz / dynamic。"""
    meta_path = Path(ov_dir) / "metadata.yaml"
    out: dict = {"export_imgsz": None, "export_dynamic": False}
    if not meta_path.is_file():
        return out
    try:
        import yaml

        data = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        imgsz = data.get("imgsz")
        if isinstance(imgsz, (list, tuple)) and imgsz:
            out["export_imgsz"] = int(imgsz[0])
        elif isinstance(imgsz, int):
            out["export_imgsz"] = imgsz
        args = data.get("args") or {}
        dyn = args.get("dynamic", data.get("dynamic", False))
        out["export_dynamic"] = bool(dyn)
    except Exception as exc:
        logger.warning("读 OpenVINO metadata 失败 %s: %s", meta_path, exc)
    return out


def get_export_meta_for_path(pt_path: str | Path) -> Optional[dict]:
    """返回注册模型的 export_imgsz / export_dynamic；未注册返回 None。

    若库内为空但磁盘有 OpenVINO 目录，会从 metadata.yaml 现读（并尽量回写库）。
    """
    pt = Path(pt_path).resolve()
    with db._lock, db.connect() as conn:
        row = conn.execute(
            """SELECT id, export_imgsz, export_dynamic, openvino_path, ov_status
               FROM model WHERE path=?""",
            (str(pt),),
        ).fetchone()
    if not row:
        return None
    d = {
        "export_imgsz": row["export_imgsz"],
        "export_dynamic": bool(row["export_dynamic"])
        if row["export_dynamic"] is not None
        else False,
        "openvino_path": row["openvino_path"],
        "ov_status": row["ov_status"],
    }
    # 旧记录 NULL：从磁盘 metadata 补全
    if d["export_imgsz"] is None and d.get("openvino_path"):
        parsed = _parse_ov_metadata(d["openvino_path"])
        if parsed.get("export_imgsz"):
            d["export_imgsz"] = parsed["export_imgsz"]
            d["export_dynamic"] = parsed.get("export_dynamic", False)
            try:
                with db._lock, db.connect() as conn:
                    conn.execute(
                        """UPDATE model SET export_imgsz=?, export_dynamic=?
                           WHERE id=? AND (export_imgsz IS NULL OR export_imgsz='')""",
                        (
                            d["export_imgsz"],
                            1 if d["export_dynamic"] else 0,
                            row["id"],
                        ),
                    )
            except Exception:
                pass
    return d


def backfill_export_meta_from_disk() -> int:
    """启动时：给 openvino 已 ready 但 export_imgsz 为空的旧模型回填元数据。"""
    with db._lock, db.connect() as conn:
        rows = conn.execute(
            """SELECT id, openvino_path FROM model
               WHERE openvino_path IS NOT NULL AND openvino_path != ''
                 AND (export_imgsz IS NULL)"""
        ).fetchall()
    n = 0
    for r in rows:
        parsed = _parse_ov_metadata(r["openvino_path"])
        if not parsed.get("export_imgsz"):
            continue
        with db._lock, db.connect() as conn:
            conn.execute(
                """UPDATE model SET export_imgsz=?, export_dynamic=? WHERE id=?""",
                (
                    int(parsed["export_imgsz"]),
                    1 if parsed.get("export_dynamic") else 0,
                    r["id"],
                ),
            )
        n += 1
        logger.info(
            "回填模型 %s export_imgsz=%s dynamic=%s",
            r["id"],
            parsed["export_imgsz"],
            parsed.get("export_dynamic"),
        )
    return n


def set_default_autolabel(model_id: str) -> dict:
    get_model(model_id)
    with db._lock, db.connect() as conn:
        conn.execute("UPDATE model SET is_default_autolabel=0")
        conn.execute(
            "UPDATE model SET is_default_autolabel=1 WHERE id=?", (model_id,)
        )
    return get_model(model_id)


def delete_model(model_id: str) -> None:
    """删库记录并清理 .pt 与 OpenVINO 目录，不留孤儿文件。"""
    m = get_model(model_id)
    with db._lock, db.connect() as conn:
        conn.execute("DELETE FROM model WHERE id=?", (model_id,))
    p = Path(m["path"])
    if p.is_file() and config.MODELS_DIR in p.parents:
        p.unlink(missing_ok=True)
    ov = m.get("openvino_path")
    if ov and Path(ov).is_dir():
        shutil.rmtree(ov, ignore_errors=True)
    # 规范目录名也可能存在（openvino_path 为空时）
    stem_ov = config.MODELS_DIR / f"{model_id}_openvino_model"
    if stem_ov.is_dir():
        shutil.rmtree(stem_ov, ignore_errors=True)
    # 按文件 stem 的导出目录
    if p.stem:
        alt = config.MODELS_DIR / f"{p.stem}_openvino_model"
        if alt.is_dir():
            shutil.rmtree(alt, ignore_errors=True)


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
        prefer_ov = prefer_openvino_for_path(m["path"]) and m.get("ov_status") == "ready"
        preds = engine.predict_boxes_batch(
            m["path"], [str(path)], conf=conf, imgsz=imgsz, prefer_openvino=prefer_ov
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
