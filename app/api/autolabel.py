"""自动标注 API。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas import AutolabelRequest, OpenvocabRequest, TrackRequest
from app.services import dataset_svc, model_svc
from app import tasks as task_mod

router = APIRouter(prefix="/api", tags=["autolabel"])


@router.get("/datasets/{dataset_id}/autolabel-options")
def autolabel_options(dataset_id: str):
    """预标注可选模型：本数据集训练过的排最前，并标注各类别。"""
    try:
        ds = dataset_svc.get_dataset(dataset_id)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    from app import db
    from app.services.autolabel_svc import normalize_token

    ds_cls = ds["classes"]
    ds_norm = {normalize_token(c) for c in ds_cls}
    # from_job -> dataset_id
    job_ds: dict[str, str] = {}
    with db._lock, db.connect() as conn:
        for r in conn.execute("SELECT id, dataset_id FROM train_job").fetchall():
            job_ds[r["id"]] = r["dataset_id"]
    options = []
    default_path = model_svc.get_default_autolabel_path()
    for m in model_svc.list_models():
        m_cls = m.get("classes") or []
        m_norm = {normalize_token(c) for c in m_cls}
        hit = len(ds_norm & m_norm)
        from_job = m.get("from_job") or ""
        same_classes = m_norm == ds_norm and len(m_norm) > 0
        is_own = (from_job and job_ds.get(from_job) == dataset_id) or same_classes
        label_cls = "/".join(m_cls) if m_cls else "未知类别"
        tag = "（本数据集训练）" if is_own else f"（{label_cls}）"
        options.append(
            {
                "id": m["id"],
                "path": m["path"],
                "name": m["name"],
                "classes": m_cls,
                "label": f"{m['name']}{tag}",
                "is_default": bool(m.get("is_default_autolabel")),
                "is_own_dataset": is_own,
                "class_hit": hit,
            }
        )
    options.sort(key=lambda x: (0 if x["is_own_dataset"] else 1, -x["class_hit"], x["name"]))
    return {
        "dataset_id": dataset_id,
        "dataset_classes": ds_cls,
        "default_path": default_path,
        "options": options,
    }


@router.post("/datasets/{dataset_id}/autolabel")
def autolabel(dataset_id: str, body: AutolabelRequest):
    try:
        dataset_svc.get_dataset(dataset_id)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e

    model = (body.model or "").strip()
    if not model or model in ("default", "auto"):
        resolved = model_svc.get_default_autolabel_path()
        if not resolved:
            raise HTTPException(
                400,
                "还没有设置默认预标注模型。"
                "请先到「训练」页把训练结果发布到模型仓库，"
                "再到「模型仓库」点「设为默认预标注模型」。"
                "（不会再自动使用 weights 里的 0517 业务模型，以免类别错配污染标注）",
            )
        model = resolved

    # 创建任务前做一次兼容校验，失败立即返回（不必等后台任务）
    try:
        from app import config as cfg
        from app.infer import engine
        from app.services.autolabel_svc import (
            check_autolabel_compat,
            parse_class_mapping,
        )

        ds = dataset_svc.get_dataset(dataset_id)
        path = cfg.resolve_weight_path(model)
        names = engine.get_model_class_names(path)
        names_dict = {i: n for i, n in enumerate(names)}
        cmap = parse_class_mapping(body.class_map or {}, ds["classes"])
        check_autolabel_compat(names_dict, ds["classes"], cmap)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e)) from e

    task = task_mod.create_task(
        "autolabel",
        {
            "dataset_id": dataset_id,
            "model": model,
            "conf": body.conf,
            "iou": body.iou,
            "imgsz": body.imgsz,
            "class_map": body.class_map,
            "scope": body.scope,
            "overwrite": body.overwrite,
        },
        dataset_id=dataset_id,
    )
    return task


@router.post("/datasets/{dataset_id}/autolabel/openvocab")
def openvocab(dataset_id: str, body: OpenvocabRequest):
    try:
        dataset_svc.get_dataset(dataset_id)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    if not body.prompts:
        raise HTTPException(400, "至少填写一个提示词")
    task = task_mod.create_task(
        "openvocab",
        {
            "dataset_id": dataset_id,
            "prompts": body.prompts,
            "class_map": {str(k): v for k, v in body.class_map.items()},
            "conf": body.conf,
            "imgsz": body.imgsz,
            "scope": body.scope,
            "overwrite": body.overwrite,
            "preview_limit": body.preview_limit,
        },
        dataset_id=dataset_id,
    )
    return task


@router.post("/datasets/{dataset_id}/track")
def track(dataset_id: str, body: TrackRequest):
    try:
        dataset_svc.get_dataset(dataset_id)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    model = (body.model or "").strip()
    if not model or model in ("default", "auto"):
        model = model_svc.get_default_autolabel_path() or model
        if not model or model in ("default", "auto"):
            raise HTTPException(
                400,
                "没有可用的跟踪/预标注模型。请在模型仓库设默认，或放置 0517 权重到 weights/。",
            )
    task = task_mod.create_task(
        "track",
        {
            "dataset_id": dataset_id,
            "start_image_id": body.start_image_id,
            "max_frames": body.max_frames,
            "model": model,
            "conf": body.conf,
        },
        dataset_id=dataset_id,
    )
    return task


@router.post("/datasets/{dataset_id}/dedup")
def dedup(dataset_id: str):
    try:
        dataset_svc.get_dataset(dataset_id)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    task = task_mod.create_task(
        "dedup", {"dataset_id": dataset_id, "threshold": 5}, dataset_id=dataset_id
    )
    return task


@router.get("/tasks/{task_id}")
def get_task(task_id: str):
    t = task_mod.get_task(task_id)
    if not t:
        raise HTTPException(404, "任务不存在")
    return t


@router.post("/tasks/{task_id}/cancel")
def cancel_task(task_id: str):
    """取消排队/运行中的后台任务（切图等循环会检查 status）。"""
    t = task_mod.get_task(task_id)
    if not t:
        raise HTTPException(404, "任务不存在")
    if t.get("status") in ("success", "failed", "canceled"):
        return t
    from app import db

    task_mod.update_task(
        task_id,
        status="canceled",
        message="用户取消",
        finished_at=db.utcnow(),
    )
    return task_mod.get_task(task_id)
