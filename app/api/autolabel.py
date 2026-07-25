"""自动标注 API。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas import AutolabelRequest, OpenvocabRequest, TrackRequest
from app.services import dataset_svc, model_svc
from app import tasks as task_mod

router = APIRouter(prefix="/api", tags=["autolabel"])


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
                "没有可用的默认预标注模型。请先在「模型仓库」设默认，"
                "或把 0517_yolo26s/n_wheelie_multi-rider.pt 放到 weights/。",
            )
        model = resolved

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
