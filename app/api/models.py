"""模型仓库 API。"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.schemas import EvaluateRequest
from app.services import model_svc

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("")
def list_models():
    return model_svc.list_models()


@router.get("/base-options")
def base_options():
    return model_svc.list_base_models()


@router.get("/{model_id}")
def get_model(model_id: str):
    try:
        return model_svc.get_model(model_id)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e


@router.post("/{model_id}/set-default-autolabel")
def set_default(model_id: str):
    try:
        return model_svc.set_default_autolabel(model_id)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e


@router.post("/{model_id}/evaluate")
def evaluate(model_id: str, body: EvaluateRequest):
    try:
        return model_svc.evaluate_model(
            model_id, body.dataset_id, conf=body.conf, imgsz=body.imgsz
        )
    except (LookupError, ValueError) as e:
        raise HTTPException(400, str(e)) from e


@router.get("/{model_id}/download")
def download(model_id: str):
    try:
        m = model_svc.get_model(model_id)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    p = Path(m["path"])
    if not p.is_file():
        raise HTTPException(404, "模型文件丢失")
    return FileResponse(p, filename=p.name)


@router.delete("/{model_id}")
def delete(model_id: str):
    try:
        model_svc.delete_model(model_id)
        return {"ok": True}
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
