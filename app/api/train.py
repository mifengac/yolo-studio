"""训练 API。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from app.schemas import ModelRegisterRequest, TrainCreateRequest, TrainEstimateRequest
from app.services import export_svc, train_svc

router = APIRouter(prefix="/api/train", tags=["train"])


@router.post("/estimate")
def estimate(body: TrainEstimateRequest):
    try:
        report = export_svc.validate_for_train(
            body.dataset_id,
            only_confirmed=body.only_confirmed,
            min_boxes_per_class=1,
        )
    except (LookupError, ValueError) as e:
        raise HTTPException(400, str(e)) from e
    n = max(1, report["image_count"])
    cache = train_svc.choose_cache(n, body.imgsz)
    try:
        base_path, _, _ = train_svc.resolve_base_model(body.base_model)
        base_name = str(base_path)
    except Exception:
        base_name = body.base_model
    est = train_svc.estimate_seconds(
        num_images=n,
        epochs=body.epochs,
        imgsz=body.imgsz,
        base_model=base_name,
        freeze=body.freeze,
        cache=cache,
    )
    return {
        "image_count": n,
        "class_counts": report["class_counts"],
        "warnings": report.get("warnings") or [],
        **est,
    }


@router.post("")
def create_train(body: TrainCreateRequest):
    try:
        return train_svc.create_train_job(body.model_dump())
    except (ValueError, LookupError, FileNotFoundError, RuntimeError) as e:
        raise HTTPException(400, str(e)) from e


@router.get("")
def list_jobs():
    return train_svc.list_jobs()


@router.get("/{job_id}")
def get_job(job_id: str):
    try:
        return train_svc.get_job(job_id)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e


@router.get("/{job_id}/logs")
async def logs_sse(job_id: str):
    try:
        job = train_svc.get_job(job_id)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e

    log_path = Path(job["log_path"])

    async def gen():
        pos = 0
        idle = 0
        while True:
            if log_path.is_file():
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
                if chunk:
                    for line in chunk.splitlines():
                        yield f"data: {line}\n\n"
                    idle = 0
                else:
                    idle += 1
            else:
                idle += 1
            # 结束后再冲一会儿
            j = train_svc.get_job(job_id)
            if j["status"] not in ("running", "pending") and idle > 4:
                yield f"data: [status] {j['status']}\n\n"
                break
            await asyncio.sleep(1)

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/{job_id}/metrics")
def metrics(job_id: str):
    try:
        return {"series": train_svc.read_metrics_series(job_id)}
    except LookupError as e:
        raise HTTPException(404, str(e)) from e


@router.post("/{job_id}/cancel")
def cancel(job_id: str):
    try:
        return train_svc.cancel_job(job_id)
    except (LookupError, ValueError) as e:
        raise HTTPException(400, str(e)) from e


@router.post("/{job_id}/resume")
def resume(job_id: str):
    try:
        return train_svc.resume_job(job_id)
    except (LookupError, ValueError, FileNotFoundError, RuntimeError) as e:
        raise HTTPException(400, str(e)) from e


@router.post("/{job_id}/publish")
def publish(job_id: str, body: ModelRegisterRequest | None = None):
    body = body or ModelRegisterRequest()
    try:
        return train_svc.publish_job(job_id, name=body.name, notes=body.notes)
    except (LookupError, ValueError) as e:
        raise HTTPException(400, str(e)) from e


@router.get("/{job_id}/artifacts/{filename}")
def artifact(job_id: str, filename: str):
    try:
        job = train_svc.get_job(job_id)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(400, "非法文件名")
    # 常见产物位置
    candidates = [
        Path(job["run_dir"]) / "weights" / filename,
        Path(job["run_dir"]) / filename,
    ]
    for p in candidates:
        if p.is_file():
            return FileResponse(p, filename=filename)
    raise HTTPException(404, "文件不存在")
