"""数据集 API。"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app import config
from app.schemas import DatasetCreate, DatasetUpdate
from app.services import dataset_svc, track_svc

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.post("")
def create_dataset(body: DatasetCreate):
    try:
        return dataset_svc.create_dataset(body.name, body.classes, body.notes)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.get("")
def list_datasets():
    return dataset_svc.list_datasets()


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str):
    try:
        return dataset_svc.get_dataset(dataset_id)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e


@router.patch("/{dataset_id}")
def update_dataset(dataset_id: str, body: DatasetUpdate):
    try:
        return dataset_svc.update_dataset(
            dataset_id, name=body.name, classes=body.classes, notes=body.notes
        )
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str):
    try:
        dataset_svc.delete_dataset(dataset_id)
        return {"ok": True}
    except LookupError as e:
        raise HTTPException(404, str(e)) from e


@router.post("/{dataset_id}/import/files")
async def import_files(dataset_id: str, files: list[UploadFile] = File(...)):
    try:
        dataset_svc.get_dataset(dataset_id)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    blobs = []
    total = 0
    for f in files:
        data = await f.read()
        total += len(data)
        if total > config.MAX_UPLOAD_BYTES:
            raise HTTPException(400, f"单次上传超过 {config.MAX_UPLOAD_MB} MB 限制")
        blobs.append((f.filename or "image.jpg", data))
    try:
        return dataset_svc.import_files(dataset_id, blobs)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.post("/{dataset_id}/import/zip")
async def import_zip(dataset_id: str, file: UploadFile = File(...)):
    try:
        dataset_svc.get_dataset(dataset_id)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    data = await file.read()
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"ZIP 超过 {config.MAX_UPLOAD_MB} MB 限制")
    try:
        return dataset_svc.import_zip(dataset_id, data)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.post("/{dataset_id}/import/video")
async def import_video(
    dataset_id: str, file: UploadFile = File(...), fps: float = 2.0
):
    try:
        dataset_svc.get_dataset(dataset_id)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    data = await file.read()
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"视频超过 {config.MAX_UPLOAD_MB} MB 限制")
    try:
        return track_svc.import_video(
            dataset_id, data, file.filename or "video.mp4", fps=fps
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.get("/{dataset_id}/images")
def list_images(
    dataset_id: str,
    status: str | None = None,
    split: str | None = None,
    q: str | None = None,
    sort: str = "created",
    page: int = 1,
    page_size: int = 50,
):
    try:
        return dataset_svc.list_images(
            dataset_id,
            status=status,
            split=split,
            q=q,
            sort=sort,
            page=page,
            page_size=page_size,
        )
    except LookupError as e:
        raise HTTPException(404, str(e)) from e


@router.get("/{dataset_id}/next-unlabeled")
def next_unlabeled(dataset_id: str, after: str | None = None):
    try:
        item = dataset_svc.next_unlabeled(dataset_id, after=after)
        return {"item": item}
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
