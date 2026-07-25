"""数据集 API。"""

from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app import config
from app.schemas import CropImportFromDataset, CropImportParams, DatasetCreate, DatasetUpdate
from app.services import crop_svc, dataset_svc, track_svc

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
    """上传视频后立刻返回 task，后台抽帧并回报进度。"""
    from app import tasks as task_mod

    try:
        dataset_svc.get_dataset(dataset_id)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    data = await file.read()
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"视频超过 {config.MAX_UPLOAD_MB} MB 限制")
    try:
        meta = track_svc.save_video_temp(
            dataset_id, data, file.filename or "video.mp4"
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    task = task_mod.create_task(
        "import_video",
        {
            "dataset_id": dataset_id,
            "tmp_path": meta["tmp_path"],
            "group_key": meta["group_key"],
            "filename": meta["filename"],
            "fps": fps,
        },
        dataset_id=dataset_id,
        submit=True,
    )
    return task


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


@router.post("/{dataset_id}/annotations/clear-auto")
def clear_auto_annotations(dataset_id: str):
    """清空模型自动预标注（source=auto），手工标注保留。"""
    try:
        return dataset_svc.clear_auto_annotations(dataset_id)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e


def _parse_crop_params(params_json: str | None) -> dict:
    if not params_json:
        return CropImportParams().model_dump()
    try:
        raw = json.loads(params_json)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"参数 JSON 无效: {e}") from e
    return CropImportParams(**raw).model_dump()


@router.post("/{dataset_id}/crop-import/zip")
async def crop_import_zip(
    dataset_id: str,
    file: UploadFile = File(...),
    params: str | None = Form(None),
):
    """上传大图 ZIP，后台智能切图入库（原图不入库）。"""
    from app import tasks as task_mod

    try:
        dataset_svc.get_dataset(dataset_id)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    data = await file.read()
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"ZIP 超过 {config.MAX_UPLOAD_MB} MB 限制")
    try:
        p = _parse_crop_params(params)
        meta = crop_svc.save_zip_temp(dataset_id, data, file.filename or "upload.zip")
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e)) from e
    task = task_mod.create_task(
        "crop_import",
        {
            "dataset_id": dataset_id,
            "source": "zip",
            "zip_path": meta["zip_path"],
            **p,
        },
        dataset_id=dataset_id,
        submit=True,
    )
    return task


@router.post("/{dataset_id}/crop-import/from/{src_dataset_id}")
def crop_import_from_dataset(dataset_id: str, src_dataset_id: str, body: CropImportFromDataset | None = None):
    """从已有数据集（大图）切图到当前数据集。"""
    from app import tasks as task_mod

    body = body or CropImportFromDataset()
    try:
        dataset_svc.get_dataset(dataset_id)
        dataset_svc.get_dataset(src_dataset_id)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    if dataset_id == src_dataset_id:
        raise HTTPException(400, "源数据集不能与目标数据集相同，请新建一个切图数据集")
    task = task_mod.create_task(
        "crop_import",
        {
            "dataset_id": dataset_id,
            "source": "dataset",
            "src_dataset_id": src_dataset_id,
            **body.model_dump(),
        },
        dataset_id=dataset_id,
        submit=True,
    )
    return task


@router.post("/{dataset_id}/crop-import/preview")
async def crop_import_preview(
    dataset_id: str,
    file: UploadFile | None = File(None),
    src_dataset_id: str | None = Form(None),
    params: str | None = Form(None),
):
    """先切 N 张预览（默认 20），返回 base64 缩略图，不入库。"""
    try:
        dataset_svc.get_dataset(dataset_id)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    try:
        p = _parse_crop_params(params)
        if file is not None and file.filename:
            data = await file.read()
            if len(data) > config.MAX_UPLOAD_BYTES:
                raise HTTPException(400, f"ZIP 超过 {config.MAX_UPLOAD_MB} MB 限制")
            return crop_svc.preview_crops(source="zip", zip_bytes=data, params=p)
        if src_dataset_id:
            return crop_svc.preview_crops(
                source="dataset", src_dataset_id=src_dataset_id, params=p
            )
        raise HTTPException(400, "请上传 ZIP，或指定源数据集 src_dataset_id")
    except HTTPException:
        raise
    except (ValueError, FileNotFoundError, LookupError) as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"预览失败: {e}") from e
