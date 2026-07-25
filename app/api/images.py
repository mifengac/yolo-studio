"""图片与标注 API。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.schemas import AnnotationsPut, SamRequest
from app.services import dataset_svc, sam_svc

router = APIRouter(prefix="/api/images", tags=["images"])


@router.get("/{img_id}/file")
def get_file(img_id: str):
    try:
        img = dataset_svc.get_image(img_id)
        path = dataset_svc.image_file_path(img)
        if not path.is_file():
            raise HTTPException(404, "原图文件丢失")
        return FileResponse(path, filename=img["filename"])
    except LookupError as e:
        raise HTTPException(404, str(e)) from e


@router.get("/{img_id}/thumb")
def get_thumb(img_id: str):
    try:
        img = dataset_svc.get_image(img_id)
        path = dataset_svc.ensure_thumb(img)
        return FileResponse(path, media_type="image/jpeg")
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e


@router.get("/{img_id}/annotations")
def get_annotations(img_id: str):
    try:
        img = dataset_svc.get_image(img_id)
        boxes = dataset_svc.get_annotations(img_id)
        return {"image": img, "boxes": boxes}
    except LookupError as e:
        raise HTTPException(404, str(e)) from e


@router.put("/{img_id}/annotations")
def put_annotations(img_id: str, body: AnnotationsPut):
    try:
        boxes = [b.model_dump() for b in body.boxes]
        return dataset_svc.put_annotations(
            img_id, boxes, review_status=body.review_status
        )
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.post("/{img_id}/confirm")
def confirm(img_id: str):
    try:
        return dataset_svc.confirm_image(img_id)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.post("/{img_id}/sam")
def sam_point(img_id: str, body: SamRequest):
    try:
        return sam_svc.predict_box_from_points(img_id, body.points, body.labels)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e)) from e
