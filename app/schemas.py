"""Pydantic 请求/响应模型。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class DatasetCreate(BaseModel):
    name: str
    classes: list[str]
    notes: str = ""


class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    classes: Optional[list[str]] = None
    notes: Optional[str] = None


class BoxIn(BaseModel):
    class_idx: int
    cx: float
    cy: float
    w: float
    h: float
    conf: float = 1.0
    source: str = "manual"


class AnnotationsPut(BaseModel):
    boxes: list[BoxIn] = Field(default_factory=list)
    review_status: Optional[str] = None


class AutolabelRequest(BaseModel):
    # "default"/"auto" 时走模型仓库默认预标注路径，闭环才能生效
    model: str = "default"
    conf: float = 0.25
    iou: float = 0.5
    imgsz: int = 640
    class_map: dict[str, int] = Field(default_factory=dict)
    scope: str = "unlabeled"  # unlabeled | all
    overwrite: bool = False


class OpenvocabRequest(BaseModel):
    prompts: list[str]
    class_map: dict[int, int] = Field(default_factory=dict)
    conf: float = 0.1
    imgsz: int = 640
    scope: str = "unlabeled"
    overwrite: bool = False
    preview_limit: Optional[int] = None  # 先试 N 张


class SamRequest(BaseModel):
    points: list[list[float]]  # [[x,y], ...] 像素坐标
    labels: list[int] = Field(default_factory=lambda: [1])


class TrackRequest(BaseModel):
    start_image_id: str
    max_frames: int = 300
    model: str = "default"
    conf: float = 0.25


class CropImportParams(BaseModel):
    """智能切图参数（默认值实测标定，勿随意改）。"""

    model: str = "yolo26n.pt"
    target_class: str = "person"
    conf: float = 0.25
    imgsz: int = 1280
    min_box_h: int = 100
    pad_ratio: float = 0.25
    top_ratio: float = -0.15
    bottom_ratio: float = 0.55
    max_crops: int = 8000
    preview_limit: int = 20


class CropImportFromDataset(CropImportParams):
    """从已有数据集切图到当前数据集。"""

    pass


class TrainEstimateRequest(BaseModel):
    dataset_id: str
    base_model: str = "weights/yolo26n.pt"
    epochs: int = 40
    imgsz: int = 416
    batch: int = 16
    freeze: int = 10
    only_confirmed: bool = True
    val_ratio: float = 0.2


class TrainCreateRequest(BaseModel):
    dataset_id: str
    base_model: str = "weights/yolo26n.pt"
    # None = 后端按微调/从零自动：微调 20、从零 40（curl/冒烟不传时生效）
    epochs: Optional[int] = None
    imgsz: int = 416
    batch: int = 16
    val_ratio: float = 0.2
    patience: int = 10
    only_confirmed: bool = True
    freeze: int = 10
    workers: int = 8
    augment_preset: str = "default"
    force_long: bool = False  # 超过 24h 时需勾选
    # 每类最少框数；默认 10。冒烟测试可传 1
    min_boxes_per_class: int = 10


class ModelRegisterRequest(BaseModel):
    name: Optional[str] = None
    notes: str = ""


class EvaluateRequest(BaseModel):
    dataset_id: str
    conf: float = 0.25
    imgsz: int = 416


class OkResponse(BaseModel):
    ok: bool = True
    message: str = ""
    data: Any = None
