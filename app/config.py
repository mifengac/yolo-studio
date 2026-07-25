"""应用配置。线程环境变量必须在 import torch 之前设置。"""

from __future__ import annotations

import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

_env_file = BASE_DIR / ".env"
if _env_file.exists():
    load_dotenv(_env_file)
else:
    load_dotenv(BASE_DIR / ".env.example")

# --- 离线硬约束（import ultralytics 之前就要生效）---
os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.home() / ".config" / "Ultralytics"))

# --- CPU 线程：用物理核数，且必须在 import torch 前设置 ---
CPU_THREADS = int(os.getenv("CPU_THREADS", "14"))
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))

# 服务
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "5016"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# 目录
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data"))).resolve()
SQLITE_PATH = DATA_DIR / "app.db"
DATASETS_DIR = DATA_DIR / "datasets"
RUNS_DIR = DATA_DIR / "runs"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = DATA_DIR / "logs"
WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", str(BASE_DIR / "weights"))).resolve()
WEB_DIR = BASE_DIR / "web"

# 上传限制
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
THUMB_WIDTH = 320

# 推理 / 训练默认
INFER_BATCH_SIZE = int(os.getenv("INFER_BATCH_SIZE", "8"))
TRAIN_WORKERS = int(os.getenv("TRAIN_WORKERS", "8"))
DEFAULT_BASE_MODEL = "yolo26n.pt"
DEFAULT_IMGSZ = 416
DEFAULT_EPOCHS = 40
DEFAULT_BATCH = 16
DEFAULT_PATIENCE = 10
DEFAULT_FREEZE = 10
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.5

# 开放词表 / SAM 限制
OPENVOCAB_MAX_IMAGES = 200
TRACK_MAX_FRAMES = 300
# SAM image embedding 缓存条数（每条约数十～上百 MB，CPU 上建议 ≤4）
SAM_EMBED_CACHE_SIZE = 4

# 重 CPU 任务类型（走 heavy 线程池，串行；轻任务走 light 池不阻塞界面）
HEAVY_TASK_TYPES = frozenset(
    {
        "autolabel",
        "openvocab",
        "track",
        "train",
        "train_resume",
        "export",
        "evaluate",
        "openvino_export",
        "import_video",
        "dedup",  # pHash+DCT 重 CPU，不可与训练并发
    }
)

# 耗时预估基线：E5-2697 v3，1000 张，yolo26n，imgsz=640，不冻结 = 10 分钟/epoch
ESTIMATE_BASE_SEC_PER_EPOCH = 600.0
ESTIMATE_BASE_IMAGES = 1000
ESTIMATE_WARN_HOURS = 12
ESTIMATE_BLOCK_HOURS = 24

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

# 预置底模（仅绝对路径加载，禁止 YOLO('yolov8n.pt') 触发下载）
BUILTIN_BASE_MODELS = {
    "yolo26n.pt": WEIGHTS_DIR / "yolo26n.pt",
    "yolo26s.pt": WEIGHTS_DIR / "yolo26s.pt",
    "0517_yolo26n_wheelie_multi-rider.pt": WEIGHTS_DIR / "0517_yolo26n_wheelie_multi-rider.pt",
    "0517_yolo26s_wheelie_multi-rider.pt": WEIGHTS_DIR / "0517_yolo26s_wheelie_multi-rider.pt",
    "yolov8s-worldv2.pt": WEIGHTS_DIR / "yolov8s-worldv2.pt",
    "mobile_sam.pt": WEIGHTS_DIR / "mobile_sam.pt",
}


def ensure_data_dirs() -> None:
    for p in (DATA_DIR, DATASETS_DIR, RUNS_DIR, MODELS_DIR, LOGS_DIR, WEIGHTS_DIR):
        p.mkdir(parents=True, exist_ok=True)


def resolve_weight_path(name_or_path: str) -> Path:
    """把模型名或相对路径解析为绝对路径，不存在则抛人话错误。"""
    raw = (name_or_path or "").strip()
    if not raw:
        raise ValueError("模型路径不能为空")

    # 绝对路径
    p = Path(raw)
    if p.is_file():
        return p.resolve()

    # 去掉 weights/ 前缀再查
    key = raw.replace("\\", "/").split("/")[-1]
    if key in BUILTIN_BASE_MODELS and BUILTIN_BASE_MODELS[key].is_file():
        return BUILTIN_BASE_MODELS[key].resolve()

    # weights 目录
    candidate = WEIGHTS_DIR / key
    if candidate.is_file():
        return candidate.resolve()

    # data/models 注册副本
    candidate = MODELS_DIR / key
    if candidate.is_file():
        return candidate.resolve()

    # 相对项目根
    candidate = (BASE_DIR / raw).resolve()
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"找不到模型文件「{raw}」。请确认已按 docs/20260725_离线权重清单.md 放到 weights/ 目录。"
    )


def setup_logging() -> None:
    ensure_data_dirs()
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(sh)

    log_path = LOGS_DIR / "app.log"
    fh = TimedRotatingFileHandler(
        str(log_path), when="midnight", backupCount=14, encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(fh)


def apply_torch_threads() -> None:
    """在业务代码里尽早调用；config 已设好环境变量。"""
    try:
        import torch

        torch.set_num_threads(CPU_THREADS)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max(1, min(2, CPU_THREADS)))
    except Exception:
        pass
