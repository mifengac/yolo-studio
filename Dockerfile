# YOLO Studio — 纯 CPU、内网离线交付镜像
# 有网机器 build → docker save → 内网 load

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_VERBOSE=False \
    YOLO_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    YOLO_CONFIG_DIR=/root/.config/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    CPU_THREADS=14 \
    OMP_NUM_THREADS=14 \
    MKL_NUM_THREADS=14

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        fonts-dejavu-core \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r /app/requirements.txt

COPY app /app/app
COPY web /app/web
COPY scripts /app/scripts
COPY .env.example /app/.env.example

# 权重在构建时可选 COPY；运行时也可用 volume 挂载
COPY weights /app/weights

RUN mkdir -p /root/.config/Ultralytics /app/data \
    && python - <<'PY'
import os
from pathlib import Path

os.environ["YOLO_OFFLINE"] = "1"
os.environ.setdefault("YOLO_CONFIG_DIR", "/root/.config/Ultralytics")

cfg = Path("/root/.config/Ultralytics")
cfg.mkdir(parents=True, exist_ok=True)
src_candidates = list(Path("/usr/share/fonts").rglob("DejaVuSans.ttf"))
if src_candidates:
    target = cfg / "Arial.ttf"
    target.write_bytes(src_candidates[0].read_bytes())
    print("seeded font", target)

from ultralytics.utils import SETTINGS
SETTINGS.update({
    "sync": False,
    "api_key": "",
    "clearml": False,
    "comet": False,
    "dvc": False,
    "hub": False,
    "mlflow": False,
    "neptune": False,
    "raytune": False,
    "tensorboard": False,
    "wandb": False,
})
print("ultralytics settings updated, torch check next")
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
assert "+cpu" in torch.__version__ or not torch.cuda.is_available()
print("ok")
PY

EXPOSE 5016
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5016"]
