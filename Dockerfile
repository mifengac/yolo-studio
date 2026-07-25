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
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
# 与本地实测一致：torch 2.13.0+cpu / torchvision 0.28.0+cpu / ultralytics 8.4.105
# ultralytics 会拉完整 opencv-python，再卸掉只留 headless，约省 100MB+（失败则保留双份）
RUN pip install --upgrade pip \
    && pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r /app/requirements.txt \
    && pip install "git+https://github.com/ultralytics/CLIP.git" \
    && (pip uninstall -y opencv-python || true)

COPY app /app/app
COPY web /app/web
COPY scripts /app/scripts
COPY .env.example /app/.env.example

# 权重在构建时可选 COPY；运行时也可用 volume 挂载
COPY weights /app/weights

# CLIP 文本塔（开放词表）；构建前请放置 weights/clip/ViT-B-32.pt（约 338MB）
RUN mkdir -p /root/.cache/clip \
    && if [ -f /app/weights/clip/ViT-B-32.pt ]; then \
         cp /app/weights/clip/ViT-B-32.pt /root/.cache/clip/ViT-B-32.pt; \
         echo "seeded CLIP ViT-B-32.pt"; \
       else \
         echo "WARN: weights/clip/ViT-B-32.pt 缺失，开放词表在离线环境将不可用"; \
       fi

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
# 必须是 CPU 版；禁止用「无 CUDA 就放过」——CUDA wheel 在无 GPU 机器上 is_available 也是 False
assert "+cpu" in torch.__version__, f"必须是 CPU 版 torch，实际: {torch.__version__}"
print("ok")
PY

EXPOSE 5016
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5016"]
