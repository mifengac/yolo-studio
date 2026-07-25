# YOLO Studio — 纯 CPU、内网离线交付镜像
# 有网机器 build → docker save → 内网 load
#
# CLIP 权重只进最终镜像的 /root/.cache/clip/ 一份：中间 stage 整理后用 COPY --from，
# 避免「COPY weights 含 clip + 再 cp 到 cache」双份各占一层 338MB。

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
# ultralytics 会拉完整 opencv-python；卸掉后须 force-reinstall headless 恢复 cv2
RUN pip install --upgrade pip \
    && pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r /app/requirements.txt \
    && pip install "git+https://github.com/ultralytics/CLIP.git" \
    && pip uninstall -y opencv-python \
    && pip install --force-reinstall --no-deps opencv-python-headless==5.0.0.93 \
    && python -c "import cv2; from ultralytics import YOLO; print('cv2 OK', cv2.__version__)"

# ---------- 权重整理（不进入最终镜像层）----------
FROM base AS weight_prep
COPY weights /w
RUN mkdir -p /out/weights /out/clip \
    && find /w -maxdepth 1 -type f -name '*.pt' -exec cp -a {} /out/weights/ \; \
    && if [ -f /w/.gitkeep ]; then cp -a /w/.gitkeep /out/weights/; fi \
    && if [ -f /w/clip/ViT-B-32.pt ]; then \
         cp /w/clip/ViT-B-32.pt /out/clip/ViT-B-32.pt; \
         echo "CLIP ready"; \
       else \
         echo "WARN: weights/clip/ViT-B-32.pt 缺失，开放词表离线不可用"; \
       fi \
    && ls -la /out/weights \
    && ls -la /out/clip || true

# ---------- 最终镜像 ----------
FROM base

COPY app /app/app
COPY web /app/web
COPY scripts /app/scripts
COPY .env.example /app/.env.example

# 检测/SAM：仅 *.pt 根目录文件；CLIP：仅 cache 一份
COPY --from=weight_prep /out/weights /app/weights
COPY --from=weight_prep /out/clip/ViT-B-32.pt /root/.cache/clip/ViT-B-32.pt

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
assert "+cpu" in torch.__version__, f"必须是 CPU 版 torch，实际: {torch.__version__}"
# CLIP 单份
clip = Path("/root/.cache/clip/ViT-B-32.pt")
assert clip.is_file(), f"CLIP 权重缺失: {clip}"
assert not Path("/app/weights/clip").exists(), "CLIP 不应再出现在 /app/weights/clip"
print("clip single-copy OK", clip.stat().st_size)
print("ok")
PY

EXPOSE 5016
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5016"]
