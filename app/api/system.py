"""系统健康与设备信息。"""

from __future__ import annotations

import platform
import shutil

from fastapi import APIRouter

from app import config
from app.services import model_svc

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health():
    return {"status": "ok", "app": "yolo-studio", "port": config.APP_PORT}


@router.get("/system/info")
def system_info():
    torch_ver = None
    cuda = False
    try:
        import torch

        torch_ver = torch.__version__
        cuda = bool(torch.cuda.is_available())
    except Exception:
        pass

    disk = shutil.disk_usage(str(config.DATA_DIR))
    weights = []
    for name, path in config.BUILTIN_BASE_MODELS.items():
        weights.append({"name": name, "available": path.is_file(), "path": str(path)})

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_threads": config.CPU_THREADS,
        "torch": torch_ver,
        "cuda": cuda,
        "cpu_only_expected": True,
        "data_dir": str(config.DATA_DIR),
        "weights_dir": str(config.WEIGHTS_DIR),
        "disk_free_gb": round(disk.free / (1024**3), 2),
        "default_autolabel": model_svc.get_default_autolabel_path(),
        "weights": weights,
    }
