"""FastAPI 入口：API + 静态前端。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# 必须最先加载 config（线程环境变量）
from app import config

config.setup_logging()
config.apply_torch_threads()

from app import db, tasks  # noqa: E402
from app.api import autolabel, datasets, images, models, system, train  # noqa: E402
from app.services import (  # noqa: E402
    autolabel_svc,
    crop_svc,
    dedup_svc,
    openvocab_svc,
    track_svc,
    train_svc,
)

logger = logging.getLogger(__name__)


def _register_task_handlers() -> None:
    from app.services import model_svc

    tasks.register_handler("autolabel", autolabel_svc.run_autolabel)
    tasks.register_handler("openvocab", openvocab_svc.run_openvocab)
    tasks.register_handler("track", track_svc.run_track)
    tasks.register_handler("dedup", dedup_svc.run_dedup)
    tasks.register_handler("train", train_svc.run_train_job)
    tasks.register_handler("openvino_export", model_svc.run_openvino_export)
    tasks.register_handler("import_video", track_svc.run_import_video)
    tasks.register_handler("crop_import", crop_svc.run_crop_import)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_data_dirs()
    db.init_db()
    _register_task_handlers()
    tasks.start_workers()
    n = train_svc.recover_interrupted_jobs()
    if n:
        logger.warning("启动恢复：标记 %s 个中断的训练任务", n)
    # 旧模型：从 OpenVINO metadata.yaml 回填 export_imgsz / export_dynamic
    try:
        from app.services import model_svc as _model_svc

        nb = _model_svc.backfill_export_meta_from_disk()
        if nb:
            logger.info("已回填 %s 条模型的 OpenVINO 导出元数据", nb)
    except Exception as exc:
        logger.warning("回填 OpenVINO 元数据失败: %s", exc)
    # 尝试关闭 ultralytics 同步
    try:
        from ultralytics.utils import SETTINGS

        SETTINGS.update({"sync": False})
    except Exception:
        pass
    logger.info(
        "YOLO Studio 就绪 host=%s port=%s data=%s",
        config.APP_HOST,
        config.APP_PORT,
        config.DATA_DIR,
    )
    try:
        yield
    finally:
        killed = train_svc.kill_running_train_processes()
        if killed:
            logger.warning("关闭时终止 %s 个训练进程", killed)
        tasks.stop_workers()
        logger.info("YOLO Studio 已停止")


app = FastAPI(title="YOLO Studio", description="图片标注 + 模型训练一体化平台", lifespan=lifespan)

app.include_router(datasets.router)
app.include_router(images.router)
app.include_router(autolabel.router)
app.include_router(train.router)
app.include_router(models.router)
app.include_router(system.router)

web_dir = config.WEB_DIR
if web_dir.is_dir():
    vendor = web_dir / "vendor"
    if vendor.is_dir():
        app.mount("/vendor", StaticFiles(directory=str(vendor)), name="vendor")
    css = web_dir / "css"
    if css.is_dir():
        app.mount("/css", StaticFiles(directory=str(css)), name="css")
    js = web_dir / "js"
    if js.is_dir():
        app.mount("/js", StaticFiles(directory=str(js)), name="js")


@app.get("/")
def index():
    index_path = web_dir / "index.html"
    if not index_path.exists():
        return {"message": "web/index.html 缺失"}
    return FileResponse(index_path)


@app.get("/favicon.ico")
def favicon():
    return {"ok": True}
