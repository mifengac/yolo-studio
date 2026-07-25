"""通用后台任务队列：重任务 / 轻任务分池，训练与预标注等串行不堵界面。"""

from __future__ import annotations

import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional
from uuid import uuid4

from app import config, db

logger = logging.getLogger(__name__)

_heavy_executor: Optional[ThreadPoolExecutor] = None
_light_executor: Optional[ThreadPoolExecutor] = None
_handlers: dict[str, Callable[[dict], None]] = {}
_started = False
# 重任务池内已排队/运行数（用于「等待其他重任务」文案；池 max_workers=1 保证串行）
_heavy_inflight = 0
_heavy_inflight_lock = __import__("threading").Lock()


def new_task_id(prefix: str = "task") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"


def register_handler(task_type: str, fn: Callable[[dict], None]) -> None:
    _handlers[task_type] = fn


def _is_heavy(task_type: str) -> bool:
    return (task_type or "") in config.HEAVY_TASK_TYPES


def start_workers() -> None:
    global _heavy_executor, _light_executor, _started
    if _started:
        return
    # 重任务单 worker 串行，不再用全局锁占满名额
    _heavy_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ys-heavy")
    _light_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ys-light")
    _started = True
    logger.info("后台任务线程池已启动（heavy=1, light=4）")


def stop_workers() -> None:
    global _heavy_executor, _light_executor, _started
    if _heavy_executor:
        _heavy_executor.shutdown(wait=False, cancel_futures=True)
        _heavy_executor = None
    if _light_executor:
        _light_executor.shutdown(wait=False, cancel_futures=True)
        _light_executor = None
    _started = False


def create_task(
    task_type: str,
    params: dict[str, Any],
    dataset_id: Optional[str] = None,
    *,
    submit: bool = True,
) -> dict:
    task_id = new_task_id(task_type)
    now = db.utcnow()
    # 重任务若已有在飞/排队，创建时给出等待文案
    msg = "排队中"
    if _is_heavy(task_type):
        with _heavy_inflight_lock:
            if _heavy_inflight > 0:
                msg = "等待其他重任务结束…"
    row = {
        "id": task_id,
        "type": task_type,
        "dataset_id": dataset_id,
        "status": "pending",
        "params": db.dumps_json(params),
        "progress": 0.0,
        "message": msg,
        "error": None,
        "created_at": now,
        "started_at": None,
        "finished_at": None,
    }
    with db._lock, db.connect() as conn:
        conn.execute(
            """INSERT INTO task
               (id, type, dataset_id, status, params, progress, message, error,
                created_at, started_at, finished_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                row["id"],
                row["type"],
                row["dataset_id"],
                row["status"],
                row["params"],
                row["progress"],
                row["message"],
                row["error"],
                row["created_at"],
                row["started_at"],
                row["finished_at"],
            ),
        )
    if submit:
        submit_task(task_id)
    return get_task(task_id)  # type: ignore


def get_task(task_id: str) -> Optional[dict]:
    with db._lock, db.connect() as conn:
        row = conn.execute("SELECT * FROM task WHERE id=?", (task_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["params"] = db.loads_json(d.get("params"), {})
    return d


def update_task(task_id: str, **fields: Any) -> None:
    if not fields:
        return
    cols = []
    vals = []
    for k, v in fields.items():
        if k == "params" and not isinstance(v, str):
            v = db.dumps_json(v)
        cols.append(f"{k}=?")
        vals.append(v)
    vals.append(task_id)
    with db._lock, db.connect() as conn:
        conn.execute(f"UPDATE task SET {', '.join(cols)} WHERE id=?", vals)


def set_progress(task_id: str, progress: float, message: str = "") -> None:
    fields: dict[str, Any] = {"progress": max(0.0, min(100.0, float(progress)))}
    if message:
        fields["message"] = message
    update_task(task_id, **fields)


def submit_task(task_id: str) -> None:
    global _heavy_inflight
    if not _started:
        start_workers()
    assert _heavy_executor is not None and _light_executor is not None
    task = get_task(task_id)
    if not task:
        return
    if _is_heavy(task.get("type") or ""):
        with _heavy_inflight_lock:
            if _heavy_inflight > 0:
                update_task(task_id, message="等待其他重任务结束…")
            _heavy_inflight += 1
        _heavy_executor.submit(_run_task, task_id, True)
    else:
        _light_executor.submit(_run_task, task_id, False)


def _run_task(task_id: str, is_heavy: bool) -> None:
    global _heavy_inflight
    task = get_task(task_id)
    if not task:
        if is_heavy:
            with _heavy_inflight_lock:
                _heavy_inflight = max(0, _heavy_inflight - 1)
        return
    handler = _handlers.get(task["type"] or "")
    if not handler:
        update_task(
            task_id,
            status="failed",
            error=f"未知任务类型: {task['type']}",
            finished_at=db.utcnow(),
        )
        if is_heavy:
            with _heavy_inflight_lock:
                _heavy_inflight = max(0, _heavy_inflight - 1)
        return

    try:
        update_task(
            task_id,
            status="running",
            started_at=db.utcnow(),
            message="运行中",
            progress=0,
        )
        handler(task)
        cur = get_task(task_id)
        if cur and cur.get("status") == "running":
            update_task(
                task_id,
                status="success",
                progress=100,
                message="完成",
                finished_at=db.utcnow(),
            )
    except Exception as exc:
        logger.exception("任务失败 %s", task_id)
        update_task(
            task_id,
            status="failed",
            error=str(exc) or traceback.format_exc()[-500:],
            message="失败",
            finished_at=db.utcnow(),
        )
    finally:
        if is_heavy:
            with _heavy_inflight_lock:
                _heavy_inflight = max(0, _heavy_inflight - 1)


def is_training_running() -> bool:
    with db._lock, db.connect() as conn:
        row = conn.execute(
            "SELECT id FROM train_job WHERE status='running' LIMIT 1"
        ).fetchone()
    return row is not None
