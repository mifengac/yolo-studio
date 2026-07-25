"""通用后台任务队列：线程池 + task 表。训练与重 CPU 任务互斥排队。"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional
from uuid import uuid4

from app import config, db

logger = logging.getLogger(__name__)

_executor: Optional[ThreadPoolExecutor] = None
_heavy_lock = threading.Lock()  # 训练/预标注/跟踪互斥
_handlers: dict[str, Callable[[dict], None]] = {}
_started = False


def new_task_id(prefix: str = "task") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"


def register_handler(task_type: str, fn: Callable[[dict], None]) -> None:
    _handlers[task_type] = fn


def start_workers() -> None:
    global _executor, _started
    if _started:
        return
    _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ys-worker")
    _started = True
    logger.info("后台任务线程池已启动")


def stop_workers() -> None:
    global _executor, _started
    if _executor:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None
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
    row = {
        "id": task_id,
        "type": task_type,
        "dataset_id": dataset_id,
        "status": "pending",
        "params": db.dumps_json(params),
        "progress": 0.0,
        "message": "排队中",
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
    if not _executor:
        start_workers()
    assert _executor is not None
    _executor.submit(_run_task, task_id)


def _run_task(task_id: str) -> None:
    task = get_task(task_id)
    if not task:
        return
    handler = _handlers.get(task["type"] or "")
    if not handler:
        update_task(
            task_id,
            status="failed",
            error=f"未知任务类型: {task['type']}",
            finished_at=db.utcnow(),
        )
        return

    # 重 CPU 任务串行
    heavy = task["type"] in {
        "autolabel",
        "openvocab",
        "track",
        "train",
        "export",
        "evaluate",
        "openvino_export",
    }
    if heavy:
        update_task(task_id, message="等待其他重任务结束…")
        _heavy_lock.acquire()

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
        if heavy:
            _heavy_lock.release()


def is_training_running() -> bool:
    with db._lock, db.connect() as conn:
        row = conn.execute(
            "SELECT id FROM train_job WHERE status='running' LIMIT 1"
        ).fetchone()
    return row is not None
