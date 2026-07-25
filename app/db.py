"""SQLite 连接、建表与通用读写。标注以数据库为唯一真源。"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from app import config

logger = logging.getLogger(__name__)
_lock = threading.RLock()


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def connect():
    config.ensure_data_dirs()
    conn = sqlite3.connect(str(config.SQLITE_PATH), check_same_thread=False, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with _lock, connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS dataset (
              id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              classes TEXT NOT NULL,
              notes TEXT DEFAULT '',
              created_at TEXT,
              updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS image (
              id TEXT PRIMARY KEY,
              dataset_id TEXT NOT NULL,
              filename TEXT NOT NULL,
              rel_path TEXT NOT NULL,
              width INTEGER,
              height INTEGER,
              sha1 TEXT,
              phash TEXT,
              group_key TEXT,
              source TEXT,
              split TEXT DEFAULT '',
              review_status TEXT DEFAULT 'unlabeled',
              box_count INTEGER DEFAULT 0,
              uncertainty REAL DEFAULT 0,
              created_at TEXT,
              FOREIGN KEY (dataset_id) REFERENCES dataset(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_image_ds_status ON image(dataset_id, review_status);
            CREATE INDEX IF NOT EXISTS idx_image_ds_unc ON image(dataset_id, uncertainty DESC);

            CREATE TABLE IF NOT EXISTS annotation (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              image_id TEXT NOT NULL,
              class_idx INTEGER NOT NULL,
              cx REAL, cy REAL, w REAL, h REAL,
              conf REAL DEFAULT 1.0,
              source TEXT DEFAULT 'manual',
              created_at TEXT,
              FOREIGN KEY (image_id) REFERENCES image(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_anno_image ON annotation(image_id);

            CREATE TABLE IF NOT EXISTS task (
              id TEXT PRIMARY KEY,
              type TEXT,
              dataset_id TEXT,
              status TEXT,
              params TEXT,
              progress REAL DEFAULT 0,
              message TEXT,
              error TEXT,
              created_at TEXT,
              started_at TEXT,
              finished_at TEXT
            );

            CREATE TABLE IF NOT EXISTS train_job (
              id TEXT PRIMARY KEY,
              dataset_id TEXT,
              base_model TEXT,
              params TEXT,
              run_dir TEXT,
              log_path TEXT,
              status TEXT,
              metrics TEXT,
              best_pt TEXT,
              last_epoch INTEGER DEFAULT 0,
              resume_from TEXT,
              eta_seconds REAL,
              pid INTEGER,
              created_at TEXT,
              finished_at TEXT
            );

            CREATE TABLE IF NOT EXISTS model (
              id TEXT PRIMARY KEY,
              name TEXT,
              path TEXT,
              classes TEXT,
              metrics TEXT,
              from_job TEXT,
              notes TEXT,
              is_default_autolabel INTEGER DEFAULT 0,
              openvino_path TEXT,
              created_at TEXT
            );
            """
        )
        # 迁移：补齐可能缺失的列
        _ensure_columns(
            conn,
            "train_job",
            {
                "last_epoch": "INTEGER DEFAULT 0",
                "resume_from": "TEXT",
                "eta_seconds": "REAL",
                "pid": "INTEGER",
            },
        )
        _ensure_columns(
            conn,
            "model",
            {
                "openvino_path": "TEXT",
                # pending | exporting | ready | failed；旧数据 NULL 按 openvino_path 兼容
                "ov_status": "TEXT",
            },
        )
        # 旧记录兼容：有 openvino_path 视为 ready，否则 pending
        conn.execute(
            """UPDATE model SET ov_status='ready'
               WHERE openvino_path IS NOT NULL AND openvino_path != ''
                 AND (ov_status IS NULL OR ov_status='')"""
        )
        conn.execute(
            """UPDATE model SET ov_status='pending'
               WHERE (openvino_path IS NULL OR openvino_path='')
                 AND (ov_status IS NULL OR ov_status='')"""
        )
        # 去重算法从 aHash 换 pHash，旧值不兼容，清空让任务重算
        try:
            conn.execute(
                "UPDATE image SET phash=NULL WHERE phash IS NOT NULL AND phash NOT LIKE 'p1:%'"
            )
        except Exception:
            pass


def _ensure_columns(conn: sqlite3.Connection, table: str, cols: dict[str, str]) -> None:
    existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, decl in cols.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
            logger.info("Migrated %s: added %s", table, name)


def row_to_dict(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    return dict(row)


def loads_json(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def dumps_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
