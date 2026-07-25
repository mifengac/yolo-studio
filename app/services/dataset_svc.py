"""数据集与图片管理。"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from PIL import Image

from app import config, db

logger = logging.getLogger(__name__)

REVIEW_STATUSES = {
    "unlabeled",
    "auto",
    "reviewed",
    "confirmed",
    "skipped",
}


def _new_id(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"


def _parse_classes(value) -> list[str]:
    if isinstance(value, list):
        raw = value
    else:
        raw = re.split(r"[,;\n\r]+", str(value or ""))
    items: list[str] = []
    seen: set[str] = set()
    for r in raw:
        item = " ".join(str(r or "").strip().split())
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(item)
    if not items:
        raise ValueError("至少填写一个类别")
    if len(items) > 50:
        raise ValueError("类别数量过多（最多 50 个）")
    return items


def _safe_filename(name: str, fallback: str = "image.jpg") -> str:
    base = Path(name or "").name or fallback
    cleaned = re.sub(r"[^A-Za-z0-9._\u4e00-\u9fff-]+", "_", base).strip("._")
    return cleaned or fallback


def dataset_root(dataset_id: str) -> Path:
    return config.DATASETS_DIR / dataset_id


def ensure_dataset_dirs(dataset_id: str) -> Path:
    root = dataset_root(dataset_id)
    for sub in ("images", "thumbs", "exports"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def serialize_dataset(row: dict, stats: Optional[dict] = None) -> dict:
    d = {
        "id": row["id"],
        "name": row["name"],
        "classes": db.loads_json(row.get("classes"), []),
        "notes": row.get("notes") or "",
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }
    if stats:
        d.update(stats)
    return d


def get_dataset_stats(dataset_id: str) -> dict:
    with db._lock, db.connect() as conn:
        total = conn.execute(
            "SELECT COUNT(*) AS c FROM image WHERE dataset_id=?", (dataset_id,)
        ).fetchone()["c"]
        rows = conn.execute(
            """SELECT review_status, COUNT(*) AS c, COALESCE(SUM(box_count),0) AS boxes
               FROM image WHERE dataset_id=? GROUP BY review_status""",
            (dataset_id,),
        ).fetchall()
    by_status = {r["review_status"]: r["c"] for r in rows}
    labeled = sum(
        by_status.get(s, 0)
        for s in ("auto", "reviewed", "confirmed")
    )
    confirmed = by_status.get("confirmed", 0)
    total_boxes = sum(r["boxes"] for r in rows)
    return {
        "image_count": total,
        "labeled_count": labeled,
        "confirmed_count": confirmed,
        "unlabeled_count": by_status.get("unlabeled", 0),
        "auto_count": by_status.get("auto", 0),
        "skipped_count": by_status.get("skipped", 0),
        "box_count": total_boxes,
        "status_counts": by_status,
    }


def create_dataset(name: str, classes, notes: str = "") -> dict:
    name = " ".join((name or "").strip().split())
    if not name:
        raise ValueError("数据集名称不能为空")
    if len(name) > 80:
        raise ValueError("数据集名称过长")
    class_list = _parse_classes(classes)
    notes = (notes or "").strip()[:500]
    ds_id = _new_id("ds")
    now = db.utcnow()
    ensure_dataset_dirs(ds_id)
    with db._lock, db.connect() as conn:
        conn.execute(
            """INSERT INTO dataset (id, name, classes, notes, created_at, updated_at)
               VALUES (?,?,?,?,?,?)""",
            (ds_id, name, db.dumps_json(class_list), notes, now, now),
        )
    return get_dataset(ds_id)


def list_datasets(limit: int = 200) -> list[dict]:
    with db._lock, db.connect() as conn:
        rows = conn.execute(
            "SELECT * FROM dataset ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
    result = []
    for r in rows:
        d = serialize_dataset(dict(r), get_dataset_stats(r["id"]))
        result.append(d)
    return result


def get_dataset(dataset_id: str) -> dict:
    with db._lock, db.connect() as conn:
        row = conn.execute(
            "SELECT * FROM dataset WHERE id=?", (dataset_id,)
        ).fetchone()
    if not row:
        raise LookupError("数据集不存在")
    return serialize_dataset(dict(row), get_dataset_stats(dataset_id))


def update_dataset(
    dataset_id: str,
    *,
    name: Optional[str] = None,
    classes: Optional[list] = None,
    notes: Optional[str] = None,
) -> dict:
    ds = get_dataset(dataset_id)
    new_name = ds["name"]
    new_classes = ds["classes"]
    new_notes = ds["notes"]
    if name is not None:
        new_name = " ".join(name.strip().split())
        if not new_name:
            raise ValueError("数据集名称不能为空")
    if classes is not None:
        new_classes = _parse_classes(classes)
        # 类别变少时清理越界框
        if len(new_classes) < len(ds["classes"]):
            with db._lock, db.connect() as conn:
                conn.execute(
                    """DELETE FROM annotation WHERE image_id IN
                       (SELECT id FROM image WHERE dataset_id=?)
                       AND class_idx >= ?""",
                    (dataset_id, len(new_classes)),
                )
                # 刷新 box_count
                conn.execute(
                    """UPDATE image SET box_count=(
                         SELECT COUNT(*) FROM annotation WHERE annotation.image_id=image.id
                       ) WHERE dataset_id=?""",
                    (dataset_id,),
                )
    if notes is not None:
        new_notes = notes.strip()[:500]
    now = db.utcnow()
    with db._lock, db.connect() as conn:
        conn.execute(
            """UPDATE dataset SET name=?, classes=?, notes=?, updated_at=? WHERE id=?""",
            (new_name, db.dumps_json(new_classes), new_notes, now, dataset_id),
        )
    return get_dataset(dataset_id)


def delete_dataset(dataset_id: str) -> None:
    get_dataset(dataset_id)
    with db._lock, db.connect() as conn:
        conn.execute(
            "DELETE FROM annotation WHERE image_id IN (SELECT id FROM image WHERE dataset_id=?)",
            (dataset_id,),
        )
        conn.execute("DELETE FROM image WHERE dataset_id=?", (dataset_id,))
        conn.execute("DELETE FROM dataset WHERE id=?", (dataset_id,))
    root = dataset_root(dataset_id)
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size  # w, h


def _unique_name(images_dir: Path, origin: str, used: set[str]) -> str:
    safe = _safe_filename(origin)
    root, ext = Path(safe).stem, Path(safe).suffix.lower()
    if ext not in config.IMAGE_EXTS:
        ext = ".jpg"
    root = root or "image"
    candidate = f"{root}{ext}"
    i = 1
    while candidate.lower() in used or (images_dir / candidate).exists():
        candidate = f"{root}_{i}{ext}"
        i += 1
    used.add(candidate.lower())
    return candidate


def _insert_image(
    conn,
    *,
    dataset_id: str,
    filename: str,
    rel_path: str,
    width: int,
    height: int,
    sha1: str,
    source: str,
    group_key: str = "",
    phash: str = "",
) -> str:
    img_id = _new_id("img")
    now = db.utcnow()
    conn.execute(
        """INSERT INTO image
           (id, dataset_id, filename, rel_path, width, height, sha1, phash,
            group_key, source, split, review_status, box_count, uncertainty, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            img_id,
            dataset_id,
            filename,
            rel_path,
            width,
            height,
            sha1,
            phash,
            group_key or None,
            source,
            "",
            "unlabeled",
            0,
            0,
            now,
        ),
    )
    return img_id


def import_files(
    dataset_id: str,
    files: list[tuple[str, bytes]],
    *,
    source: str = "upload",
    group_key: str = "",
) -> dict:
    """files: [(filename, content_bytes), ...]"""
    ds = get_dataset(dataset_id)
    root = ensure_dataset_dirs(dataset_id)
    images_dir = root / "images"
    used = {p.name.lower() for p in images_dir.iterdir() if p.is_file()}
    imported = 0
    skipped_dup = 0
    skipped_bad = 0
    image_ids: list[str] = []

    with db._lock, db.connect() as conn:
        existing_sha = {
            r["sha1"]
            for r in conn.execute(
                "SELECT sha1 FROM image WHERE dataset_id=? AND sha1 IS NOT NULL",
                (dataset_id,),
            ).fetchall()
        }
        for fname, content in files:
            ext = Path(fname).suffix.lower()
            if ext not in config.IMAGE_EXTS:
                skipped_bad += 1
                continue
            if len(content) > config.MAX_UPLOAD_BYTES:
                skipped_bad += 1
                continue
            sha1 = hashlib.sha1(content).hexdigest()
            if sha1 in existing_sha:
                skipped_dup += 1
                continue
            filename = _unique_name(images_dir, fname, used)
            dest = images_dir / filename
            dest.write_bytes(content)
            try:
                w, h = _image_size(dest)
            except Exception:
                dest.unlink(missing_ok=True)
                skipped_bad += 1
                continue
            img_id = _insert_image(
                conn,
                dataset_id=dataset_id,
                filename=filename,
                rel_path=filename,
                width=w,
                height=h,
                sha1=sha1,
                source=source,
                group_key=group_key,
            )
            existing_sha.add(sha1)
            image_ids.append(img_id)
            imported += 1
        conn.execute(
            "UPDATE dataset SET updated_at=? WHERE id=?", (db.utcnow(), dataset_id)
        )

    return {
        "dataset_id": ds["id"],
        "imported": imported,
        "skipped_dup": skipped_dup,
        "skipped_bad": skipped_bad,
        "image_ids": image_ids,
    }


def import_zip(dataset_id: str, zip_bytes: bytes) -> dict:
    """支持纯图片 ZIP，或 YOLO 格式（images/ + labels/ + 可选 classes.txt）。"""
    get_dataset(dataset_id)
    root = ensure_dataset_dirs(dataset_id)
    images_dir = root / "images"
    used = {p.name.lower() for p in images_dir.iterdir() if p.is_file()}

    labels_map: dict[str, list[dict]] = {}  # stem -> boxes
    classes_from_zip: list[str] = []
    file_blobs: list[tuple[str, bytes, str]] = []  # name, bytes, arcname

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = info.filename.replace("\\", "/")
                # 防目录穿越
                if ".." in name.split("/"):
                    continue
                base = Path(name).name
                ext = Path(base).suffix.lower()
                lower = name.lower()
                if base.lower() == "classes.txt":
                    text = zf.read(info).decode("utf-8", errors="ignore")
                    classes_from_zip = [
                        ln.strip() for ln in text.splitlines() if ln.strip()
                    ]
                    continue
                if ext == ".txt" and ("label" in lower or "/labels/" in lower):
                    text = zf.read(info).decode("utf-8", errors="ignore")
                    boxes = []
                    for line in text.splitlines():
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        try:
                            boxes.append(
                                {
                                    "class_idx": int(float(parts[0])),
                                    "cx": float(parts[1]),
                                    "cy": float(parts[2]),
                                    "w": float(parts[3]),
                                    "h": float(parts[4]),
                                    "conf": 1.0,
                                    "source": "manual",
                                }
                            )
                        except ValueError:
                            continue
                    labels_map[Path(base).stem] = boxes
                    continue
                if ext in config.IMAGE_EXTS:
                    file_blobs.append((base, zf.read(info), name))
    except zipfile.BadZipFile as exc:
        raise ValueError("不是有效的 ZIP 文件") from exc

    if classes_from_zip:
        try:
            update_dataset(dataset_id, classes=classes_from_zip)
        except ValueError:
            pass

    imported = 0
    labeled = 0
    skipped_dup = 0
    skipped_bad = 0

    with db._lock, db.connect() as conn:
        existing_sha = {
            r["sha1"]
            for r in conn.execute(
                "SELECT sha1 FROM image WHERE dataset_id=? AND sha1 IS NOT NULL",
                (dataset_id,),
            ).fetchall()
        }
        for base, content, _arc in file_blobs:
            sha1 = hashlib.sha1(content).hexdigest()
            if sha1 in existing_sha:
                skipped_dup += 1
                continue
            filename = _unique_name(images_dir, base, used)
            dest = images_dir / filename
            dest.write_bytes(content)
            try:
                w, h = _image_size(dest)
            except Exception:
                dest.unlink(missing_ok=True)
                skipped_bad += 1
                continue
            img_id = _insert_image(
                conn,
                dataset_id=dataset_id,
                filename=filename,
                rel_path=filename,
                width=w,
                height=h,
                sha1=sha1,
                source="zip",
            )
            existing_sha.add(sha1)
            imported += 1
            stem = Path(base).stem
            boxes = labels_map.get(stem) or labels_map.get(Path(filename).stem)
            if boxes:
                now = db.utcnow()
                for b in boxes:
                    conn.execute(
                        """INSERT INTO annotation
                           (image_id, class_idx, cx, cy, w, h, conf, source, created_at)
                           VALUES (?,?,?,?,?,?,?,?,?)""",
                        (
                            img_id,
                            b["class_idx"],
                            b["cx"],
                            b["cy"],
                            b["w"],
                            b["h"],
                            b.get("conf", 1.0),
                            "manual",
                            now,
                        ),
                    )
                conn.execute(
                    """UPDATE image SET box_count=?, review_status='reviewed' WHERE id=?""",
                    (len(boxes), img_id),
                )
                labeled += 1
        conn.execute(
            "UPDATE dataset SET updated_at=? WHERE id=?", (db.utcnow(), dataset_id)
        )

    return {
        "imported": imported,
        "labeled": labeled,
        "skipped_dup": skipped_dup,
        "skipped_bad": skipped_bad,
    }


def list_images(
    dataset_id: str,
    *,
    status: Optional[str] = None,
    split: Optional[str] = None,
    q: Optional[str] = None,
    sort: str = "created",
    page: int = 1,
    page_size: int = 50,
) -> dict:
    get_dataset(dataset_id)
    page = max(1, page)
    page_size = max(1, min(200, page_size))
    where = ["dataset_id=?"]
    params: list[Any] = [dataset_id]
    if status:
        where.append("review_status=?")
        params.append(status)
    if split:
        where.append("split=?")
        params.append(split)
    if q:
        where.append("filename LIKE ?")
        params.append(f"%{q}%")
    where_sql = " AND ".join(where)
    order = {
        "uncertainty": "uncertainty DESC, created_at ASC",
        "filename": "filename ASC",
        "created": "created_at ASC",
    }.get(sort, "created_at ASC")

    with db._lock, db.connect() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) AS c FROM image WHERE {where_sql}", params
        ).fetchone()["c"]
        rows = conn.execute(
            f"""SELECT * FROM image WHERE {where_sql}
                ORDER BY {order}
                LIMIT ? OFFSET ?""",
            params + [page_size, (page - 1) * page_size],
        ).fetchall()
    items = [dict(r) for r in rows]
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": items,
    }


def get_image(image_id: str) -> dict:
    with db._lock, db.connect() as conn:
        row = conn.execute("SELECT * FROM image WHERE id=?", (image_id,)).fetchone()
    if not row:
        raise LookupError("图片不存在")
    return dict(row)


def image_file_path(image: dict) -> Path:
    return dataset_root(image["dataset_id"]) / "images" / image["rel_path"]


def thumb_path(image: dict) -> Path:
    return dataset_root(image["dataset_id"]) / "thumbs" / f"{image['id']}.jpg"


def ensure_thumb(image: dict) -> Path:
    tp = thumb_path(image)
    if tp.is_file():
        return tp
    src = image_file_path(image)
    if not src.is_file():
        raise FileNotFoundError("原图文件丢失")
    tp.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        if w > config.THUMB_WIDTH:
            nh = int(h * config.THUMB_WIDTH / w)
            im = im.resize((config.THUMB_WIDTH, max(1, nh)), Image.Resampling.LANCZOS)
        im.save(tp, "JPEG", quality=85)
    return tp


def get_annotations(image_id: str) -> list[dict]:
    with db._lock, db.connect() as conn:
        rows = conn.execute(
            "SELECT * FROM annotation WHERE image_id=? ORDER BY id", (image_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def put_annotations(
    image_id: str,
    boxes: list[dict],
    review_status: Optional[str] = None,
) -> dict:
    img = get_image(image_id)
    ds = get_dataset(img["dataset_id"])
    n_cls = len(ds["classes"])
    cleaned = []
    for b in boxes:
        ci = int(b.get("class_idx", 0))
        if ci < 0 or ci >= n_cls:
            raise ValueError(
                f"类别索引 {ci} 无效，当前数据集只有 {n_cls} 个类别（0~{n_cls-1}）"
            )
        cx, cy, w, h = float(b["cx"]), float(b["cy"]), float(b["w"]), float(b["h"])
        if w <= 0 or h <= 0:
            continue
        cleaned.append(
            {
                "class_idx": ci,
                "cx": min(1.0, max(0.0, cx)),
                "cy": min(1.0, max(0.0, cy)),
                "w": min(1.0, max(0.0, w)),
                "h": min(1.0, max(0.0, h)),
                "conf": float(b.get("conf", 1.0)),
                "source": b.get("source") or "manual",
            }
        )
    status = review_status
    if status is None:
        status = "reviewed" if cleaned else "unlabeled"
    if status not in REVIEW_STATUSES:
        raise ValueError(f"非法状态: {status}")

    now = db.utcnow()
    with db._lock, db.connect() as conn:
        conn.execute("DELETE FROM annotation WHERE image_id=?", (image_id,))
        for b in cleaned:
            conn.execute(
                """INSERT INTO annotation
                   (image_id, class_idx, cx, cy, w, h, conf, source, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    image_id,
                    b["class_idx"],
                    b["cx"],
                    b["cy"],
                    b["w"],
                    b["h"],
                    b["conf"],
                    b["source"],
                    now,
                ),
            )
        conn.execute(
            """UPDATE image SET box_count=?, review_status=? WHERE id=?""",
            (len(cleaned), status, image_id),
        )
        conn.execute(
            "UPDATE dataset SET updated_at=? WHERE id=?",
            (now, img["dataset_id"]),
        )
    return {"image_id": image_id, "box_count": len(cleaned), "review_status": status}


def confirm_image(image_id: str) -> dict:
    boxes = get_annotations(image_id)
    return put_annotations(
        image_id,
        boxes,
        review_status="confirmed",
    )


def clear_auto_annotations(dataset_id: str) -> dict:
    """只删 source='auto' 的预标注框，手工标注不动。

    若某图删完 auto 后没有任何框，review_status 改回 unlabeled，box_count 归零；
    若还剩 manual 框，重算 box_count，状态保留 confirmed/reviewed。
    """
    get_dataset(dataset_id)
    with db._lock, db.connect() as conn:
        auto_cnt = conn.execute(
            """SELECT COUNT(*) AS c FROM annotation a
               JOIN image i ON i.id=a.image_id
               WHERE i.dataset_id=? AND a.source='auto'""",
            (dataset_id,),
        ).fetchone()["c"]
        # 受影响的图片
        img_ids = [
            r["id"]
            for r in conn.execute(
                """SELECT DISTINCT i.id FROM image i
                   JOIN annotation a ON a.image_id=i.id
                   WHERE i.dataset_id=? AND a.source='auto'""",
                (dataset_id,),
            ).fetchall()
        ]
        conn.execute(
            """DELETE FROM annotation WHERE source='auto' AND image_id IN (
                 SELECT id FROM image WHERE dataset_id=?
               )""",
            (dataset_id,),
        )
        for iid in img_ids:
            rem = conn.execute(
                "SELECT COUNT(*) AS c FROM annotation WHERE image_id=?",
                (iid,),
            ).fetchone()["c"]
            if rem == 0:
                conn.execute(
                    """UPDATE image SET box_count=0, review_status='unlabeled',
                       uncertainty=0 WHERE id=?""",
                    (iid,),
                )
            else:
                conn.execute(
                    "UPDATE image SET box_count=? WHERE id=?",
                    (rem, iid),
                )
        conn.execute(
            "UPDATE dataset SET updated_at=? WHERE id=?",
            (db.utcnow(), dataset_id),
        )
    return {
        "dataset_id": dataset_id,
        "deleted_auto_boxes": int(auto_cnt or 0),
        "affected_images": len(img_ids),
    }


def next_unlabeled(dataset_id: str, after: Optional[str] = None) -> Optional[dict]:
    get_dataset(dataset_id)
    with db._lock, db.connect() as conn:
        if after:
            # 按 uncertainty 降序找 after 之后的下一张
            cur = conn.execute(
                "SELECT uncertainty, created_at FROM image WHERE id=?", (after,)
            ).fetchone()
            if cur:
                row = conn.execute(
                    """SELECT * FROM image
                       WHERE dataset_id=? AND review_status IN ('unlabeled','auto')
                       AND (uncertainty < ? OR (uncertainty = ? AND created_at > ?)
                            OR (uncertainty = ? AND created_at = ? AND id > ?))
                       ORDER BY uncertainty DESC, created_at ASC, id ASC
                       LIMIT 1""",
                    (
                        dataset_id,
                        cur["uncertainty"],
                        cur["uncertainty"],
                        cur["created_at"],
                        cur["uncertainty"],
                        cur["created_at"],
                        after,
                    ),
                ).fetchone()
                if row:
                    return dict(row)
        row = conn.execute(
            """SELECT * FROM image
               WHERE dataset_id=? AND review_status IN ('unlabeled','auto')
               ORDER BY uncertainty DESC, created_at ASC
               LIMIT 1""",
            (dataset_id,),
        ).fetchone()
    return dict(row) if row else None


def touch_dataset(dataset_id: str) -> None:
    with db._lock, db.connect() as conn:
        conn.execute(
            "UPDATE dataset SET updated_at=? WHERE id=?", (db.utcnow(), dataset_id)
        )
