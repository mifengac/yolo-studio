"""pHash 近重复检测。"""

from __future__ import annotations

import logging
from collections import defaultdict

from PIL import Image

from app import db
from app.services import dataset_svc

logger = logging.getLogger(__name__)


def _average_hash(path, hash_size: int = 8) -> str:
    with Image.open(path) as im:
        im = im.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
        pixels = list(im.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if p >= avg else "0" for p in pixels)
    # 转 16 进制
    return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"


def hamming(a: str, b: str) -> int:
    if not a or not b or len(a) != len(b):
        return 999
    x = int(a, 16) ^ int(b, 16)
    return bin(x).count("1")


def run_dedup(task: dict) -> None:
    from app import tasks as task_mod

    params = task.get("params") or {}
    dataset_id = task.get("dataset_id") or params.get("dataset_id")
    task_id = task["id"]
    threshold = int(params.get("threshold", 5))

    page = 1
    items: list[dict] = []
    while True:
        chunk = dataset_svc.list_images(
            dataset_id, page=page, page_size=200, sort="created"
        )
        if not chunk["items"]:
            break
        items.extend(chunk["items"])
        if page * 200 >= chunk["total"]:
            break
        page += 1

    total = len(items)
    hashes: list[tuple[dict, str]] = []
    for i, it in enumerate(items):
        path = dataset_svc.image_file_path(it)
        if not path.is_file():
            continue
        try:
            ph = _average_hash(path)
        except Exception:
            continue
        with db._lock, db.connect() as conn:
            conn.execute("UPDATE image SET phash=? WHERE id=?", (ph, it["id"]))
        hashes.append((it, ph))
        if (i + 1) % 20 == 0:
            task_mod.set_progress(
                task_id, 50.0 * (i + 1) / max(total, 1), f"计算哈希 {i+1}/{total}"
            )

    # 聚类：简单并查集
    parent = {h[0]["id"]: h[0]["id"] for h in hashes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    n = len(hashes)
    for i in range(n):
        for j in range(i + 1, n):
            if hamming(hashes[i][1], hashes[j][1]) <= threshold:
                union(hashes[i][0]["id"], hashes[j][0]["id"])
        if (i + 1) % 50 == 0:
            task_mod.set_progress(
                task_id, 50 + 40.0 * (i + 1) / max(n, 1), f"聚类 {i+1}/{n}"
            )

    groups: dict[str, list[dict]] = defaultdict(list)
    id_to_item = {h[0]["id"]: h[0] for h in hashes}
    for iid in parent:
        groups[find(iid)].append(id_to_item[iid])

    skipped = 0
    group_count = 0
    for root, members in groups.items():
        if len(members) < 2:
            continue
        group_count += 1
        gk = f"dup_{root[:12]}"
        # 保留第一张 unlabeled，其余 skipped（不覆盖 confirmed）
        members_sorted = sorted(members, key=lambda x: x.get("created_at") or "")
        keep = members_sorted[0]
        with db._lock, db.connect() as conn:
            conn.execute(
                "UPDATE image SET group_key=? WHERE id=?", (gk, keep["id"])
            )
            for m in members_sorted[1:]:
                if m.get("review_status") in ("confirmed", "reviewed"):
                    conn.execute(
                        "UPDATE image SET group_key=? WHERE id=?", (gk, m["id"])
                    )
                    continue
                conn.execute(
                    """UPDATE image SET group_key=?, review_status='skipped'
                       WHERE id=?""",
                    (gk, m["id"]),
                )
                skipped += 1

    task_mod.set_progress(task_id, 100, f"近重复组 {group_count}，跳过 {skipped} 张")
    task_mod.update_task(
        task_id, message=f"去重完成：{group_count} 组，跳过 {skipped} 张近重复图"
    )
