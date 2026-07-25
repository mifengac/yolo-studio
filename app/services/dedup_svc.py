"""近重复检测：真 pHash + 分桶，避免 O(n²) 全比较。"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from PIL import Image

from app import db
from app.services import dataset_svc

logger = logging.getLogger(__name__)

# 哈希版本前缀；旧 aHash 无此前缀，迁移时清空
_PHASH_PREFIX = "p1:"


def _dct_2d(a: np.ndarray) -> np.ndarray:
    """二维 DCT-II（numpy 实现，不引 scipy）。"""
    # 行 DCT
    n = a.shape[0]
    m = a.shape[1]
    # 用 cv2 若可用更快
    try:
        import cv2

        return cv2.dct(a.astype(np.float32))
    except Exception:
        pass

    def dct1(x: np.ndarray) -> np.ndarray:
        N = x.shape[-1]
        out = np.zeros_like(x, dtype=np.float64)
        for k in range(N):
            alpha = np.sqrt(1.0 / N) if k == 0 else np.sqrt(2.0 / N)
            out[..., k] = alpha * np.sum(
                x * np.cos(np.pi * (np.arange(N) + 0.5) * k / N), axis=-1
            )
        return out

    return dct1(dct1(a.astype(np.float64)).T).T


def perceptual_hash(path, hash_size: int = 8) -> str:
    """pHash：32×32 → DCT → 左上 8×8（排除直流）→ 与中位数比较。

    返回带版本前缀的 16 进制串，例如 p1:a3f1...
    """
    with Image.open(path) as im:
        im = im.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        pixels = np.asarray(im, dtype=np.float32)
    dct = _dct_2d(pixels)
    # 左上 hash_size×hash_size，去掉 [0,0] 直流
    block = dct[:hash_size, :hash_size].copy()
    block[0, 0] = 0.0
    # 用非直流系数的中位数
    flat = block.flatten()
    med = float(np.median(flat[1:])) if flat.size > 1 else float(np.median(flat))
    bits = (block > med).astype(np.uint8).flatten()
    # 64 bit -> 16 hex
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    nhex = hash_size * hash_size // 4
    return f"{_PHASH_PREFIX}{val:0{nhex}x}"


def hamming(a: str, b: str) -> int:
    if not a or not b:
        return 999
    # 去掉版本前缀再比
    if a.startswith(_PHASH_PREFIX):
        a = a[len(_PHASH_PREFIX) :]
    if b.startswith(_PHASH_PREFIX):
        b = b[len(_PHASH_PREFIX) :]
    if len(a) != len(b):
        return 999
    x = int(a, 16) ^ int(b, 16)
    return bin(x).count("1")


def _bucket_keys(hex_hash: str) -> list[str]:
    """64 位哈希切 4 段 16 位，做倒排索引 key。"""
    h = hex_hash
    if h.startswith(_PHASH_PREFIX):
        h = h[len(_PHASH_PREFIX) :]
    # 16 hex chars = 64 bit；每段 4 hex = 16 bit
    if len(h) < 16:
        h = h.zfill(16)
    return [f"{i}:{h[i * 4 : (i + 1) * 4]}" for i in range(4)]


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
            ph = perceptual_hash(path)
        except Exception:
            continue
        with db._lock, db.connect() as conn:
            conn.execute("UPDATE image SET phash=? WHERE id=?", (ph, it["id"]))
        hashes.append((it, ph))
        if (i + 1) % 20 == 0:
            task_mod.set_progress(
                task_id, 50.0 * (i + 1) / max(total, 1), f"计算 pHash {i+1}/{total}"
            )

    # 并查集
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

    # 分桶：至少一段 16-bit 相同才进入精确比较
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, (_, ph) in enumerate(hashes):
        for k in _bucket_keys(ph):
            buckets[k].append(idx)

    compared: set[tuple[int, int]] = set()
    n = len(hashes)
    for bidx, members in buckets.items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if a > b:
                    a, b = b, a
                if (a, b) in compared:
                    continue
                compared.add((a, b))
                if hamming(hashes[a][1], hashes[b][1]) <= threshold:
                    union(hashes[a][0]["id"], hashes[b][0]["id"])

    task_mod.set_progress(
        task_id, 90, f"分桶候选对 {len(compared)}（全量两两约 {n*(n-1)//2}）"
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
