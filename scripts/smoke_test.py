#!/usr/bin/env python3
"""冒烟测试：建数据集 → 导图 → 预标注 → 存标注 → 导出预估 →（可选）短训。

用法：
  # 服务已在 5016 启动时
  python scripts/smoke_test.py

  # 含 2 epoch 训练（需 weights 下可用底模；每类框数门槛放宽为 1）
  python scripts/smoke_test.py --train
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx

BASE = "http://127.0.0.1:5016"


def make_image_bytes(i: int) -> bytes:
    im = Image.new("RGB", (640, 480), (30 + i * 20, 40, 50))
    d = ImageDraw.Draw(im)
    d.rectangle([100 + i * 10, 80, 280 + i * 10, 260], outline=(255, 0, 0), width=3)
    d.text((120, 100), f"sample-{i}", fill=(255, 255, 0))
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    return buf.getvalue()


def wait_task(c: httpx.Client, task_id: str, timeout: float = 300.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        t = c.get(f"/api/tasks/{task_id}").json()
        if t.get("status") in ("success", "failed", "canceled"):
            return t
        time.sleep(1.5)
    raise TimeoutError(f"task {task_id} timeout")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="跑 2 epoch 短训")
    parser.add_argument("--base", default=BASE)
    args = parser.parse_args()
    base = args.base.rstrip("/")

    c = httpx.Client(base_url=base, timeout=120.0)
    print("1) health")
    r = c.get("/api/health")
    r.raise_for_status()
    print("   ", r.json())

    print("2) create dataset")
    r = c.post(
        "/api/datasets",
        json={"name": f"smoke_{int(time.time())}", "classes": ["wheelie", "multi rider"]},
    )
    r.raise_for_status()
    ds = r.json()
    ds_id = ds["id"]
    print("   ", ds_id)

    print("3) import 5 images")
    files = []
    for i in range(5):
        files.append(("files", (f"s{i}.jpg", make_image_bytes(i), "image/jpeg")))
    r = c.post(f"/api/datasets/{ds_id}/import/files", files=files)
    r.raise_for_status()
    print("   ", r.json())

    print("4) autolabel (model=default，无权重则跳过)")
    r = c.post(
        f"/api/datasets/{ds_id}/autolabel",
        json={"model": "default", "scope": "unlabeled", "conf": 0.25, "overwrite": False},
    )
    if r.status_code < 400:
        task = r.json()
        print("   task", task.get("id"), "model=", (task.get("params") or {}).get("model"))
        try:
            done = wait_task(c, task["id"], timeout=180)
            print("   autolabel", done.get("status"), done.get("message"), done.get("error"))
        except Exception as exc:
            print("   autolabel wait:", exc)
    else:
        print("   skip autolabel:", r.status_code, r.text[:200])

    print("5) list images + put annotations")
    r = c.get(f"/api/datasets/{ds_id}/images")
    r.raise_for_status()
    items = r.json()["items"]
    assert len(items) >= 5
    for img in items:
        # 每类至少 1 框，方便 --train 放宽门槛
        boxes = [
            {
                "class_idx": 0,
                "cx": 0.4,
                "cy": 0.4,
                "w": 0.2,
                "h": 0.25,
                "conf": 1.0,
                "source": "manual",
            },
            {
                "class_idx": 1,
                "cx": 0.6,
                "cy": 0.55,
                "w": 0.18,
                "h": 0.22,
                "conf": 1.0,
                "source": "manual",
            },
        ]
        rr = c.put(
            f"/api/images/{img['id']}/annotations",
            json={"boxes": boxes, "review_status": "confirmed"},
        )
        rr.raise_for_status()
    print("   annotated", len(items))

    print("6) train estimate")
    r = c.post(
        "/api/train/estimate",
        json={
            "dataset_id": ds_id,
            "base_model": "yolo26n.pt",
            "epochs": 2,
            "imgsz": 320,
            "freeze": 10,
            "only_confirmed": True,
        },
    )
    print("   estimate status", r.status_code, r.text[:240])

    if args.train:
        print("7) short train 2 epoch imgsz=320")
        info = c.get("/api/system/info").json()
        base_model = None
        for w in info.get("weights") or []:
            if w.get("available") and "yolo26n" in w["name"]:
                base_model = w["name"]
                break
        if not base_model:
            for w in info.get("weights") or []:
                if (
                    w.get("available")
                    and w["name"].endswith(".pt")
                    and "sam" not in w["name"]
                    and "world" not in w["name"]
                ):
                    base_model = w["name"]
                    break
        if not base_model:
            print("   跳过训练：weights/ 下无可用检测底模")
            print("SMOKE OK (no train)")
            return 0
        r = c.post(
            "/api/train",
            json={
                "dataset_id": ds_id,
                "base_model": base_model,
                "epochs": 2,
                "imgsz": 320,
                "batch": 4,
                "freeze": 10,
                "only_confirmed": True,
                "workers": 2,
                "force_long": True,
                "min_boxes_per_class": 1,
            },
        )
        if r.status_code >= 400:
            print("   train submit failed:", r.text)
            return 1
        job = r.json()
        job_id = job["id"]
        print("   job", job_id)
        j = job
        for _ in range(600):
            j = c.get(f"/api/train/{job_id}").json()
            print(f"   status={j['status']} epoch={j.get('last_epoch')}")
            if j["status"] in ("success", "failed", "canceled", "interrupted"):
                break
            time.sleep(5)
        assert j["status"] == "success", j
        assert j.get("best_pt"), "missing best_pt"
        print("   best_pt", j["best_pt"])
        m = c.post(f"/api/train/{job_id}/publish", json={}).json()
        print("   published", m["id"], "openvino", m.get("openvino_path"), "(异步导出可能稍后才有)")

    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
