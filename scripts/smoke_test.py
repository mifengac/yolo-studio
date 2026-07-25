#!/usr/bin/env python3
"""冒烟测试：建数据集 → 导图 → 预标注 → 存标注 → 预估 → 短训 2 epoch（默认）。

用法：
  # 服务已在 5016 启动时（默认含训练；结束后清理本脚本创建的数据集/模型）
  python scripts/smoke_test.py

  # 跳过训练
  python scripts/smoke_test.py --no-train

  # 保留数据集与发布模型（调试用）
  python scripts/smoke_test.py --keep
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


def _print_train_log_tail(job: dict, n: int = 20) -> None:
    """训练失败时打印日志末尾，便于排查。"""
    log_path = job.get("log_path")
    if not log_path:
        run_dir = job.get("run_dir")
        if run_dir:
            log_path = str(Path(run_dir) / "train.log")
    if not log_path:
        print("   (无 log_path)")
        return
    p = Path(log_path)
    if not p.is_file():
        print(f"   日志不存在: {p}")
        return
    try:
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        print(f"   --- train.log 最后 {n} 行 ({p}) ---")
        for line in lines[-n:]:
            print("   |", line)
        print("   --- end ---")
    except Exception as exc:
        print(f"   读日志失败: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="跳过短训（默认会跑 2 epoch）",
    )
    # 兼容旧参数：--train 仍可写，无实际作用（默认已训）
    parser.add_argument(
        "--train",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="保留本脚本创建的数据集与发布模型（默认清理，避免仓库残留 smoke_*）",
    )
    parser.add_argument("--base", default=BASE)
    args = parser.parse_args()
    base = args.base.rstrip("/")
    do_train = not args.no_train
    keep = args.keep

    c = httpx.Client(base_url=base, timeout=120.0)
    published_id = None
    ds_id = None
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

    if do_train:
        print("7) short train 2 epoch imgsz=320（默认执行）")
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
            print("   失败：weights/ 下无可用检测底模，无法完成默认短训")
            return 1
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
        if j["status"] != "success":
            print("   训练失败 status=", j["status"], "error=", j.get("error") or j.get("message"))
            _print_train_log_tail(j, 20)
            return 1
        if not j.get("best_pt"):
            print("   训练结束但缺少 best_pt")
            _print_train_log_tail(j, 20)
            return 1
        print("   best_pt", j["best_pt"])
        try:
            m = c.post(f"/api/train/{job_id}/publish", json={}).json()
            published_id = m.get("id")
            print(
                "   published",
                m["id"],
                "openvino",
                m.get("openvino_path"),
                "ov_status",
                m.get("ov_status"),
                "(异步导出可能稍后才 ready)",
            )
        except Exception as exc:
            print("   publish 警告（非致命）:", exc)
    else:
        print("7) 已跳过训练（--no-train）")

    if not keep:
        print("8) 清理本脚本创建的资源")
        if published_id:
            try:
                c.delete(f"/api/models/{published_id}")
                print("   deleted model", published_id)
            except Exception as exc:
                print("   delete model:", exc)
        if ds_id:
            try:
                c.delete(f"/api/datasets/{ds_id}")
                print("   deleted dataset", ds_id)
            except Exception as exc:
                print("   delete dataset:", exc)
    else:
        print("8) --keep：保留数据集与模型")

    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())