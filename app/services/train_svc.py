"""训练任务：导出数据集、subprocess 调 yolo、日志 SSE、断点续训、耗时预估。"""

from __future__ import annotations

import csv
import logging
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from app import config, db
from app.infer import engine
from app.services import export_svc

logger = logging.getLogger(__name__)

_EPOCH_RE = re.compile(r"\s*(\d+)/(\d+)\s+")


def new_job_id() -> str:
    return f"train_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"


def choose_cache(num_images: int, imgsz: int) -> str | bool:
    """按内存预估自动选 cache。"""
    est = num_images * imgsz * imgsz * 3 * 1.2
    gb = est / (1024**3)
    if gb < 8:
        return "ram"
    if gb < 16:
        return "disk"
    return False


def estimate_seconds(
    *,
    num_images: int,
    epochs: int,
    imgsz: int,
    base_model: str,
    freeze: int,
    cache: str | bool,
) -> dict:
    """经验公式预估总秒数。"""
    base = config.ESTIMATE_BASE_SEC_PER_EPOCH
    img_factor = max(1, num_images) / config.ESTIMATE_BASE_IMAGES
    size_factor = (imgsz / 640.0) ** 2
    name = base_model.lower()
    if "yolo26n" in name or name.endswith("n.pt") or "/n" in name:
        model_factor = 1.0
    elif "yolo26s" in name or name.endswith("s.pt"):
        model_factor = 3.0
    elif name.endswith("m.pt") or "yolo26m" in name:
        model_factor = 7.0
    else:
        model_factor = 2.0
    freeze_factor = 0.55 if freeze and int(freeze) > 0 else 1.0
    if cache == "ram":
        cache_factor = 0.78
    elif cache == "disk":
        cache_factor = 0.9
    else:
        cache_factor = 1.0

    sec_per_epoch = base * img_factor * size_factor * model_factor * freeze_factor * cache_factor
    total = sec_per_epoch * max(1, epochs)
    tips = []
    if total > config.ESTIMATE_WARN_HOURS * 3600:
        tips.append(
            f"⚠️ 预计需要 {total/3600:.1f} 小时。"
            "建议：把底模换成 yolo26n，或把图片尺寸降到 416，或开启 freeze=10。"
        )
    blocked = total > config.ESTIMATE_BLOCK_HOURS * 3600
    if blocked:
        tips.append("预计超过 24 小时，需勾选「我知道会很久，仍然继续」才能提交。")
    return {
        "seconds": total,
        "seconds_per_epoch": sec_per_epoch,
        "human": _fmt_duration(total),
        "cache": cache if cache is not False else "off",
        "tips": tips,
        "blocked": blocked,
        "warn": total > config.ESTIMATE_WARN_HOURS * 3600,
    }


def _fmt_duration(sec: float) -> str:
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} 小时 {m} 分"
    if m:
        return f"{m} 分 {s} 秒"
    return f"{s} 秒"


def resolve_base_model(base_model: str) -> tuple[Path, list[str] | None, bool]:
    """返回 (path, classes_or_None, is_finetune)。

    若 base_model 是模型仓库 id，则读注册表。
    """
    # 仓库模型
    with db._lock, db.connect() as conn:
        row = conn.execute("SELECT * FROM model WHERE id=?", (base_model,)).fetchone()
    if row:
        p = Path(row["path"])
        if not p.is_file():
            raise FileNotFoundError(f"注册模型文件丢失: {row['path']}")
        classes = db.loads_json(row["classes"], [])
        return p, classes, True

    # 权重名 / 路径
    p = config.resolve_weight_path(base_model)
    # 内置微调底模
    name = p.name.lower()
    is_ft = "wheelie" in name or "multi-rider" in name
    classes = None
    if is_ft:
        try:
            classes = engine.get_model_class_names(p)
        except Exception:
            classes = ["wheelie", "multi rider"]
    return p, classes, is_ft


def check_finetune_compat(base_classes: list[str] | None, ds_classes: list[str]) -> None:
    if base_classes is None:
        return
    if len(base_classes) != len(ds_classes):
        raise ValueError(
            f"这个模型认识的是 {base_classes}，但你的数据集有 {len(ds_classes)} 个类别 "
            f"{ds_classes}，类别对不上，不能在它基础上继续训练。请改用从零开始的底模"
            f"（如 yolo26n.pt）。"
        )
    # 顺序一致校验（宽松：归一化后比）
    from app.services.autolabel_svc import normalize_token

    for a, b in zip(base_classes, ds_classes):
        if normalize_token(a) != normalize_token(b):
            raise ValueError(
                f"类别顺序不一致：模型是 {base_classes}，数据集是 {ds_classes}。"
                "请调整数据集类别顺序，或改用从零开始的底模。"
            )


def create_train_job(params: dict) -> dict:
    dataset_id = params["dataset_id"]
    from app.services import dataset_svc

    dataset = dataset_svc.get_dataset(dataset_id)
    only_confirmed = bool(params.get("only_confirmed", True))
    # 校验
    report = export_svc.validate_for_train(
        dataset_id, only_confirmed=only_confirmed, min_boxes_per_class=1
    )
    if not report["ok"]:
        raise ValueError("训练数据不足：每个类别至少要有 1 个框，且至少 2 张可用图")

    base_path, base_classes, is_ft = resolve_base_model(params.get("base_model", "yolo26n.pt"))
    check_finetune_compat(base_classes if is_ft else None, dataset["classes"])

    epochs = int(params.get("epochs", config.DEFAULT_EPOCHS))
    if is_ft and "epochs" not in params:
        epochs = 20
    imgsz = int(params.get("imgsz", config.DEFAULT_IMGSZ))
    batch = int(params.get("batch", config.DEFAULT_BATCH))
    freeze = int(params.get("freeze", config.DEFAULT_FREEZE))
    workers = int(params.get("workers", config.TRAIN_WORKERS))
    patience = int(params.get("patience", config.DEFAULT_PATIENCE))
    val_ratio = float(params.get("val_ratio", 0.2))
    force_long = bool(params.get("force_long", False))

    n_img = report["image_count"]
    cache = choose_cache(n_img, imgsz)
    est = estimate_seconds(
        num_images=n_img,
        epochs=epochs,
        imgsz=imgsz,
        base_model=str(base_path),
        freeze=freeze,
        cache=cache,
    )
    if est["blocked"] and not force_long:
        raise ValueError(
            est["tips"][-1] if est["tips"] else "预计训练超过 24 小时，请确认后勾选继续"
        )

    # 同一时刻只允许一个训练
    with db._lock, db.connect() as conn:
        running = conn.execute(
            "SELECT id FROM train_job WHERE status='running' LIMIT 1"
        ).fetchone()
    if running:
        raise RuntimeError(
            f"已有训练任务 {running['id']} 在运行。CPU 训练同一时刻只允许一个，请等待完成或取消后再提交。"
        )

    job_id = new_job_id()
    run_dir = config.RUNS_DIR / job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    full_params = {
        **params,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "freeze": freeze,
        "workers": workers,
        "patience": patience,
        "val_ratio": val_ratio,
        "cache": cache if cache is not False else False,
        "device": "cpu",
        "amp": False,
        "base_model_path": str(base_path),
        "is_finetune": is_ft,
        "estimate": est,
    }
    now = db.utcnow()
    with db._lock, db.connect() as conn:
        conn.execute(
            """INSERT INTO train_job
               (id, dataset_id, base_model, params, run_dir, log_path, status,
                metrics, best_pt, last_epoch, resume_from, eta_seconds, pid,
                created_at, finished_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                job_id,
                dataset_id,
                str(base_path),
                db.dumps_json(full_params),
                str(run_dir),
                str(log_path),
                "pending",
                None,
                None,
                0,
                None,
                est["seconds"],
                None,
                now,
                None,
            ),
        )

    # 走通用任务队列
    from app import tasks as task_mod

    task_mod.create_task(
        "train",
        {"job_id": job_id},
        dataset_id=dataset_id,
        submit=True,
    )
    return get_job(job_id)


def get_job(job_id: str) -> dict:
    with db._lock, db.connect() as conn:
        row = conn.execute("SELECT * FROM train_job WHERE id=?", (job_id,)).fetchone()
    if not row:
        raise LookupError("训练任务不存在")
    d = dict(row)
    d["params"] = db.loads_json(d.get("params"), {})
    d["metrics"] = db.loads_json(d.get("metrics"), {})
    return d


def list_jobs(limit: int = 50) -> list[dict]:
    with db._lock, db.connect() as conn:
        rows = conn.execute(
            "SELECT * FROM train_job ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["params"] = db.loads_json(d.get("params"), {})
        d["metrics"] = db.loads_json(d.get("metrics"), {})
        out.append(d)
    return out


def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    cols, vals = [], []
    for k, v in fields.items():
        if k in ("params", "metrics") and not isinstance(v, str):
            v = db.dumps_json(v)
        cols.append(f"{k}=?")
        vals.append(v)
    vals.append(job_id)
    with db._lock, db.connect() as conn:
        conn.execute(f"UPDATE train_job SET {', '.join(cols)} WHERE id=?", vals)


def append_log(log_path: str | Path, msg: str) -> None:
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


def _resolve_yolo_cmd() -> list[str]:
    """跨平台：用当前解释器 -m ultralytics。"""
    return [sys.executable, "-m", "ultralytics"]


def run_train_job(task: dict) -> None:
    """任务处理器入口。"""
    from app import tasks as task_mod

    params = task.get("params") or {}
    job_id = params.get("job_id")
    job = get_job(job_id)
    _execute_training(job, resume=False)
    task_mod.set_progress(task["id"], 100, "训练完成")


def _execute_training(job: dict, *, resume: bool) -> None:
    job_id = job["id"]
    p = job.get("params") or {}
    run_dir = Path(job["run_dir"])
    log_path = Path(job["log_path"])
    dataset_id = job["dataset_id"]

    update_job(job_id, status="running", finished_at=None)
    append_log(log_path, f"开始训练 job={job_id} resume={resume}")

    if not resume:
        exp = export_svc.export_yolo_dataset(
            dataset_id,
            only_confirmed=bool(p.get("only_confirmed", True)),
            val_ratio=float(p.get("val_ratio", 0.2)),
            out_name=job_id,
        )
        append_log(
            log_path,
            f"导出完成 train={exp['train_count']} val={exp['val_count']} yaml={exp['data_yaml']}",
        )
        append_log(log_path, f"cache 策略: {p.get('cache')}（按内存自动判定）")
        data_yaml = exp["data_yaml"]
        model_path = p.get("base_model_path") or job["base_model"]
    else:
        data_yaml = None
        model_path = job.get("resume_from") or str(run_dir / "weights" / "last.pt")
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"找不到续训权重: {model_path}")

    epochs = int(p.get("epochs", 40))
    imgsz = int(p.get("imgsz", 416))
    batch = int(p.get("batch", 16))
    freeze = int(p.get("freeze", 10))
    workers = int(p.get("workers", 8))
    patience = int(p.get("patience", 10))
    cache = p.get("cache", False)
    if cache is False or cache == "off":
        cache_arg = "False"
    else:
        cache_arg = str(cache)

    cmd = _resolve_yolo_cmd() + [
        "detect",
        "train",
        f"model={model_path}",
        f"epochs={epochs}",
        f"imgsz={imgsz}",
        f"batch={batch}",
        f"project={config.RUNS_DIR.resolve().as_posix()}",
        f"name={job_id}",
        "exist_ok=True",
        f"workers={workers}",
        f"cache={cache_arg}",
        "device=cpu",
        "amp=False",
        f"patience={patience}",
        "verbose=True",
    ]
    if freeze and freeze > 0 and not resume:
        cmd.append(f"freeze={freeze}")
    if resume:
        cmd.append("resume=True")
    else:
        cmd.append(f"data={data_yaml}")

    append_log(log_path, "命令: " + " ".join(cmd))
    env = os.environ.copy()
    env["YOLO_OFFLINE"] = "1"
    env["OMP_NUM_THREADS"] = str(config.CPU_THREADS)
    env["MKL_NUM_THREADS"] = str(config.CPU_THREADS)

    with open(log_path, "a", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd=str(config.BASE_DIR),
            env=env,
            start_new_session=True,
        )
    update_job(job_id, pid=proc.pid)

    # 轮询进度
    last_epoch = 0
    t0 = time.time()
    while True:
        ret = proc.poll()
        # 解析 epoch
        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore")[-8000:]
            for m in _EPOCH_RE.finditer(text):
                last_epoch = max(last_epoch, int(m.group(1)))
        except Exception:
            pass
        if last_epoch > 0:
            elapsed = time.time() - t0
            sec_per = elapsed / max(last_epoch, 1)
            remain = sec_per * max(0, epochs - last_epoch)
            update_job(
                job_id,
                last_epoch=last_epoch,
                eta_seconds=remain,
                resume_from=str(run_dir / "weights" / "last.pt"),
            )
        if ret is not None:
            break
        time.sleep(5)

    if proc.returncode != 0:
        update_job(
            job_id,
            status="failed",
            finished_at=db.utcnow(),
            pid=None,
        )
        append_log(log_path, f"训练失败 exit={proc.returncode}")
        raise RuntimeError(f"训练进程退出码 {proc.returncode}，请查看日志")

    # 收集产物
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    metrics = _read_metrics(run_dir / "results.csv")
    update_job(
        job_id,
        status="success",
        best_pt=str(best) if best.is_file() else (str(last) if last.is_file() else None),
        metrics=metrics,
        last_epoch=epochs,
        eta_seconds=0,
        finished_at=db.utcnow(),
        pid=None,
        resume_from=str(last) if last.is_file() else None,
    )
    append_log(log_path, f"训练成功 metrics={metrics}")


def _read_metrics(csv_path: Path) -> dict:
    if not csv_path.is_file():
        return {}
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}
        last = rows[-1]

        def g(*keys):
            for k in keys:
                if k in last and last[k] not in ("", None):
                    try:
                        return float(last[k])
                    except ValueError:
                        return last[k]
            return None

        return {
            "epoch": g("epoch"),
            "precision": g("metrics/precision(B)", "metrics/precision"),
            "recall": g("metrics/recall(B)", "metrics/recall"),
            "mAP50": g("metrics/mAP50(B)", "metrics/mAP50"),
            "mAP50-95": g("metrics/mAP50-95(B)", "metrics/mAP50-95"),
            "box_loss": g("train/box_loss"),
            "cls_loss": g("train/cls_loss"),
        }
    except Exception:
        return {}


def read_metrics_series(job_id: str) -> list[dict]:
    job = get_job(job_id)
    csv_path = Path(job["run_dir"]) / "results.csv"
    if not csv_path.is_file():
        return []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    series = []
    for r in rows:
        item = {"epoch": r.get("epoch")}
        for k, outk in [
            ("train/box_loss", "box_loss"),
            ("train/cls_loss", "cls_loss"),
            ("metrics/mAP50(B)", "mAP50"),
            ("metrics/mAP50-95(B)", "mAP50-95"),
            ("metrics/precision(B)", "precision"),
            ("metrics/recall(B)", "recall"),
        ]:
            if k in r and r[k] not in ("", None):
                try:
                    item[outk] = float(r[k])
                except ValueError:
                    pass
        series.append(item)
    return series


def cancel_job(job_id: str) -> dict:
    job = get_job(job_id)
    if job["status"] != "running":
        raise ValueError("只有运行中的任务可以取消")
    pid = job.get("pid")
    if pid:
        try:
            os.killpg(pid, signal.SIGTERM)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception as exc:
                logger.warning("kill %s failed: %s", pid, exc)
    update_job(job_id, status="canceled", finished_at=db.utcnow(), pid=None)
    append_log(job["log_path"], "用户取消训练")
    return get_job(job_id)


def resume_job(job_id: str) -> dict:
    job = get_job(job_id)
    if job["status"] not in ("interrupted", "failed", "canceled"):
        if job["status"] == "running":
            raise ValueError("任务仍在运行")
        if job["status"] == "success":
            raise ValueError("任务已完成，无需续训")
    last = job.get("resume_from") or str(Path(job["run_dir"]) / "weights" / "last.pt")
    if not Path(last).is_file():
        raise FileNotFoundError("找不到 last.pt，无法续训")

    with db._lock, db.connect() as conn:
        running = conn.execute(
            "SELECT id FROM train_job WHERE status='running' LIMIT 1"
        ).fetchone()
    if running:
        raise RuntimeError("已有其他训练在运行")

    update_job(job_id, status="pending", finished_at=None)
    from app import tasks as task_mod

    def _handler_resume(task):
        j = get_job(job_id)
        _execute_training(j, resume=True)

    # 直接用 train 类型但自定义：提交专用闭包
    task_mod.register_handler("train_resume", _handler_resume)
    task_mod.create_task(
        "train_resume",
        {"job_id": job_id},
        dataset_id=job["dataset_id"],
        submit=True,
    )
    return get_job(job_id)


def recover_interrupted_jobs() -> int:
    """启动时扫描 running 任务，进程不在则标 interrupted。"""
    with db._lock, db.connect() as conn:
        rows = conn.execute(
            "SELECT * FROM train_job WHERE status='running'"
        ).fetchall()
    n = 0
    for r in rows:
        pid = r["pid"]
        alive = False
        if pid:
            try:
                os.kill(pid, 0)
                alive = True
            except OSError:
                alive = False
        if not alive:
            update_job(
                r["id"],
                status="interrupted",
                finished_at=db.utcnow(),
                pid=None,
            )
            append_log(r["log_path"] or (Path(r["run_dir"]) / "train.log"), "服务重启，任务中断")
            n += 1
    return n


def publish_job(job_id: str, name: Optional[str] = None, notes: str = "") -> dict:
    from app.services import model_svc

    job = get_job(job_id)
    if job["status"] != "success" or not job.get("best_pt"):
        raise ValueError("只有训练成功且存在 best.pt 的任务可以发布")
    from app.services import dataset_svc

    ds = dataset_svc.get_dataset(job["dataset_id"])
    return model_svc.register_model(
        path=job["best_pt"],
        name=name or f"{ds['name']}-{job_id[-6:]}",
        classes=ds["classes"],
        metrics=job.get("metrics") or {},
        from_job=job_id,
        notes=notes,
        export_ov=True,
        imgsz=int((job.get("params") or {}).get("imgsz", 416)),
    )
