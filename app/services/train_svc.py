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
    min_boxes = int(params.get("min_boxes_per_class", 10))
    # 校验：每类至少 min_boxes 个框（默认 10；冒烟可传 1）
    report = export_svc.validate_for_train(
        dataset_id, only_confirmed=only_confirmed, min_boxes_per_class=min_boxes
    )
    if not report["ok"]:
        detail = "；".join(report.get("warnings") or []) or "数据不足"
        raise ValueError(
            f"训练数据不足：每个类别至少要有 {min_boxes} 个框，且至少 2 张可用图。"
            f"{detail}"
        )

    base_path, base_classes, is_ft = resolve_base_model(params.get("base_model", "yolo26n.pt"))
    check_finetune_compat(base_classes if is_ft else None, dataset["classes"])

    # epochs 未传时：微调 20 / 从零 40；前端显式传值时以客户端为准
    epochs_raw = params.get("epochs", None)
    if epochs_raw is None:
        epochs = 20 if is_ft else int(config.DEFAULT_EPOCHS)
    else:
        epochs = int(epochs_raw)
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
    # 绝对路径：启动恢复时靠 run_dir/job_id 匹配 /proc cmdline
    run_dir = (config.RUNS_DIR / job_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = (run_dir / "train.log").resolve()
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


def _resolve_yolo_bin() -> str:
    """定位 ultralytics 的 yolo 可执行文件（本地 venv 与容器均可）。

    注意：ultralytics 8.4 没有 __main__.py，不能用 `python -m ultralytics`。
    """
    import shutil

    cand = Path(sys.executable).parent / "yolo"
    if cand.is_file():
        return str(cand)
    found = shutil.which("yolo")
    if found:
        return found
    raise FileNotFoundError(
        "未找到 ultralytics 的 yolo 命令，请确认 ultralytics 已正确安装"
    )


def _resolve_yolo_cmd() -> list[str]:
    """训练/续训共用：返回 [yolo_bin]。"""
    return [_resolve_yolo_bin()]


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
        # 写入 params，续训后做 best.pt val 时还能找到 data.yaml
        p = {**p, "data_yaml": data_yaml}
        update_job(job_id, params=p)
        model_path = p.get("base_model_path") or job["base_model"]
    else:
        data_yaml = p.get("data_yaml")
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

    # 轮询进度：优先 results.csv 行数（日志里 tqdm 进度常不落盘）
    last_epoch = 0
    t0 = time.time()
    while True:
        ret = proc.poll()
        try:
            csv_n = _count_epochs(run_dir / "results.csv")
            if csv_n > last_epoch:
                last_epoch = csv_n
            # 日志兜底：只认「当前轮/总轮数」且 total==epochs
            text = log_path.read_text(encoding="utf-8", errors="ignore")[-8000:]
            for m in _EPOCH_RE.finditer(text):
                cur, total = int(m.group(1)), int(m.group(2))
                if total == epochs and 0 <= cur <= epochs:
                    last_epoch = max(last_epoch, cur)
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

    # 以 results.csv 实际行数为准；退出码 0 不代表跑满（睡眠/SIGTERM/早停都会干扰）
    actual = _count_epochs(run_dir / "results.csv")
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    resume_path = str(last) if last.is_file() else None

    # 用户已点取消：保留 canceled，只回填真实轮次
    try:
        cur = get_job(job_id)
        if cur.get("status") == "canceled":
            update_job(
                job_id,
                last_epoch=actual,
                pid=None,
                resume_from=resume_path,
            )
            append_log(
                log_path,
                f"训练已取消：目标 {epochs} 轮，实际完成 {actual} 轮（退出码 {proc.returncode}）",
            )
            return
    except Exception:
        pass

    # 正常收尾（含 patience 早停）会写 results.png；中途被杀则通常没有
    clean = _training_finished_cleanly(run_dir)
    incomplete = (proc.returncode != 0) or (actual < epochs and not clean)

    if incomplete:
        # last.pt 在且已有进度 → interrupted（可续训）；否则 failed
        can_resume = last.is_file() and actual > 0
        status = "interrupted" if can_resume else "failed"
        metrics = _read_metrics(run_dir / "results.csv")
        if metrics:
            metrics.setdefault("metrics_from", "partial results.csv")
        update_job(
            job_id,
            status=status,
            last_epoch=actual,
            finished_at=db.utcnow(),
            pid=None,
            resume_from=resume_path,
            best_pt=str(best) if best.is_file() else resume_path,
            metrics=metrics or None,
            eta_seconds=0,
        )
        append_log(
            log_path,
            f"训练未完成：目标 {epochs} 轮，实际完成 {actual} 轮"
            f"（退出码 {proc.returncode}，收尾标记={'有' if clean else '无'}）。"
            + ("可点「继续训练」从断点接着跑。" if can_resume else "找不到可用的 last.pt，无法续训。"),
        )
        if status == "failed":
            raise RuntimeError(
                f"训练失败：目标 {epochs} 轮，实际 {actual} 轮，退出码 {proc.returncode}，请查看日志"
            )
        return

    # 真正完成（跑满 或 早停正常收尾）
    metrics = _collect_success_metrics(
        job=job,
        run_dir=run_dir,
        best_pt=best,
        data_yaml=data_yaml,
        imgsz=imgsz,
        log_path=log_path,
    )
    update_job(
        job_id,
        status="success",
        best_pt=str(best) if best.is_file() else resume_path,
        metrics=metrics,
        last_epoch=actual,  # 真实完成轮数，不要写死成目标 epochs
        eta_seconds=0,
        finished_at=db.utcnow(),
        pid=None,
        resume_from=resume_path,
    )
    append_log(
        log_path,
        f"训练成功：目标 {epochs} 轮，实际完成 {actual} 轮 metrics={metrics}",
    )


def _count_epochs(csv_path: Path) -> int:
    """results.csv 每轮一行（不含表头），行数即已完成轮数。"""
    if not csv_path.is_file():
        return 0
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            return sum(1 for _ in csv.DictReader(f))
    except Exception:
        return 0


def _training_finished_cleanly(run_dir: Path) -> bool:
    """训练循环正常结束（含早停）时 ultralytics 会写 results.png。"""
    return (run_dir / "results.png").is_file()


def _row_to_metrics(row: dict) -> dict:
    def g(*keys):
        for k in keys:
            if k in row and row[k] not in ("", None):
                try:
                    return float(row[k])
                except ValueError:
                    return row[k]
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


def _fitness_of_row(row: dict) -> float:
    """ultralytics 默认 fitness ≈ 0.1*mAP50 + 0.9*mAP50-95。"""
    def f(*keys):
        for k in keys:
            if k in row and row[k] not in ("", None):
                try:
                    return float(row[k])
                except ValueError:
                    pass
        return 0.0

    m50 = f("metrics/mAP50(B)", "metrics/mAP50")
    m95 = f("metrics/mAP50-95(B)", "metrics/mAP50-95")
    return 0.1 * m50 + 0.9 * m95


def _read_metrics(csv_path: Path) -> dict:
    """从 results.csv 取 fitness 最高的那一行（不再用最后一行）。"""
    if not csv_path.is_file():
        return {}
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}
        best_row = max(rows, key=_fitness_of_row)
        return _row_to_metrics(best_row)
    except Exception:
        return {}


def _resolve_data_yaml(
    job: dict, run_dir: Path, data_yaml: Optional[str]
) -> Optional[Path]:
    """定位训练用的 data.yaml（续训时本地变量可能为空）。"""
    if data_yaml:
        p = Path(data_yaml)
        if p.is_file():
            return p
    params = job.get("params") or {}
    for key in ("data_yaml", "data"):
        v = params.get(key)
        if v and Path(v).is_file():
            return Path(v)
    # 导出约定：datasets/<id>/exports/<job_id>/data.yaml
    ds_id = job.get("dataset_id")
    if ds_id:
        cand = config.DATASETS_DIR / ds_id / "exports" / job["id"] / "data.yaml"
        if cand.is_file():
            return cand
    # args.yaml 里的 data 字段
    args_path = run_dir / "args.yaml"
    if args_path.is_file():
        try:
            for line in args_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("data:"):
                    val = line.split(":", 1)[1].strip().strip("'\"")
                    if val and Path(val).is_file():
                        return Path(val)
        except Exception:
            pass
    return None


def _parse_val_log_metrics(text: str) -> dict:
    """从 yolo detect val 日志解析 all 行的 P/R/mAP（备用）。"""
    # 典型：all  100  200  0.95  0.90  0.96  0.75
    pat = re.compile(
        r"^\s*all\s+\d+\s+\d+\s+"
        r"([0-9]*\.?[0-9]+)\s+"
        r"([0-9]*\.?[0-9]+)\s+"
        r"([0-9]*\.?[0-9]+)\s+"
        r"([0-9]*\.?[0-9]+)",
        re.MULTILINE,
    )
    matches = list(pat.finditer(text))
    if not matches:
        return {}
    m = matches[-1]
    return {
        "precision": float(m.group(1)),
        "recall": float(m.group(2)),
        "mAP50": float(m.group(3)),
        "mAP50-95": float(m.group(4)),
    }


def _validate_best_pt(
    *,
    best_pt: Path,
    data_yaml: Path,
    imgsz: int,
    run_dir: Path,
    log_path: Path,
) -> dict:
    """用 best.pt 跑一次 val，返回指标。

    优先走 ultralytics Python API（CLI 的 LOGGER 不进重定向文件，日志常为空）。
    失败再尝试 CLI + 日志解析。
    """
    append_log(
        log_path,
        f"开始用 best.pt 做验证: model={best_pt} data={data_yaml} imgsz={imgsz}",
    )
    # 1) Python API
    try:
        from ultralytics import YOLO

        model = YOLO(str(best_pt))
        res = model.val(
            data=str(data_yaml),
            imgsz=imgsz,
            device="cpu",
            verbose=False,
            project=str(run_dir.resolve()),
            name="val_best",
            exist_ok=True,
            plots=False,
        )
        box = getattr(res, "box", None)
        if box is None:
            raise RuntimeError("val 结果没有 box 指标")
        parsed = {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "mAP50": float(box.map50),
            "mAP50-95": float(box.map),
        }
        return parsed
    except Exception as exc:
        append_log(log_path, f"best.pt Python val 失败，尝试 CLI: {exc}")

    # 2) CLI 兜底
    val_dir = run_dir / "val_best"
    val_dir.mkdir(parents=True, exist_ok=True)
    val_log = val_dir / "val.log"
    cmd = [
        _resolve_yolo_bin(),
        "detect",
        "val",
        f"model={best_pt}",
        f"data={data_yaml}",
        f"imgsz={imgsz}",
        "device=cpu",
        "verbose=True",
        f"project={run_dir.resolve().as_posix()}",
        "name=val_best",
        "exist_ok=True",
    ]
    env = os.environ.copy()
    env["YOLO_OFFLINE"] = "1"
    env["OMP_NUM_THREADS"] = str(config.CPU_THREADS)
    env["MKL_NUM_THREADS"] = str(config.CPU_THREADS)
    with open(val_log, "w", encoding="utf-8") as lf:
        proc = subprocess.run(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            cwd=str(config.BASE_DIR),
            env=env,
            timeout=3600,
        )
    text = val_log.read_text(encoding="utf-8", errors="ignore") if val_log.is_file() else ""
    if proc.returncode != 0:
        raise RuntimeError(f"best.pt val 退出码 {proc.returncode}")
    parsed = _parse_val_log_metrics(text)
    if not parsed:
        raise RuntimeError("best.pt val 日志里解析不到指标")
    return parsed


def _collect_success_metrics(
    *,
    job: dict,
    run_dir: Path,
    best_pt: Path,
    data_yaml: Optional[str],
    imgsz: int,
    log_path: Path,
) -> dict:
    """训练成功后优先用 best.pt val 指标；失败则回退到 results.csv 最优行。"""
    yaml_path = _resolve_data_yaml(job, run_dir, data_yaml)
    if best_pt.is_file() and yaml_path is not None:
        try:
            val_m = _validate_best_pt(
                best_pt=best_pt,
                data_yaml=yaml_path,
                imgsz=imgsz,
                run_dir=run_dir,
                log_path=log_path,
            )
            # 附上 results.csv 最优轮次作对照
            csv_m = _read_metrics(run_dir / "results.csv")
            out = {
                "epoch": csv_m.get("epoch"),
                "precision": val_m.get("precision"),
                "recall": val_m.get("recall"),
                "mAP50": val_m.get("mAP50"),
                "mAP50-95": val_m.get("mAP50-95"),
                "box_loss": csv_m.get("box_loss"),
                "cls_loss": csv_m.get("cls_loss"),
                "metrics_from": "best.pt val",
            }
            append_log(log_path, f"best.pt val 指标: {out}")
            return out
        except Exception as exc:
            append_log(log_path, f"best.pt val 失败，回退 results.csv 最优行: {exc}")
            m = _read_metrics(run_dir / "results.csv")
            # 任务书要求标明来源；实际取 fitness 最高行作兜底
            m["metrics_from"] = "best fitness epoch (val failed)"
            return m

    m = _read_metrics(run_dir / "results.csv")
    if not best_pt.is_file():
        m["metrics_from"] = "best fitness epoch (no best.pt)"
    else:
        m["metrics_from"] = "best fitness epoch (no data.yaml)"
    return m


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


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _is_our_train_process(pid: int, job: dict) -> bool:
    """读 /proc/<pid>/cmdline，确认是本 job 的 ultralytics 训练进程。

    防止容器重启后 pid 被无关进程复用，误 killpg 整组进程。
    """
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except (OSError, PermissionError):
        return False
    cmdline = raw.replace(b"\0", b" ").decode("utf-8", "ignore")
    job_id = str(job.get("id") or "")
    run_dir = ""
    if job.get("run_dir"):
        try:
            run_dir = str(Path(job["run_dir"]).resolve())
        except Exception:
            run_dir = str(job["run_dir"])
    # 训练走 yolo CLI（或旧版 -m ultralytics）；cmdline 含 detect train + job_id
    has_yolo = (
        "ultralytics" in cmdline
        or "/yolo " in f" {cmdline} "
        or cmdline.strip().endswith("yolo")
        or " yolo " in f" {cmdline} "
        or ("detect" in cmdline and "train" in cmdline)
    )
    has_job = bool(job_id) and job_id in cmdline
    has_run = bool(run_dir) and run_dir in cmdline
    return has_yolo and (has_job or has_run)


def _kill_pid(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
    deadline = time.time() + 5
    while time.time() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.2)
    try:
        os.killpg(pid, signal.SIGKILL)
    except Exception:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass


def recover_interrupted_jobs() -> int:
    """启动时扫描 running 任务：一律标 interrupted；仅确认是本 job 训练进程才杀。"""
    with db._lock, db.connect() as conn:
        rows = conn.execute(
            "SELECT * FROM train_job WHERE status='running'"
        ).fetchall()
    n = 0
    for r in rows:
        job = dict(r)
        pid = r["pid"]
        if pid and _pid_alive(pid) and _is_our_train_process(int(pid), job):
            try:
                _kill_pid(int(pid))
            except Exception as exc:
                logger.warning("终止残留训练 pid=%s 失败: %s", pid, exc)
        elif pid and _pid_alive(pid):
            logger.warning(
                "pid=%s 存活但不是本 job 的训练进程（疑似 pid 复用），跳过终止 job=%s",
                pid,
                r["id"],
            )
        resume = r["resume_from"] or str(Path(r["run_dir"]) / "weights" / "last.pt")
        run_dir = Path(r["run_dir"])
        actual = _count_epochs(run_dir / "results.csv")
        update_job(
            r["id"],
            status="interrupted",
            last_epoch=actual if actual > 0 else r["last_epoch"],
            finished_at=db.utcnow(),
            pid=None,
            resume_from=resume if Path(resume).is_file() else r["resume_from"],
        )
        append_log(
            r["log_path"] or (run_dir / "train.log"),
            f"服务重启，任务中断（已完成约 {actual} 轮，请点「继续训练」从断点恢复）",
        )
        n += 1
    return n


def kill_running_train_processes() -> int:
    """关服务时杀掉所有 running 训练子进程，并标为 interrupted。

    关服时 pid 为本进程刚拉起的，仍用 cmdline 校验以免误伤。
    """
    with db._lock, db.connect() as conn:
        rows = conn.execute(
            "SELECT * FROM train_job WHERE status='running'"
        ).fetchall()
    n = 0
    for r in rows:
        job = dict(r)
        pid = r["pid"]
        if pid and _pid_alive(pid) and _is_our_train_process(int(pid), job):
            _kill_pid(int(pid))
        elif pid and _pid_alive(pid):
            logger.warning(
                "关服时 pid=%s 不是本 job 训练进程，跳过终止 job=%s",
                pid,
                r["id"],
            )
        resume = r["resume_from"] or str(Path(r["run_dir"]) / "weights" / "last.pt")
        run_dir = Path(r["run_dir"])
        actual = _count_epochs(run_dir / "results.csv")
        update_job(
            r["id"],
            status="interrupted",
            last_epoch=actual if actual > 0 else r["last_epoch"],
            finished_at=db.utcnow(),
            pid=None,
            resume_from=resume if Path(resume).is_file() else r["resume_from"],
        )
        append_log(
            r["log_path"] or (run_dir / "train.log"),
            f"服务关闭，训练进程已终止（已完成约 {actual} 轮）",
        )
        n += 1
    return n


def publish_job(job_id: str, name: Optional[str] = None, notes: str = "") -> dict:
    from app.services import dataset_svc, model_svc

    job = get_job(job_id)
    if job["status"] != "success" or not job.get("best_pt"):
        raise ValueError("只有训练成功且存在 best.pt 的任务可以发布")

    ds = dataset_svc.get_dataset(job["dataset_id"])
    # 先快速注册 .pt，OpenVINO 走后台任务，避免 HTTP 挂几分钟
    return model_svc.register_model(
        path=job["best_pt"],
        name=name or f"{ds['name']}-{job_id[-6:]}",
        classes=ds["classes"],
        metrics=job.get("metrics") or {},
        from_job=job_id,
        notes=notes,
        export_ov=False,
        schedule_openvino=True,
        imgsz=int((job.get("params") or {}).get("imgsz", 416)),
    )
