# YOLO Studio —— 图片标注 + 模型训练一体化 Web 平台（实现提示词）

> 本文件是一份**自包含的实现任务书**，交给编码 agent 后应能直接开工，不需要再追问需求。

---

## 一、背景

用户在**公安内网（完全离线，无互联网）**环境做"飙车炸街"图片识别，需要识别两类目标：
- `wheelie` —— 翘车头
- `multi rider` —— 多人骑车（一车多人）

现状与痛点：

1. 已用 ultralytics 训练出两个可用模型，放在 `~/project/bczj-classifier/models/`：
   - `0517_yolo26n_wheelie_multi-rider.pt`（微，5.4 MB）
   - `0517_yolo26s_wheelie_multi-rider.pt`（小，20 MB）
2. 训练数据的标签是用 **labelImg** 手工打的，而且必须把图片**带到外网**去标，流程繁琐、有数据合规风险、效率极低。
3. 用户希望有一个**内网自建的 Web 系统**：既能打标签，又能直接训练模型，形成闭环。

## 二、目标

交付一个名为 **`yolo-studio`** 的独立项目（新建目录 `~/project/yolo-studio/`），用 Python + FastAPI 实现，具备：

1. 数据集管理（图片/视频导入、类别定义、进度统计）
2. **好用的** Web 标注工作台（Canvas 画框、全键盘操作）
3. **多种自动标注手段**，最大程度减少手工画框
4. YOLO 模型训练（实时日志 + 指标曲线）
5. 模型仓库与发布，训练产物可一键设为"默认预标注模型"，形成 `标注 → 训练 → 更好的预标注 → 更省力的标注` 迭代闭环
6. 全程**离线可用**，Docker 方式交付内网，端口 **5016**
7. **在一台没有 GPU 的老服务器上跑得动** —— 这是本项目最强的约束，动手前必须先读完「第四章之二」

## 三、必须先读的现有代码（重要：不要从零造轮子）

`~/project/multi-rider/` 是一个 Flask 项目，其中 **`modules/training/` 已经实现了本系统约 60% 的后端业务逻辑**。请先完整阅读，然后**移植 + 重构**为 FastAPI 风格，而不是重写：

| 文件 | 行数 | 可复用的内容 |
|---|---|---|
| `multi-rider/modules/training/routes.py` | 856 | 34 个接口的入参/出参结构、序列化函数 |
| `multi-rider/modules/training/services/dataset_service.py` | 638 | 数据集目录结构、ZIP 导入、资产管理、标注读写 |
| `multi-rider/modules/training/services/auto_annotate_service.py` | 294 | 模型预标注、类别名归一化与映射逻辑 |
| `multi-rider/modules/training/services/auto_annotate_task_service.py` | 150 | 预标注异步任务 |
| `multi-rider/modules/training/services/train_task_service.py` | 849 | 训练任务、数据集划分、data.yaml 生成、subprocess 调 yolo、产物收集 |
| `multi-rider/modules/training/services/model_registry_service.py` | 401 | 模型注册、版本槽位、回滚 |
| `multi-rider/shared/inference/infer_service.py` | — | YOLO 模型缓存加载、批量推理、torch 线程数控制 |

另外参考 `~/project/bczj-classifier/`（FastAPI + Docker 离线交付的成熟范例）：
- `app/main.py`、`app/api/*`、`app/tasks.py` —— FastAPI 项目组织方式
- `Dockerfile`、`docker-compose.yml`、`run.sh` —— 离线镜像交付流程
- `static/vendor/` —— 前端依赖本地化的做法

**移植时要改进的地方**（原 Flask 版的不足）：
- 原版没有真正的 Canvas 标注器前端，本项目必须补上
- 原版只有"已有模型预标注"一种自动标注方式，本项目要扩展到 5 种
- 原版 `_resolve_yolo_executable()` 只找 Windows 的 `yolo.exe`，必须改成跨平台（Linux 优先）
- 原版标注同时存 DB 和 txt 容易不一致，本项目改为 **DB 唯一真源**

## 四、技术栈（硬性约束）

| 层 | 选型 | 说明 |
|---|---|---|
| 后端 | Python 3.12 + FastAPI + Uvicorn | 与现有项目一致 |
| 数据库 | SQLite（开启 WAL） | 单文件、免运维、内网友好 |
| 训练 | ultralytics >= 8.3 + **PyTorch CPU 版** | 见第四章之二，纯 CPU，必须装 CPU-only torch |
| 推理 | ultralytics + **OpenVINO Runtime** | Intel CPU 上比 PyTorch 快 2~3 倍，预标注必须走这条 |
| 前端 | Vue 3（global build）+ 原生 Canvas | **禁止任何 CDN**，全部 vendor 本地化 |
| 图表 | ECharts（本地 vendor） | 训练曲线 |
| 异步任务 | 进程内 ThreadPoolExecutor + 任务表 | **禁止引入 Celery / Redis / RabbitMQ** |
| 部署 | Docker + docker-compose，端口 **5016** | 用户指定 |

**离线硬约束（必须逐条落实，否则内网直接跑不起来）**：
1. 设置环境变量 `YOLO_OFFLINE=1`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`
2. 写入 ultralytics 配置关闭在线检查：`settings.update({'sync': False})`，并预置 `Arial.ttf` 到 `~/.config/Ultralytics/`
3. 所有权重预置在 `weights/` 目录，代码中**绝不允许**出现按名字自动下载模型的写法（如 `YOLO('yolov8n.pt')` 这种会触发下载的调用），一律用绝对路径加载
4. 前端所有 js/css/字体放 `web/vendor/`，HTML 中不得出现任何 `http(s)://` 外链
5. `requirements.txt` 之外提供 `wheels/` 离线安装目录说明

## 四之二、★ 目标机器是纯 CPU，这一章是本项目最关键的约束

### 实际部署环境

| 项 | 值 |
|---|---|
| CPU | Intel Xeon **E5-2697 v3**，14 核 28 线程，Haswell 架构（2014），支持 AVX2、**不支持 AVX-512** |
| 内存 | 32 GB |
| GPU | **无，一块都没有** |
| 系统 | CentOS Stream 10 |
| 部署方式 | Docker |
| 端口 | **5016** |
| 网络 | 完全离线 |

### 必须做的事（漏一条就会出大问题）

**1. 装 CPU-only 版 PyTorch，绝对不要装默认的 CUDA 版**

默认 `pip install torch` 会拉 2.5 GB 的 CUDA 版本，在这台机器上一点用没有，还会把 Docker 镜像撑到 6 GB+，拷进内网非常痛苦。必须：

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.5.1+cpu
torchvision==0.20.1+cpu
```
镜像能从 6 GB 降到 1.5 GB 左右。构建后用 `python -c "import torch; print(torch.__version__)"` 确认输出带 `+cpu`。

**2. 线程数要调对，用物理核数而不是逻辑核数**

超线程对 YOLO 训练往往是负优化。统一在 `config.py` 里设置，且允许 `.env` 覆盖：

```python
CPU_THREADS = int(os.getenv("CPU_THREADS", "14"))   # 物理核数
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
torch.set_num_threads(CPU_THREADS)
```
注意：这几个环境变量**必须在 import torch 之前设置**才生效，所以要放在 `config.py` 顶部，且 `main.py` 第一行就 import config。

**3. 所有涉及模型的调用强制 `device='cpu'`、`amp=False`**

CPU 上没有混合精度，`amp=True` 会触发一次没必要的 GPU 检查甚至联网下载检测脚本（离线环境直接卡住）。训练、验证、推理全部显式传 `device='cpu', amp=False`。

**4. 预标注/推理走 OpenVINO，不要用 PyTorch 直接推**

模型注册进模型仓库时，后台自动导出一份 OpenVINO 格式：
```python
model.export(format='openvino', imgsz=<训练时的 imgsz>, half=False)
```
产物存到 `data/models/<model_id>_openvino_model/`。预标注时优先加载 OpenVINO 版本，找不到才退回 `.pt`。这一步能让预标注速度快 2~3 倍，是纯 CPU 环境下体感差别最大的优化。

导出失败要能优雅降级（记日志 + 继续用 .pt），不能因为导出失败就让模型不可用。

**5. 内存缓存策略要按数据量自动决定**

32 GB 内存，训练时把图片缓存进内存能省掉反复读盘解码，快 20~30%。但数据量大了会 OOM。规则：

```
预估内存 = 图片数 × imgsz × imgsz × 3 bytes × 1.2
预估 < 8 GB  → cache='ram'
8~16 GB      → cache='disk'
> 16 GB      → cache=False
```
把这个判断写成 `train_svc.py` 里的函数，并把选择结果打进训练日志，让用户知道系统做了什么决定。

**6. DataLoader workers 与 Docker 共享内存**

`workers=8`（不要设成 14，数据加载抢了计算的核反而慢）。同时 docker-compose 必须设 `shm_size: '2gb'`，否则多 worker 的 DataLoader 会报 `bus error` / `DataLoader worker killed`——这是 Docker 里跑 PyTorch 最常见的坑。

### 训练默认参数（针对这台机器调过，不要照抄 GPU 时代的默认值）

| 参数 | 默认值 | 理由 |
|---|---|---|
| `base_model` | **yolo26n**（微） | s 版在这台机器上慢 3 倍，不值 |
| `imgsz` | **416** | 640 的计算量是 416 的 2.4 倍；飙车炸街目标在画面里不算小，416 够用 |
| `epochs` | **40** | 配合早停，基本 25~35 轮就收敛 |
| `batch` | **16** | 内存够，再大对 CPU 提速有限 |
| `patience` | **10** | 早停，别浪费几小时跑无效轮次 |
| `workers` | **8** | 留核给计算 |
| `device` | `cpu` | 固定 |
| `amp` | `False` | 固定 |
| `freeze` | **10**（可关） | 冻结主干只训头部，省 40~50% 时间 |
| `cache` | 自动 | 按上面的规则算 |

### ★ 从已有模型微调（本项目最重要的省时手段）

用户手上已经有 `0517_yolo26s_wheelie_multi-rider.pt`，它已经认识"翘车头"和"多人骑车"了。新数据不该从 COCO 预训练权重从头开始学，而应该**在这个模型基础上接着训**。

界面上"底模"选择要分成两组，并且**默认选中第二组**：

- **从零开始**：`yolo26n.pt` / `yolo26s.pt`（COCO 预训练）→ 需要 40+ epoch
- **在已有模型上继续训练（推荐）**：列出模型仓库里所有已注册模型 → 只需 **15~25 epoch**

选择第二组时，前端要给出提示：
> "已选择在现有模型基础上微调，训练轮次会自动降到 20 轮，预计耗时比从零训练少 60%。"

同时后端要校验：底模的类别列表必须和数据集类别兼容（数量与顺序一致），不一致要报人话错误，例如
> "这个模型认识的是 [翘车头, 多人骑车]，但你的数据集有 3 个类别，类别对不上，不能在它基础上继续训练。请改用从零开始的底模。"

### ★ 训练耗时预估器（必做，否则用户会以为系统卡死）

CPU 训练动辄几小时，界面上**必须**让用户提前知道要等多久。

**两段式预估**：

1. **提交前粗估**：参数表单里任何一项改变，前端实时调 `POST /api/train/estimate`，后端按经验公式返回预计耗时。公式基线（E5-2697 v3，14 线程，1000 张图，yolo26n，imgsz 640，不冻结 = **10 分钟/epoch**），再按下列系数缩放：
   - 图片数：线性
   - imgsz：`(imgsz/640)²`
   - 模型：n=1.0，s=3.0，m=7.0
   - freeze=10：×0.55
   - cache='ram'：×0.78

2. **第一轮实测后校准**：第 1 个 epoch 跑完，用实测秒数 × 剩余轮数替换掉估算值，前端显示"预计还需 3 小时 12 分（已根据实测校准）"。

**红线警告**：预估超过 **12 小时**，提交按钮旁边显示红色警告并给出具体的降参建议：
> "⚠️ 预计需要 31 小时。建议：把底模从 yolo26s 换成 yolo26n（省 20 小时），或把图片尺寸从 640 降到 416（再省 6 小时）。"

超过 24 小时则默认禁用提交按钮，需要用户勾选"我知道会很久，仍然继续"才放行。

### ★ 断点续训与容器重启恢复（必做）

一次训练要跑几小时甚至跨夜，容器重启、误操作、断电都可能中断。

1. `train_job` 表增加字段：`last_epoch`（已完成轮次）、`resume_from`（last.pt 路径）、`eta_seconds`（最新预估剩余秒数）
2. 训练进程用 subprocess 启动，PID 记进任务表
3. **应用启动时**扫描所有 `status='running'` 的训练任务：进程已不在 → 状态改为 `interrupted`，并在界面上给一个醒目的「继续训练」按钮
4. 「继续训练」调用 ultralytics 的 resume：
   ```python
   YOLO(f"{run_dir}/weights/last.pt").train(resume=True)
   ```
5. docker-compose 设 `restart: unless-stopped`，但**不要**让容器自动重跑训练——必须由用户点击「继续训练」，避免死循环重启把机器跑垮

### 部署要点（CentOS Stream 10 + Docker）

1. **SELinux**：CentOS 默认 enforcing，volume 挂载必须加 `:z`，否则容器读不到 `data/` 和 `weights/`：
   ```yaml
   volumes:
     - ./data:/app/data:z
     - ./weights:/app/weights:z
   ```
2. **防火墙**：README 里写明放行端口：
   ```bash
   firewall-cmd --permanent --add-port=5016/tcp && firewall-cmd --reload
   ```
3. **docker-compose 关键配置**：
   ```yaml
   services:
     yolo-studio:
       ports: ["5016:5016"]
       shm_size: '2gb'          # 必须，否则 DataLoader 崩
       environment:
         - CPU_THREADS=14
         - OMP_NUM_THREADS=14
         - YOLO_OFFLINE=1
       restart: unless-stopped
   ```
4. **不要限制 cpus**：训练就指望这 14 个核，别用 `cpus: "4"` 之类的限制把自己捆住
5. **基础镜像**用 `python:3.12-slim`（Debian 系），不必跟宿主机 CentOS 保持一致；需要 `libgl1`、`libglib2.0-0` 给 OpenCV
6. 交付流程照搬 `bczj-classifier/README.md`：有网机器 build → `docker save` → 拷贝 → 内网 `docker load` → `docker compose up -d`

### 其他模型在 CPU 上的可用性（实现前先掂量）

| 能力 | CPU 上的表现 | 结论 |
|---|---|---|
| yolo26n 预标注（OpenVINO） | 约 30~60 ms/张 | ✅ 放心用 |
| yolo26s 预标注（OpenVINO） | 约 100~180 ms/张 | ✅ 可用 |
| YOLO-World 开放词表 | 约 0.6~1.2 s/张 | ⚠️ 只能小批量试，UI 上默认限制单次 ≤ 200 张，并明确提示耗时 |
| **MobileSAM** 点选 | 编码约 1.5~3 s/张，之后每次点选 < 100 ms | ✅ 可用，但**必须做 image embedding 的 LRU 缓存**（缓存 8 张），并在切图时后台预编码下一张 |
| 原版 SAM / SAM2-large | 单张 10 s 以上 | ❌ **禁止使用**，只用 MobileSAM |
| 视频跟踪传播 | 约 3~8 帧/秒 | ⚠️ 单次任务最多 300 帧，超出要分批，并显示进度 |

## 五、目录结构

```
yolo-studio/
├── app/
│   ├── main.py                 # FastAPI 入口，挂载静态目录与路由
│   ├── config.py               # 环境变量、路径常量、日志
│   ├── db.py                   # SQLite 连接、建表 DDL、迁移
│   ├── tasks.py                # 通用后台任务队列（线程池 + 任务表）
│   ├── api/
│   │   ├── datasets.py         # 数据集 CRUD / 导入 / 导出 / 统计
│   │   ├── images.py           # 图片列表、原图、缩略图、下一张待标
│   │   ├── annotations.py      # 标注读写、确认状态
│   │   ├── autolabel.py        # 预标注（模型 / 开放词表 / SAM / 跟踪）
│   │   ├── train.py            # 训练任务、日志 SSE、指标
│   │   ├── models.py           # 模型仓库、发布、评估
│   │   └── system.py           # 健康检查、设备信息（GPU/CPU）
│   ├── services/
│   │   ├── dataset_svc.py
│   │   ├── annotation_svc.py
│   │   ├── autolabel_svc.py    # 已有模型预标注
│   │   ├── openvocab_svc.py    # YOLO-World 开放词表零样本
│   │   ├── sam_svc.py          # SAM 点选辅助
│   │   ├── track_svc.py        # 视频抽帧 + 跟踪传播
│   │   ├── dedup_svc.py        # pHash 近重复检测
│   │   ├── active_svc.py       # 主动学习排序
│   │   ├── export_svc.py       # 划分 train/val + data.yaml + txt 生成
│   │   └── train_svc.py
│   ├── infer/
│   │   └── engine.py           # 模型缓存加载（带锁）、批量推理
│   └── schemas.py              # Pydantic 模型
├── web/
│   ├── index.html
│   ├── css/app.css
│   ├── js/
│   │   ├── app.js              # 路由与壳层
│   │   ├── datasets.js
│   │   ├── labeler.js          # ★ Canvas 标注器（本项目最核心的前端文件）
│   │   ├── train.js
│   │   └── models.js
│   └── vendor/                 # vue.global.prod.js、echarts.min.js 等
├── data/                       # 运行时数据（docker volume 挂载）
│   ├── app.db
│   ├── datasets/<ds_id>/{images,thumbs,exports}
│   ├── runs/<job_id>/          # 训练输出
│   ├── models/                 # 已注册模型副本（.pt）
│   └── models/<id>_openvino_model/   # 自动导出的 OpenVINO 版本，预标注优先用它
├── weights/                    # 预置权重（不进 git）
│   ├── yolo26n.pt              # ★ 默认训练底模（CPU 环境首选）
│   ├── yolo26s.pt              # 备选，CPU 上慢 3 倍，非必要不用
│   ├── 0517_yolo26s_wheelie_multi-rider.pt   # 从 bczj-classifier 拷入，冷启动预标注 + 微调底模
│   ├── 0517_yolo26n_wheelie_multi-rider.pt   # 同上，微版
│   ├── yolov8s-worldv2.pt      # 从 multi-rider 拷入，开放词表零样本
│   └── mobile_sam.pt           # SAM 点选辅助（★ 只用 MobileSAM，原版 SAM 在 CPU 上不可用）
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── run.sh
└── README.md
```

## 六、数据模型（SQLite）

```sql
-- 数据集
CREATE TABLE dataset (
  id TEXT PRIMARY KEY,              -- ds_20260725_143000_a1b2c3
  name TEXT NOT NULL,
  classes TEXT NOT NULL,            -- JSON 数组，如 ["wheelie","multi rider"]
  notes TEXT DEFAULT '',
  created_at TEXT, updated_at TEXT
);

-- 图片
CREATE TABLE image (
  id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  filename TEXT NOT NULL,
  rel_path TEXT NOT NULL,           -- 相对 datasets/<ds_id>/images/
  width INTEGER, height INTEGER,
  sha1 TEXT,                        -- 完全重复去重
  phash TEXT,                       -- 近重复聚类
  group_key TEXT,                   -- 近重复组 / 视频来源片段，划分数据集时防泄漏
  source TEXT,                      -- upload / zip / video / db
  split TEXT DEFAULT '',            -- train / val / test，导出时写入
  review_status TEXT DEFAULT 'unlabeled',
      -- unlabeled 未标 / auto 已预标待审 / reviewed 已审 / confirmed 已确认 / skipped 跳过
  box_count INTEGER DEFAULT 0,
  uncertainty REAL DEFAULT 0,       -- 主动学习排序分，越大越该优先标
  created_at TEXT
);
CREATE INDEX idx_image_ds_status ON image(dataset_id, review_status);

-- 标注框（归一化 YOLO 格式，一行一个框）
CREATE TABLE annotation (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  image_id TEXT NOT NULL,
  class_idx INTEGER NOT NULL,
  cx REAL, cy REAL, w REAL, h REAL, -- 均为 0~1 归一化值
  conf REAL DEFAULT 1.0,
  source TEXT DEFAULT 'manual',     -- manual / auto / openvocab / sam / track
  created_at TEXT
);
CREATE INDEX idx_anno_image ON annotation(image_id);

-- 通用后台任务
CREATE TABLE task (
  id TEXT PRIMARY KEY,
  type TEXT,                        -- import / autolabel / track / train / export / dedup
  dataset_id TEXT,
  status TEXT,                      -- pending / running / success / failed / canceled
  params TEXT, progress REAL DEFAULT 0,
  message TEXT, error TEXT,
  created_at TEXT, started_at TEXT, finished_at TEXT
);

-- 训练任务
CREATE TABLE train_job (
  id TEXT PRIMARY KEY,
  dataset_id TEXT, base_model TEXT,
  params TEXT,                      -- JSON: epochs/imgsz/batch/device/patience/augment
  run_dir TEXT, log_path TEXT,
  status TEXT, metrics TEXT,        -- JSON: mAP50 / mAP50-95 / precision / recall
  best_pt TEXT,
  created_at TEXT, finished_at TEXT
);

-- 模型仓库
CREATE TABLE model (
  id TEXT PRIMARY KEY,
  name TEXT, path TEXT,
  classes TEXT, metrics TEXT,
  from_job TEXT, notes TEXT,
  is_default_autolabel INTEGER DEFAULT 0,   -- 是否为默认预标注模型（闭环关键）
  created_at TEXT
);
```

**关键规则**：标注数据以 **数据库为唯一真源**。YOLO 的 `labels/*.txt` 只在**导出或训练开始时**由 `export_svc.py` 一次性生成到 `exports/<timestamp>/`，绝不双向同步。

## 七、功能需求（按里程碑交付）

### M1 —— 数据集 + 标注工作台（最优先，做完就能替代 labelImg）

**后端接口**：
```
POST   /api/datasets                     建数据集（name, classes[]）
GET    /api/datasets                     列表（含标注进度统计）
GET    /api/datasets/{id}                详情
PATCH  /api/datasets/{id}                改名 / 增删类别（删类别需二次确认并级联处理已有框）
DELETE /api/datasets/{id}                删除
POST   /api/datasets/{id}/import/files   多图上传
POST   /api/datasets/{id}/import/zip     ZIP 上传（支持内含 images/ + labels/ 的 YOLO 格式，一并导入已有标注）
GET    /api/datasets/{id}/images         分页列表，支持 status/split/关键字/排序(uncertainty) 过滤
GET    /api/images/{img_id}/file         原图
GET    /api/images/{img_id}/thumb        缩略图（首次访问生成并缓存到 thumbs/）
GET    /api/images/{img_id}/annotations  取标注
PUT    /api/images/{img_id}/annotations  整图覆盖式保存（body: boxes[] + review_status）
POST   /api/images/{img_id}/confirm      标记确认
GET    /api/datasets/{id}/next-unlabeled?after={img_id}   取下一张待标（按 uncertainty 降序）
```

**前端标注器 `web/js/labeler.js` 要求（本项目成败关键，务必做扎实）**：

布局：左侧图片缩略列表（带状态色标）| 中间 Canvas 画布 | 右侧框列表 + 类别面板 + 进度。

交互：
- 图片自适应画布，**滚轮缩放**（以鼠标位置为锚点），**空格 + 拖拽**平移
- 按 `W` 进入画框模式，鼠标拖拽画矩形；松开即生成框并自动选中
- 点击框选中；拖拽 8 个控制点改大小；拖拽框体移动；`Delete` 删除
- 数字键 `1`~`9` 快速切换当前类别（也用于修改选中框的类别）
- `A` / `D` 上一张 / 下一张；`Enter` 确认本图并自动跳下一张
- `Ctrl+Z` 撤销 / `Ctrl+Shift+Z` 重做（至少 20 步栈）
- `C` **复制上一张的全部标注**到当前图（连拍/相似图场景极其省时）
- **切换图片时自动保存**，无需手动点保存
- 预标注结果用虚线框 + 置信度数字显示，人工确认后转为实线
- 顶部置信度滑块，实时过滤低置信度的预标框
- 显示当前进度：`已标 128 / 共 500`

**验收标准（M1）**：一名不看文档的民警，10 分钟内能独立完成 50 张图的标注，且全程不用鼠标点"保存"。

### M2 —— 自动标注（核心价值）

按优先级实现以下 5 种，**全部走后台任务，前端轮询 `/api/tasks/{id}` 显示进度**：

**① 已有模型预标注**（最先做）
```
POST /api/datasets/{id}/autolabel
body: { model: "weights/0517_yolo26s_wheelie_multi-rider.pt",
        conf: 0.25, iou: 0.5, imgsz: 640,
        class_map: {"wheelie": 0, "multi rider": 1},
        scope: "unlabeled" | "all",
        overwrite: false }
```
- 类别名归一化与映射逻辑直接移植 `auto_annotate_service.py` 的 `_normalize_token` / `_parse_class_mapping`
- 批量推理（batch=8），写入 annotation 表，`source='auto'`，图片状态置 `auto`
- 同时计算 `uncertainty`（建议公式：`1 - max(conf)`，无框时给 0.9，供主动学习排序）

**② 开放词表零样本**（新类别冷启动）
```
POST /api/datasets/{id}/autolabel/openvocab
body: { prompts: ["motorcycle with front wheel lifted", "three people on one motorcycle"],
        class_map: {0: 0, 1: 1}, conf: 0.1 }
```
- 用 `weights/yolov8s-worldv2.pt`，调 `model.set_classes(prompts)` 后推理
- UI 上提供提示词输入框 + "先试 20 张看效果"的预览按钮（避免跑完整个数据集才发现提示词不行）

**③ SAM 点选辅助**（交互式，实时接口不走后台任务）
```
POST /api/images/{img_id}/sam
body: { points: [[x, y]], labels: [1] }   -- 像素坐标，label 1=前景 0=背景
resp: { box: [cx, cy, w, h], score: 0.93 }
```
- 用 `weights/mobile_sam.pt`，取掩码外接矩形返回
- 前端：按 `E` 进入点选模式，在目标上点一下就出框，再按数字键给类别
- 图片 embedding 做 LRU 缓存（同一张图连续点选不重复编码）

**④ 视频跟踪传播**（有视频源时收益最大）
```
POST /api/datasets/{id}/import/video   上传 mp4/avi/mov，按 fps 抽帧入库，同一视频写同一 group_key
POST /api/datasets/{id}/track          从已标注的关键帧出发，用 ByteTrack 传播到后续帧
body: { start_image_id: "...", max_frames: 300 }
```
- 用 ultralytics 的 `model.track(persist=True, tracker='bytetrack.yaml')`
- 传播出的框 `source='track'`，状态为 `auto`，仍需人工抽查

**⑤ 主动学习 + 近重复去重**
```
POST /api/datasets/{id}/dedup      计算 pHash，汉明距离 <= 5 归为一组，写 group_key，
                                   组内只保留 1 张为 review_status='unlabeled'，其余标 'skipped'
GET  /api/datasets/{id}/images?sort=uncertainty   按不确定度降序返回，让人先标最有价值的图
```

### M3 —— 训练

```
POST /api/train/estimate         ★ 试算耗时（参数一变前端就调，见第四章之二）
POST /api/train                  创建训练任务
body: { dataset_id,
        base_model: "weights/yolo26n.pt",   -- 或模型仓库里已有模型的 id（微调模式）
        epochs: 40, imgsz: 416, batch: 16,
        val_ratio: 0.2, patience: 10, only_confirmed: true,
        freeze: 10, workers: 8,
        augment_preset: "default" | "strong" }
        -- device 固定 cpu、amp 固定 False、cache 自动判定，均不由前端传入
GET  /api/train/{job_id}         状态 + 指标 + last_epoch + eta_seconds
GET  /api/train/{job_id}/logs    ★ SSE 实时日志流
GET  /api/train/{job_id}/metrics 解析 results.csv 返回曲线数据（box_loss/cls_loss/mAP50/mAP50-95）
POST /api/train/{job_id}/cancel  终止（kill 子进程，任务置 canceled）
POST /api/train/{job_id}/resume  ★ 断点续训（从 last.pt 继续）
POST /api/train/{job_id}/publish 产物注册进模型仓库（同时后台导出 OpenVINO 版本）
GET  /api/train/{job_id}/artifacts/{filename}   下载 best.pt / 混淆矩阵 / PR 曲线图
```

实现要点：
- 训练前由 `export_svc.py` 生成标准 YOLO 目录（`images/train`、`images/val`、`labels/train`、`labels/val`）+ `data.yaml`
- **划分必须按 `group_key` 分组划分**，保证近重复图/同一视频的帧不会同时落在 train 和 val（否则 mAP 虚高，是这类项目最常见的坑）
- `only_confirmed=true` 时只用 `review_status='confirmed'` 的图
- 训练开始前校验：类别数 > 0、每个类别至少 10 个框、train/val 都非空，不满足直接报错并给出人话提示；微调模式还要校验底模类别与数据集类别兼容
- subprocess 调用（**Linux 优先，绝对不要硬编码 `yolo.exe`**，原 multi-rider 的 `_resolve_yolo_executable()` 是 Windows 专用的，必须重写）：
  ```python
  [sys.executable, "-m", "ultralytics", "detect", "train",
   f"data={yaml}", "device=cpu", "amp=False", ...]
  ```
  stdout/stderr 重定向到 `runs/<job_id>/train.log`，PID 记进任务表
- SSE 接口 tail 该日志文件推给前端
- **每个 epoch 结束时**从日志解析出已完成轮次，更新 `last_epoch` 与 `eta_seconds`（第 1 轮之后用实测速度校准预估）
- 训练结束自动跑一次 val（`device='cpu'`），把 mAP50 / mAP50-95 / precision / recall 存进 `train_job.metrics`
- 前端训练页：参数表单（带实时耗时预估）+ **醒目的进度条与剩余时间** + 实时日志黑框 + ECharts 双轴曲线（loss 下降 / mAP 上升）
- 因为一次训练可能跨夜，训练页要能关掉浏览器再回来看；任务全部状态持久化在数据库，不依赖前端连接

### M4 —— 模型仓库与闭环

```
GET  /api/models                          列表（含指标，支持多版本对比）
POST /api/models/{id}/set-default-autolabel   ★ 设为默认预标注模型
POST /api/models/{id}/evaluate            在指定数据集上跑评估
GET  /api/models/{id}/download            下载 .pt
DELETE /api/models/{id}
```

**闭环要求**：设为默认预标注模型后，M2 的"① 已有模型预标注"默认就用这个模型。前端在数据集页显著位置提示：
> "本数据集已标注 320 张，建议现在训练一版新模型，之后的预标注会更准。"

## 八、非功能要求

1. **中文界面**，所有错误提示说人话（例如不要抛 `KeyError: class_idx`，要说"这张图有个框的类别已被删除，请重新选择类别"）
2. 单机并发按 3~5 人使用设计；SQLite 开启 WAL，写操作加锁
3. 图片上传单次上限 500 MB，ZIP 内单文件校验，防目录穿越（`..` 路径）
4. 缩略图统一 320px 宽，懒生成并缓存
5. 训练任务同一时刻**只允许一个在跑**（14 核全给它，跑两个只会互相拖慢），排队执行；训练进行中时，预标注/跟踪等重 CPU 任务也要排队等待，并在界面上提示"正在训练模型，其他任务已排队"
6. 所有接口有明确的 Pydantic 请求/响应模型，自动生成的 `/docs` 可用
7. 日志分级输出到 `data/logs/app.log`，按天切割
8. **纯 CPU 环境下，任何超过 10 秒的操作都必须有进度反馈**，不允许出现前端转圈没有任何信息的情况

## 九、交付物

1. 完整可运行的 `~/project/yolo-studio/` 项目
2. `README.md`：功能说明、本地开发、Docker 离线交付步骤（参照 `bczj-classifier/README.md` 的组织方式），**必须包含 CentOS Stream 10 上的 SELinux 与防火墙说明**
3. `Dockerfile` + `docker-compose.yml`，端口 **5016**，`shm_size: 2gb`，`data/` 与 `weights/` 用 volume 挂载并带 `:z`
4. **`docs/20260725_离线权重清单.md`**：列出所有需要在外网提前下载的权重文件名、下载地址、文件大小、放置路径，方便用户一次性备齐带进内网
5. **`docs/20260725_CPU训练调优说明.md`**：给用户看的大白话文档，说明为什么默认用 yolo26n + 416、什么时候该微调什么时候该从零训、一次训练大概要等多久、内存不够怎么办
6. 一个冒烟测试脚本 `scripts/smoke_test.py`：建数据集 → 导 5 张图 → 预标注 → 存标注 → 导出 → 训 2 epoch（imgsz=320 保证几分钟内跑完）→ 校验产物存在 → 导出 OpenVINO 成功

## 十、验收标准

| 项 | 标准 |
|---|---|
| 离线 | `docker run --network none` 启动后，全部功能（含标注、预标注、训练）正常，无任何联网请求 |
| **CPU-only** | 容器内 `python -c "import torch;print(torch.__version__)"` 输出必须带 `+cpu`；镜像体积 < 2 GB |
| **线程** | 训练时 `top` 能看到进程占用 ~1400% CPU（14 核吃满），不是只用 1 个核 |
| **预估** | 训练提交前显示的预计耗时，与实际耗时误差在 ±30% 以内 |
| **续训** | 训练跑到一半 `docker compose restart`，重启后任务显示"已中断"，点「继续训练」能从断点接着跑完 |
| **OpenVINO** | 同一批 200 张图，OpenVINO 预标注耗时明显低于 PyTorch 直推（至少快 1.5 倍），且两者检测结果框数一致（允许 ±5%） |
| 标注效率 | 500 张图，先跑预标注，人工审核确认全部完成 ≤ 60 分钟 |
| 训练 | 能在界面上完成一次完整训练，实时看到日志与曲线，产物 best.pt 可下载并注册 |
| 闭环 | 新训练的模型能设为默认预标注模型，下一批预标注确实调用了它 |
| 兼容 | 能导入 labelImg 产出的 YOLO 格式 ZIP（images + labels + classes.txt），也能导出成同样格式 |
| 前端 | `web/index.html` 及所有 js 中不含任何 `http://` / `https://` 外链 |

## 十一、约定

- 交流与代码注释一律用**中文**
- **绝不在 `~/project` 顶层执行 `git init`**；如需版本管理，只在 `~/project/yolo-studio/` 内部 init
- 新建的非代码文件（md / sql / 文档）文件名加 `YYYYMMDD_` 前缀
- 遇到需求不明确的地方，先按本文件的默认值实现，并在 README 的"待确认事项"里列出来
