# YOLO Studio

内网离线可用的 **图片标注 + YOLO 模型训练** 一体化 Web 平台。

- 后端：Python 3.12 + FastAPI + ultralytics（**纯 CPU**）
- 前端：Vue 3 + 原生 Canvas 标注器 + ECharts（**本地 vendor，无 CDN**）
- 存储：SQLite（WAL）
- 部署：Docker，端口 **5016**
- 目标机器：Intel Xeon E5-2697 v3（14 核）/ 32 GB / 无 GPU / CentOS Stream 10 内网
- 镜像体积：多阶段去 CLIP 双份后实测 **3.77 GB**（此前双份约 4.45 GB；CLIP 只在 `/root/.cache/clip/`）。硬成本含 torch、openvino、polars、CLIP 一份与检测权重。`docker save` 拷内网请按 **≥ 4 GB** 准备

完整实现说明见 [docs/20260725_yolo_studio_设计与实现提示词.md](docs/20260725_yolo_studio_设计与实现提示词.md)。

## 功能概览

1. 数据集管理（多图 / ZIP / 视频抽帧导入）
2. **智能切图导入**：卡口大图用 person 检测切人+车身区域入库（默认 `bottom_ratio=1.30`，须与 bczj-classifier 一致；原图不入库，同图切图共享 group_key）
3. Canvas 标注工作台（画框、键盘快捷键、撤销重做、自动保存）
4. 自动标注：已有模型预标注、开放词表（YOLO-World）、MobileSAM 点选、视频跟踪、近重复去重
5. CPU 训练：耗时预估、实时日志 SSE、指标曲线、断点续训
6. 模型仓库：发布、默认预标注、OpenVINO 导出加速推理

## 目录要点

```
yolo-studio/
├── app/           # FastAPI 后端
├── web/           # 前端（vendor 需本地化）
├── weights/       # 预置权重（不进 git）
├── data/          # 运行时数据（volume）
├── docs/          # 设计与离线说明
└── scripts/       # vendor 下载、冒烟测试
```

## 准备权重与前端依赖

```bash
# 1) 权重（详见 docs/20260725_离线权重清单.md）
mkdir -p weights
cp ../bczj-classifier/models/0517_yolo26*.pt weights/

# 2) 前端 vendor（有网执行一次）
bash scripts/fetch_vendor.sh
```

## 本地开发

```bash
cp .env.example .env
bash run.sh
# 浏览器 http://<IP>:5016
```

## Docker 离线交付

**有网机器 build → save → 拷进内网 → load → compose up**

### 1. 有网机构建

```bash
# 确保 weights/*.pt 与 web/vendor/*.js 已就绪
docker build -t yolo-studio:latest .
docker save -o yolo-studio.tar yolo-studio:latest
```

### 2. 内网导入并启动

```bash
docker load -i yolo-studio.tar
cp .env.example .env
mkdir -p data weights
# SELinux（CentOS 默认 enforcing）：compose 已使用 :z
docker compose up -d
```

访问：`http://<服务器IP>:5016`

### 3. 防火墙（CentOS）

```bash
firewall-cmd --permanent --add-port=5016/tcp && firewall-cmd --reload
```

### 4. 断网自检

```bash
docker run --rm --network none -p 5016:5016 \
  -v "$(pwd)/data:/app/data:z" \
  -v "$(pwd)/weights:/app/weights:z" \
  --shm-size=2g \
  yolo-studio:latest
```

### 5. 确认 CPU 版 PyTorch

```bash
docker exec -it $(docker ps -qf ancestor=yolo-studio) \
  python -c "import torch; print(torch.__version__)"
# 输出应带 +cpu
```

## 冒烟测试

```bash
# 服务启动后（默认含 2 epoch 短训，需 weights 底模）
pip install httpx pillow
python scripts/smoke_test.py
# 只测接口、跳过训练
python scripts/smoke_test.py --no-train
```

## 标注快捷键

| 键 | 作用 |
|---|---|
| W | 画框 |
| E | SAM 点选 |
| 空格+拖 | 平移 |
| 滚轮 | 缩放 |
| 1-9 | 切换类别 |
| Del | 删除选中框 |
| A / D | 上一张 / 下一张 |
| Enter | 确认并下一张 |
| C | 复制上一张标注 |
| Ctrl+Z | 撤销 |

## API

启动后访问 `/docs` 查看 OpenAPI。健康检查：`GET /api/health`。

## 待确认事项

- 权重请按 [docs/20260725_离线权重清单.md](docs/20260725_离线权重清单.md) 备齐后再 build 镜像（`COPY weights`）。
- 正式 mAP 以训练 val 为准；模型仓库「评估」接口为框数级粗评。
- Docker 镜像：torch **2.13.0+cpu**，构建时断言版本串含 `+cpu`；体积实测约 **3.77 GB**（见上文）。
- 开放词表（YOLO-World）在纯 CPU 上约 **0.6~1.2 秒/张**，大批量请先「试 20 张」再全量。

## 版本管理

请在本目录 `yolo-studio/` 内使用 git，**不要**在 `~/project` 顶层 `git init`。
