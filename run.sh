#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  .venv/bin/pip install -U pip
  # CPU 版 torch
  .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  .venv/bin/pip install -r requirements.txt
fi

# 前端 vendor 若不存在则提示
if [[ ! -f web/vendor/vue.global.prod.js ]]; then
  echo "缺少 web/vendor/vue.global.prod.js，请先执行 scripts/fetch_vendor.sh"
  exit 1
fi

export YOLO_OFFLINE="${YOLO_OFFLINE:-1}"
export CPU_THREADS="${CPU_THREADS:-14}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPU_THREADS}"

exec .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port "${APP_PORT:-5016}" --reload
