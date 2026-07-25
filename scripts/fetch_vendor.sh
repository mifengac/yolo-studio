#!/usr/bin/env bash
# 在有网机器下载前端本地依赖（禁止 CDN）
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p web/vendor
curl -fsSL -o web/vendor/vue.global.prod.js \
  https://unpkg.com/vue@3.5.13/dist/vue.global.prod.js
curl -fsSL -o web/vendor/echarts.min.js \
  https://unpkg.com/echarts@5.5.1/dist/echarts.min.js
echo "vendor 已就绪:"
ls -lh web/vendor/
