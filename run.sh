#!/usr/bin/env bash
# 激活 .venv 并启动 RN18 图形界面（Linux/macOS，与 run.bat 等价）
set -euo pipefail
if [ ! -x ".venv/bin/python" ]; then echo "[失败] 未找到 .venv，请先运行 ./setup_env.sh"; exit 1; fi
echo "正在激活 .venv 并启动 RN18 图形界面 ..."
source ".venv/bin/activate"
python app.py
