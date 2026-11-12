#!/usr/bin/env bash
# RN18 交付环境一键安装（Linux/macOS，与 setup_env.bat 等价）
set -euo pipefail
echo "[1/6] 检查 Python ..."
python3 --version || { echo "[失败] 未找到 python3，请安装 Python 3.10+"; exit 1; }
echo "[2/6] 创建虚拟环境 .venv ..."
if [ ! -x ".venv/bin/python" ]; then python3 -m venv .venv; else echo "       .venv 已存在，跳过创建。"; fi
echo "[3/6] 升级 pip ..."
".venv/bin/python" -m pip install --upgrade pip
echo "[4/6] 安装 PyTorch CPU 版 ..."
echo "  提示：CPU 版必须带 --index-url https://download.pytorch.org/whl/cpu，否则可能装成 CUDA 版或版本不匹配。"
".venv/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch==2.2.2+cpu torchvision==0.17.2+cpu
echo "[5/6] 安装其余锁定依赖（requirements.txt）..."
".venv/bin/python" -m pip install -r requirements.txt
echo "[6/6] 验证安装 ..."
".venv/bin/python" -c "import torch,torchvision,gradio,numpy; print('import ok')"
echo ""
echo "环境就绪，运行 ./run.sh 启动"
