#!/usr/bin/env bash
# ==============================================================
#  RN18 - environment setup (Linux / macOS)
#  Same steps as setup_env.bat. Keep this file pure ASCII with LF.
#  Chinese guide: docs/STARTUP.md  (section 1)
# ==============================================================
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/6] Checking Python ..."
python3 --version

echo "[2/6] Creating virtual environment .venv ..."
if [ -x ".venv/bin/python" ]; then
  echo "       .venv already exists, skip."
else
  python3 -m venv .venv
fi

echo "[3/6] Upgrading pip ..."
".venv/bin/python" -m pip install --upgrade pip

echo "[4/6] Installing PyTorch CPU build ..."
echo "       torch==2.2.2+cpu / torchvision==0.17.2+cpu"
echo "       CPU wheels only, from the official PyTorch CPU index."
".venv/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch==2.2.2+cpu torchvision==0.17.2+cpu

echo "[5/6] Installing requirements.txt ..."
".venv/bin/python" -m pip install -r requirements.txt

echo "[6/6] Verifying imports ..."
".venv/bin/python" -c "import torch, torchvision, gradio, numpy; print('import ok')"

echo
echo "Done. Run ./run.sh to launch the GUI."
