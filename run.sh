#!/usr/bin/env bash
# ==============================================================
#  RN18 - launch the GUI (Linux / macOS)
#  Activates .venv, then runs app.py. Run ./setup_env.sh first.
#  Keep this file pure ASCII with LF.
# ==============================================================
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "[ERROR] .venv not found. Run ./setup_env.sh first."
  exit 1
fi

echo "Activating .venv and starting the RN18 GUI ..."
# shellcheck source=/dev/null
source ".venv/bin/activate"
python app.py
