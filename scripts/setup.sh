#!/usr/bin/env bash
# Creates .venv with Python 3.12 and the training dependencies.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv venv --python 3.12 --seed --allow-existing "$ROOT/.venv"
PYTHON="$ROOT/.venv/bin/python"
# opencv-python-headless and opencv-python share the cv2 folder, so keep only the GUI build.
if uv pip show --python "$PYTHON" opencv-python-headless >/dev/null 2>&1; then
  uv pip uninstall --python "$PYTHON" opencv-python-headless
  uv pip install --python "$PYTHON" --reinstall opencv-python
fi
UV_HTTP_TIMEOUT=300 uv pip install --python "$PYTHON" -e "$ROOT[train,dev]"
"$PYTHON" -m watch check
