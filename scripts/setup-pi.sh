#!/usr/bin/env bash
# Creates .venv with Python 3.12 and the runtime dependencies on a Raspberry Pi, without the training extras.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv venv --python 3.12 --seed --allow-existing "$ROOT/.venv"
PYTHON="$ROOT/.venv/bin/python"
UV_HTTP_TIMEOUT=300 uv pip install --python "$PYTHON" -e "$ROOT[dev]"
"$PYTHON" -m watch check
