#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"

run_watch() {
  if [[ ! -x "$PYTHON" ]]; then
    echo "No venv found, run ./scripts/setup.sh first." >&2
    exit 1
  fi
  exec "$PYTHON" -m watch "$@"
}
