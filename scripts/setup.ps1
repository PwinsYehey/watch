# Creates .venv-win with Python 3.12 and the runtime dependencies.
$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
}

$env:UV_HTTP_TIMEOUT = "300"
$env:UV_LINK_MODE = "copy"
uv venv --python 3.12 --seed --allow-existing "$Root\.venv-win"
$Python = "$Root\.venv-win\Scripts\python.exe"
uv pip install --python $Python -e "$Root[dev]"
& $Python -m watch check
