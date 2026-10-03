$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $Root ".venv-win\Scripts\python.exe"

function Invoke-Watch {
    if (-not (Test-Path $Python)) {
        Write-Error "No venv found, run .\scripts\setup.ps1 first."
    }
    & $Python -m watch @args
    exit $LASTEXITCODE
}
