# WATCH

**W**akefulness **A**ssessment through **T**racking of **C**ues & **H**ead: real-time driver and pilot alertness monitoring from eye closure, yawning and head droop.

Two small CNNs (eye open/closed, yawn/no yawn) are combined with MediaPipe face-landmark geometry (EAR, MAR, head pitch and roll) and per-user calibration.

## Platforms

| Machine | Used for | Setup |
|---|---|---|
| Linux / WSL2 | data, training, evaluation, export | `./scripts/setup.sh` |
| Windows | live monitor, photo prediction, web demo | `.\scripts\setup.ps1` |

Both need Python 3.10 to 3.12; the setup scripts install 3.12 through [uv](https://docs.astral.sh/uv/) automatically.
On Windows, allow local scripts once with `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Actions

Every action is `./scripts/<action>.sh` on Linux and `.\scripts\<action>.ps1` on Windows, both running `python -m watch <action>`.

| Action | What it does |
|---|---|
| `setup` | create the venv and install dependencies, then run `check` |
| `fetch-data` | download MRL, CEW, NTHU-DDD and YawDD into `data/raw/` (`--only mrl_eye cew nthu_ddd yawdd`) |
| `prepare-data` | extract NTHU eye crops and YawDD mouth crops, audit them and rebuild `data/splits/` |
| `train eye\|yawn\|all` | train and keep the new model only if it beats the current one on validation (`--force`, `--sweep`) |
| `evaluate [eye\|yawn\|all]` | test-set report, confusion matrix and ROC curve in `results/<model>/` |
| `benchmark [eye\|yawn\|all]` | inference latency and model size |
| `export` | write `web/models/`, `web/config.js` and `models/*.tflite` |
| `live` | webcam monitor (`--camera N`), press `q` to quit |
| `web` | serve the browser demo at http://localhost:8765 (`--port N`) |
| `predict <image>` | eye, mouth and head state for one photo |
| `test` | run the unit tests (extra arguments go to pytest) |
| `pipeline` | `fetch-data`, `prepare-data`, `train all`, `evaluate all`, `export` |

`fetch-data`, `prepare-data`, `train`, `evaluate` and `pipeline` run on the Linux machine only.

## Live monitor

1. Calibration takes 4 seconds: look at the camera normally.
2. The banner shows **NORMAL**, **WARNING**, **CRITICAL** or **NO FACE DETECTED**, and an alarm beeps while critical.
3. A sign of drowsiness is eyes closed for 1.5 s, a yawn held for 2 s, or the head drooped for 2 s, or a high average of any of these over a short window.
4. One sign gives a warning and starts a 30 s watch. It turns critical if the sign repeats, another sign appears, or it lasts (head drooped 4 s). Eyes closed for 1.5 s goes straight to critical. Three yawns are needed on their own.
5. Critical drops back to a warning after 5 s without signs. If the face leaves the view, the current level is held.
6. The dashboard beside the video shows the alert level, each sign with its timer, what is being detected, and recent level changes.

All thresholds live in `configs/config.yaml`; run `export` after changing them so the web demo picks them up.

## Layout

```
configs/config.yaml   paths, datasets, model and alert settings
scripts/              setup and action scripts (.sh and .ps1)
src/watch/            the package (python -m watch)
data/raw/             downloaded datasets and extracted crops (not tracked)
data/splits/          subject-disjoint train/val/test manifests
models/               trained .keras and exported .tflite models
results/              evaluation output (not tracked)
web/                  browser demo
tests/                unit tests
```
