# WATCH

**W**akefulness **A**ssessment through **T**racking of **C**ues & **H**ead: real-time driver and pilot alertness monitoring from eye closure, yawning and head droop.

Two small CNNs (eye open/closed, yawn/no yawn) are combined with MediaPipe face-landmark geometry (EAR, MAR, head pitch and roll) and per-user calibration.

## Platforms

| Machine | Used for | Setup |
|---|---|---|
| Linux / WSL2 | data, training, evaluation, export | `./scripts/setup.sh` |
| Windows | live monitor, photo prediction, web demo | `.\scripts\setup.ps1` |
| Raspberry Pi | live monitor from a Pi camera, viewed in a browser | `./scripts/setup-pi.sh` |

All need Python 3.10 to 3.12; the setup scripts install 3.12 through [uv](https://docs.astral.sh/uv/) automatically.
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
| `live` | webcam monitor (`--camera N`, or `--camera pi` for a Raspberry Pi camera module; `--stream [PORT]` to view it in a browser, `--allow-shutdown` to add a power-off button there): `q` quit, `m` mute, `e` export a session report |
| `web` | serve the browser demo at http://localhost:8765 (`--port N`) |
| `predict <image>` | eye, mouth and head state for one photo |
| `test` | run the unit tests (extra arguments go to pytest) |
| `pipeline` | `fetch-data`, `prepare-data`, `train all`, `evaluate all`, `export` |

`fetch-data`, `prepare-data`, `train`, `evaluate` and `pipeline` run on the Linux machine only.

## Live monitor

1. Calibration takes 4 seconds: look at the camera normally.
2. The banner shows **NORMAL**, **WARNING**, **CRITICAL** or **NO FACE DETECTED**, and an alarm beeps while critical.
3. A sign of drowsiness is eyes closed for 1.5 s, a yawn held for 2 s, or the head drooped (down, to a side, or diagonally) for 1.5 s, or a high average of any of these over a short window.
4. One sign gives a warning and starts a 30 s watch. It turns critical if the sign repeats, another sign appears, or it lasts (eyes closed 3 s, head drooped 4 s). Three yawns are needed on their own.
5. Critical drops back to a warning after 5 s without signs. If the face leaves the view, the level is held; when the face comes back after 3 s or more, it recalibrates and starts again from normal.
6. If the camera disconnects, the dashboard shows **CAMERA LOST** and keeps retrying it; the alarm sounds after 3 s, and it recalibrates once the camera is back.
7. The dashboard beside the video shows the alert level, each sign with its timer, what is being detected, and recent level changes.
8. The shoulders and head are tracked by a body tracker too: calibration measures the sitting posture, and the dashboard shows a Body sign, whether a person is in view, and how far the shoulders have dropped or leaned and the head has dropped toward them. A slump raises alerts like the other signs; while a hand is raised to the face or helmet, the shoulder lean is ignored, since lifting an arm tilts the shoulders. When a visor and mask hide the face but the person is still in view, the dashboard shows **FACE COVERED** and keeps monitoring this posture.
9. If the eyes cannot be read for 60 s while the person is still in view (a tinted visor, or a visor and mask), it beeps and asks for a check-in: press `Space` (or the "I'm awake" button in the web demo). No answer within 5 s gives a warning, and 5 s more makes it critical. Answers and their reaction times go into the session log.
10. Export saves a session report: a PNG line graph of the pilot's alertness over time (normal, warning, critical, with no face, calibrating and camera lost shaded), plus a CSV log of every change and its reason (`E` in the live monitor, saved to `results/sessions/`; the Export button in the web demo).

All thresholds live in `configs/config.yaml`; run `export` after changing them so the web demo picks them up.

## Raspberry Pi

A Pi with a camera module and no screen runs the same monitor and shows it in a browser on another machine:

1. Connect the camera ribbon with the Pi powered off, then check it with `rpicam-hello --list-cameras`. An IMX219 camera may need a boot setting first; see the guide.
2. Run `./scripts/setup-pi.sh` once.
3. Run `./scripts/live.sh --camera pi --stream` and open `http://<pi address>:8766` on the other machine.

The page has buttons for mute, export and stop (the `m`, `e` and `q` keys also work), and plays the alarm once **Enable alarm sound** is clicked. With `--allow-shutdown` it also has a button that powers the Pi off. Reports are saved on the Pi in `results/sessions/`.

The hardware in use is a Raspberry Pi 4 Model B (8 GB), a 32 GB microSD card and an IMX219 night vision camera with two IR LED modules. The picture is black and white with lowered exposure, so a face near the IR LEDs stays readable.

The full walkthrough, with the hardware, step-by-step use, every command, a troubleshooting table and the known limits, is in [docs/raspberry-pi.md](docs/raspberry-pi.md).

To start the monitor whenever the Pi boots, install [scripts/watch-pi.service](scripts/watch-pi.service) as described at the top of that file. The service restarts the monitor if it stops, so **Stop monitor** then acts as a restart; use `systemctl --user stop watch.service` to stop it for good.

## Web demo performance

Low FPS on a laptop with two GPUs usually means the browser is on the built-in one (the dashboard's GPU row is yellow). To use the dedicated GPU on Windows:

1. **Settings → System → Display → Graphics → Microsoft Edge → Options → High performance → Save.**
2. In `edge://settings/system`, turn on **Use graphics acceleration when available**.
3. Close every Edge window and reopen it. The GPU row should turn green.

## Architecture

The full flow from camera input to alerts, display and reports is in [docs/architecture-flowchart.png](docs/architecture-flowchart.png). Its source, [docs/architecture-flowchart.mmd](docs/architecture-flowchart.mmd), is a [Mermaid](https://mermaid.js.org) flowchart that can be edited and re-rendered at [mermaid.live](https://mermaid.live).

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
docs/                 architecture flowchart
tests/                unit tests
```
