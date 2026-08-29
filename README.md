# FRAMES

Facial Recognition & Alertness Monitoring Engine System — real-time pilot/driver alertness monitoring via eye closure, yawning, and head-droop.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "tensorflow[and-cuda]>=2.16,<2.19"
pip install -r requirements.txt -e .
```

`cli/live.py` needs a camera, so run it from a native Windows Python install (with `opencv-python` instead of `opencv-python-headless`), not WSL.

## Run

```bash
python cli/inspect_data.py       # audit datasets
python cli/make_splits.py        # build eye train/val/test split
python cli/train.py              # train the eye model
python cli/evaluate.py           # eye test-set metrics
python cli/extract_yawdd.py      # extract labeled mouth frames from YawDD videos
python cli/make_yawn_splits.py   # build yawn train/val/test split
python cli/train_yawn.py         # train the yawn model
python cli/evaluate_yawn.py      # yawn test-set metrics
python cli/benchmark.py          # inference speed/FPS
python cli/predict.py photo.jpg  # eye/mouth/head state from a photo
python cli/live.py               # live webcam alertness monitor
```
