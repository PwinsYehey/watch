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
python cli/eye/extract_nthu.py     # extract labeled eye frames from NTHU-DDD videos
python cli/eye/inspect_data.py     # audit eye datasets (mrl_eye/cew)
python cli/eye/make_splits.py      # build eye train/val/test split
python cli/eye/train.py            # train the eye model
python cli/eye/grid_search.py      # sweep eye model architecture/dropout/lr, resumable
python cli/eye/evaluate.py         # eye test-set metrics
python cli/eye/benchmark.py        # eye model inference speed/FPS

python cli/yawn/extract_yawdd.py   # extract labeled mouth frames from YawDD videos
python cli/yawn/make_splits.py     # build yawn train/val/test split
python cli/yawn/train.py           # train the yawn model
python cli/yawn/evaluate.py        # yawn test-set metrics

python cli/predict.py photo.jpg    # eye/mouth/head state from a photo
python cli/live.py                 # live webcam alertness monitor
python cli/convert_web_models.py   # export both .keras models to TF.js for web/
```
