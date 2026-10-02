# WATCH

## Setup

**Full pipeline (data, training, evaluation) — WSL/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "tensorflow[and-cuda]>=2.16,<2.19"
pip install -r requirements.txt -e .
```

**Just running live.py — native Windows (needs a camera, so not WSL):**

```bash
python -m venv .venv-win
.venv-win\Scripts\activate
pip install -r requirements-live.txt -e .
python cli\live.py
```

## Run

```bash
# eye model
python cli/eye/extract_nthu.py     # extract labeled eye frames from NTHU-DDD videos
python cli/eye/inspect_data.py     # audit eye datasets (mrl_eye/cew)
python cli/eye/make_splits.py      # build eye train/val/test split
python cli/eye/train.py            # train the eye model
python cli/eye/grid_search.py      # sweep eye model architecture/dropout/lr, resumable
python cli/eye/evaluate.py         # eye test-set metrics
python cli/eye/benchmark.py        # eye model inference speed/FPS

# yawn model
python cli/yawn/extract_yawdd.py   # extract labeled mouth frames from YawDD videos
python cli/yawn/make_splits.py     # build yawn train/val/test split
python cli/yawn/train.py           # train the yawn model
python cli/yawn/evaluate.py        # yawn test-set metrics

# shared
python cli/predict.py photo.jpg    # eye/mouth/head state from a photo
python cli/live.py                 # live webcam alertness monitor
python cli/convert_web_models.py   # export both .keras models to TF.js for web/
```

## Layout

```
cli/eye/, cli/yawn/   pipeline scripts per model
cli/                  shared: predict.py, live.py, convert_web_models.py
src/frames/           the package
configs/config.yaml   all paths, hyperparams, thresholds
data/raw/             raw datasets (not tracked)
data/splits/          train/val/test manifests (tracked)
models/               trained checkpoints (tracked)
results/              evaluation outputs (not tracked)
web/                  browser demo
tests/                unit tests
```
