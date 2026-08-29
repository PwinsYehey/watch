# FRAMES

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "tensorflow[and-cuda]>=2.16,<2.19"
pip install -r requirements.txt -e .
```

## Datasets

Place the datasets in:
- `data/raw/mrl_eye/` — MRL Eye Dataset
- `data/raw/cew/` — CEW (Closed Eyes in the Wild)

## Run

```bash
python cli/inspect_data.py      # audit datasets
python cli/make_splits.py       # build train/val/test split
python cli/grid_search.py       # compare architectures/hyperparameters
python cli/train.py             # train the model
python cli/evaluate.py          # test-set metrics
python cli/predict.py path/to/photo.jpg   # predict eye state from a photo
```
