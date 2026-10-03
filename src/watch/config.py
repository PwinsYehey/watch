from functools import cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
MODEL_NAMES = ("eye", "yawn")


@cache
def load_config() -> dict:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["paths"] = {key: PROJECT_ROOT / value for key, value in config["paths"].items()}
    return config


def manifest_path(name: str) -> Path:
    return load_config()["paths"]["splits_dir"] / f"{name}_manifest.csv"


def model_path(name: str, suffix: str = ".keras") -> Path:
    return load_config()["paths"]["models_dir"] / f"{name}_model{suffix}"


def results_dir(name: str) -> Path:
    path = load_config()["paths"]["results_dir"] / name
    path.mkdir(parents=True, exist_ok=True)
    return path
