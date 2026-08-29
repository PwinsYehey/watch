"""Load the project's central YAML config."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve all path-like entries relative to the project root so scripts
    # behave the same no matter which directory they're invoked from.
    for key, value in config["paths"].items():
        config["paths"][key] = str(PROJECT_ROOT / value)

    return config
