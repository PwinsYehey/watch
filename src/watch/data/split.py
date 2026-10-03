from pathlib import Path

import numpy as np
import pandas as pd

from watch.config import PROJECT_ROOT, load_config, manifest_path
from watch.data.manifest import audit, build_manifest


def _assign(subjects: list[str], ratios: dict, rng: np.random.Generator) -> dict[str, str]:
    subjects = list(subjects)
    rng.shuffle(subjects)
    n_train = round(len(subjects) * ratios["train"])
    n_val = min(round(len(subjects) * ratios["val"]), len(subjects) - n_train)
    return {
        subject: "train" if i < n_train else "val" if i < n_train + n_val else "test"
        for i, subject in enumerate(subjects)
    }


def split_manifest(manifest: pd.DataFrame, ratios: dict, seed: int, force: dict[str, str] | None = None) -> pd.DataFrame:
    """Assigns whole subjects to train/val/test, stratified by dataset and each subject's majority label."""
    manifest = manifest.copy()
    manifest["group_id"] = manifest["dataset"] + "::" + manifest["subject_id"].astype(str)

    groups = manifest.groupby("group_id").agg(dataset=("dataset", "first"), label=("label", lambda s: s.mode().iat[0]))
    rng = np.random.default_rng(seed)
    assignment = {}
    for _, stratum in groups.groupby(["dataset", "label"]):
        assignment.update(_assign(list(stratum.index), ratios, rng))
    assignment.update(force or {})

    manifest["split"] = manifest["group_id"].map(assignment)
    return manifest


def check_no_leakage(manifest: pd.DataFrame) -> None:
    leaked = manifest.groupby("group_id")["split"].nunique().loc[lambda s: s > 1]
    if len(leaked):
        raise RuntimeError(f"{len(leaked)} subject(s) span multiple splits: {list(leaked.index)}")


def report(manifest: pd.DataFrame) -> None:
    print(manifest.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
    print(manifest.groupby("split")["group_id"].nunique().rename("subjects").to_string())


def make_split(name: str) -> Path:
    config = load_config()
    manifest = build_manifest(name)
    audit(manifest)
    manifest = split_manifest(manifest, config["split"]["ratios"], config["seed"], config["split"]["force"])
    check_no_leakage(manifest)
    report(manifest)

    # Relative POSIX paths keep the manifest valid on any OS and clone location.
    manifest["filepath"] = manifest["filepath"].map(lambda p: Path(p).relative_to(PROJECT_ROOT).as_posix())
    path = manifest_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    print(f"Saved {path}")
    return path


def load_split(name: str) -> pd.DataFrame:
    path = manifest_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found, run ./scripts/prepare-data.sh first.")
    manifest = pd.read_csv(path, dtype={"subject_id": str})
    manifest["filepath"] = manifest["filepath"].map(lambda p: str(PROJECT_ROOT / p))
    return manifest
