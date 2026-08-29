from pathlib import Path

import numpy as np
import pandas as pd


def _assign_subject_splits(subjects: list[str], ratios: dict, rng: np.random.Generator) -> dict[str, str]:
    subjects = list(subjects)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = round(n * ratios["train"])
    n_val = round(n * ratios["val"])
    # the remainder always goes to test, so rounding can never drop a subject
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    assignment = {}
    for subject in subjects[:n_train]:
        assignment[subject] = "train"
    for subject in subjects[n_train : n_train + n_val]:
        assignment[subject] = "val"
    for subject in subjects[n_train + n_val :]:
        assignment[subject] = "test"
    return assignment


def split_manifest(manifest: pd.DataFrame, ratios: dict, seed: int = 42) -> pd.DataFrame:
    """Adds a 'split' column to `manifest`, namespacing subject_id per-dataset so identical ids across datasets never collide."""
    manifest = manifest.copy()
    manifest["group_id"] = manifest["dataset"] + "::" + manifest["subject_id"].astype(str)

    majority_label = manifest.groupby("group_id")["label"].agg(lambda s: s.mode().iat[0])

    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    for label, subjects_for_label in majority_label.groupby(majority_label):
        assignment.update(_assign_subject_splits(list(subjects_for_label.index), ratios, rng))

    manifest["split"] = manifest["group_id"].map(assignment)
    return manifest


def report_split(manifest: pd.DataFrame) -> None:
    print("\nSplit sizes (images):")
    print(manifest.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())

    print("\nSplit sizes (unique subjects):")
    print(manifest.groupby("split")["group_id"].nunique().to_string())

    overlap = (
        manifest.groupby("group_id")["split"].nunique().loc[lambda s: s > 1]
    )
    if len(overlap):
        raise RuntimeError(f"Leakage bug: {len(overlap)} subject(s) span multiple splits: {list(overlap.index)}")
    print("\nNo subject appears in more than one split. OK.")


def main():
    import argparse

    from frames.config import PROJECT_ROOT, load_config
    from frames.data.manifest import build_manifest

    parser = argparse.ArgumentParser(description="Build a subject-disjoint train/val/test split across enabled datasets.")
    parser.add_argument("--output", default=None, help="Output CSV path (default: <splits_dir>/manifest.csv)")
    args = parser.parse_args()

    config = load_config()
    raw_dir = Path(config["paths"]["raw_dir"])

    manifests = []
    for name, ds_config in config["datasets"].items():
        if not ds_config.get("enabled", True):
            continue
        manifests.append(build_manifest(name, raw_dir / ds_config["raw_subdir"]))

    combined = pd.concat(manifests, ignore_index=True)
    combined = split_manifest(combined, config["split"]["ratios"], seed=config["project"]["seed"])
    report_split(combined)

    # store paths relative to the project root so manifest.csv is portable across machines/clones
    combined["filepath"] = combined["filepath"].apply(lambda p: str(Path(p).relative_to(PROJECT_ROOT)))

    splits_dir = Path(config["paths"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else splits_dir / "manifest.csv"
    combined.to_csv(output_path, index=False)
    print(f"\nSaved combined split manifest to: {output_path}")


def main_yawn():
    from frames.config import PROJECT_ROOT, load_config
    from frames.data.manifest import build_manifest

    config = load_config()
    raw_dir = Path(config["paths"]["raw_dir"])

    manifests = []
    for name, ds_config in config["yawn_dataset"].items():
        if not ds_config.get("enabled", True):
            continue
        manifests.append(build_manifest(name, raw_dir / ds_config["raw_subdir"]))

    combined = pd.concat(manifests, ignore_index=True)
    combined = split_manifest(combined, config["split"]["ratios"], seed=config["project"]["seed"])
    report_split(combined)

    combined["filepath"] = combined["filepath"].apply(lambda p: str(Path(p).relative_to(PROJECT_ROOT)))

    splits_dir = Path(config["paths"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)
    output_path = splits_dir / "yawn_manifest.csv"
    combined.to_csv(output_path, index=False)
    print(f"\nSaved yawn split manifest to: {output_path}")


if __name__ == "__main__":
    main()
