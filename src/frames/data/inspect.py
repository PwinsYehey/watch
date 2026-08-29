"""Reports subject counts, images-per-subject, and class balance for a dataset manifest, before any training happens."""

from pathlib import Path

import pandas as pd


def summarize(manifest: pd.DataFrame, dataset_name: str) -> None:
    if manifest.empty:
        print(f"[{dataset_name}] manifest is EMPTY -- check raw_dir contents and filename conventions.")
        return

    n_images = len(manifest)
    n_subjects = manifest["subject_id"].nunique()
    per_subject = manifest.groupby("subject_id").size()

    print(f"\n=== {dataset_name} ===")
    print(f"Images: {n_images}")
    print(f"Unique subjects: {n_subjects}")
    print(f"Images per subject: min={per_subject.min()}, median={per_subject.median():.0f}, max={per_subject.max()}")

    print("\nClass balance:")
    print(manifest["label"].value_counts().to_string())

    print("\nClass balance per subject (first 10 subjects):")
    print(manifest.groupby(["subject_id", "label"]).size().unstack(fill_value=0).head(10).to_string())

    if n_subjects < 15:
        print(
            f"\n WARNING: only {n_subjects} unique subjects. A subject-disjoint test split "
            "will be evaluating generalization to very few unseen people -- note this as a "
            "limitation, don't rely on this dataset alone for the final claim."
        )

    avg_per_subject = n_images / n_subjects
    if avg_per_subject > 1000:
        print(
            f"\n NOTE: ~{avg_per_subject:.0f} images/subject on average. Verify this isn't "
            "near-duplicate video frames (check a few filenames/timestamps) before trusting "
            "that more images per subject means more signal."
        )


def main():
    import argparse

    from frames.config import load_config
    from frames.data.manifest import build_manifest

    parser = argparse.ArgumentParser(description="Audit raw dataset(s) for subject counts and class balance.")
    parser.add_argument("--dataset", choices=["mrl_eye", "cew", "all"], default="all")
    args = parser.parse_args()

    config = load_config()
    raw_dir = Path(config["paths"]["raw_dir"])

    dataset_names = list(config["datasets"].keys()) if args.dataset == "all" else [args.dataset]

    for name in dataset_names:
        ds_config = config["datasets"][name]
        if not ds_config.get("enabled", True):
            continue
        manifest = build_manifest(name, raw_dir / ds_config["raw_subdir"])
        summarize(manifest, name)


if __name__ == "__main__":
    main()
