"""The one property this whole repo exists to guarantee: no subject leaks
across train/val/test. This is a regression test for the exact bug found
in the original motorcycle-rider dataset."""

import pandas as pd

from frames.data.split import split_manifest, report_split


def _make_manifest(n_subjects=50, images_per_subject=10):
    rows = []
    for i in range(n_subjects):
        subject_id = f"s{i:04d}"
        label = "open" if i % 2 == 0 else "closed"
        for j in range(images_per_subject):
            rows.append(
                {
                    "filepath": f"/fake/{subject_id}_{j}.png",
                    "dataset": "fake",
                    "subject_id": subject_id,
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def test_no_subject_spans_multiple_splits():
    manifest = _make_manifest()
    ratios = {"train": 0.7, "val": 0.2, "test": 0.1}
    result = split_manifest(manifest, ratios, seed=42)

    per_group_splits = result.groupby("group_id")["split"].nunique()
    assert (per_group_splits == 1).all(), "a subject/group appeared in more than one split"


def test_report_split_raises_on_synthetic_leak():
    manifest = _make_manifest()
    result = split_manifest(manifest, {"train": 0.7, "val": 0.2, "test": 0.1}, seed=42)

    # Force a leak: reassign one row from an otherwise train-only group to test
    train_group = result.loc[result["split"] == "train", "group_id"].iloc[0]
    leaking_row = result[result["group_id"] == train_group].index[0]
    result.loc[leaking_row, "split"] = "test"

    try:
        report_split(result)
        assert False, "expected report_split to raise on a leaked subject"
    except RuntimeError:
        pass


def test_all_splits_present_and_non_overlapping_sizes():
    manifest = _make_manifest(n_subjects=100)
    result = split_manifest(manifest, {"train": 0.7, "val": 0.2, "test": 0.1}, seed=42)

    counts = result.groupby("split")["group_id"].nunique()
    assert set(counts.index) == {"train", "val", "test"}
    assert counts.sum() == manifest["subject_id"].nunique()
