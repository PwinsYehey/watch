import pandas as pd
import pytest

from watch.data.split import check_no_leakage, split_manifest

RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}


def _manifest(n_subjects=50, images_per_subject=10):
    return pd.DataFrame(
        [
            {"filepath": f"/fake/s{i}_{j}.png", "dataset": "fake", "subject_id": f"s{i:04d}", "label": "open" if i % 2 else "closed"}
            for i in range(n_subjects)
            for j in range(images_per_subject)
        ]
    )


def test_no_subject_spans_multiple_splits():
    result = split_manifest(_manifest(), RATIOS, seed=42)
    assert (result.groupby("group_id")["split"].nunique() == 1).all()


def test_leak_is_detected():
    result = split_manifest(_manifest(), RATIOS, seed=42)
    result.loc[result.index[0], "split"] = "test" if result.loc[result.index[0], "split"] != "test" else "train"
    with pytest.raises(RuntimeError):
        check_no_leakage(result)


def test_every_subject_assigned_to_all_three_splits():
    manifest = _manifest(n_subjects=100)
    counts = split_manifest(manifest, RATIOS, seed=42).groupby("split")["group_id"].nunique()
    assert set(counts.index) == {"train", "val", "test"}
    assert counts.sum() == manifest["subject_id"].nunique()


def test_forced_split_overrides_random_assignment():
    result = split_manifest(_manifest(), RATIOS, seed=42, force={"fake::s0000": "test"})
    assert set(result.loc[result["subject_id"] == "s0000", "split"]) == {"test"}
