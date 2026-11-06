from pathlib import Path

import pandas as pd

from watch.config import load_config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _images(root: Path, pattern: str = "*"):
    return sorted(p for p in root.rglob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _rows(dataset: str, entries) -> pd.DataFrame:
    return pd.DataFrame(
        [{"filepath": str(path), "dataset": dataset, "subject_id": subject, "label": label} for path, subject, label in entries],
        columns=["filepath", "dataset", "subject_id", "label"],
    )


def build_mrl_eye(root: Path) -> pd.DataFrame:
    """Parses <subject>_<image>_<gender>_<glasses>_<eyestate>_..., the MRL filename convention."""
    entries = []
    for path in _images(root):
        parts = path.stem.split("_")
        if len(parts) >= 5 and parts[4] in ("0", "1", "b"):
            entries.append((path, parts[0], {"0": "eyes_closed", "1": "eyes_opened", "b": "blocked"}[parts[4]]))
    return _rows("mrl_eye", entries)


def build_cew(root: Path) -> pd.DataFrame:
    """Labels from the closed/open folder name, grouping each face's _L/_R crops under one subject."""
    entries = []
    for path in _images(root):
        folder = path.parent.name.lower()
        label = "blocked" if "blocked" in folder else "eyes_closed" if "closed" in folder else "eyes_opened" if "open" in folder else None
        if label:
            entries.append((path, path.stem.removesuffix("_L").removesuffix("_R"), label))
    return _rows("cew", entries)


def _build_labeled_folders(dataset: str, root: Path, folders: dict[str, str]) -> pd.DataFrame:
    return _rows(dataset, [(path, path.stem.split("_")[0], label) for folder, label in folders.items() for path in _images(root / folder)])


def build_nthu_eye(root: Path) -> pd.DataFrame:
    return _build_labeled_folders("nthu_eye", root, {"open": "eyes_opened", "closed": "eyes_closed"})


def build_yawdd_eye(root: Path) -> pd.DataFrame:
    return _build_labeled_folders("yawdd_eye", root, {"open": "eyes_opened", "closed": "eyes_closed", "blocked": "blocked"})


def build_yawdd(root: Path) -> pd.DataFrame:
    return _build_labeled_folders("yawdd", root, {"yawn": "yawn", "no_yawn": "no_yawn"})


def build_own_eye(root: Path) -> pd.DataFrame:
    """Crops from `watch record`, named <subject>_<take>_<side>_<n>, with a take_id so whole takes can be held out."""
    rows = _build_labeled_folders("own_eye", root, {"open": "eyes_opened", "closed": "eyes_closed"})
    rows["take_id"] = rows["filepath"].map(lambda p: "_".join(Path(p).stem.split("_")[:2]))
    return rows


BUILDERS = {
    "mrl_eye": build_mrl_eye,
    "cew": build_cew,
    "nthu_eye": build_nthu_eye,
    "yawdd_eye": build_yawdd_eye,
    "yawdd": build_yawdd,
    "own_eye": build_own_eye,
}


def build_manifest(name: str) -> pd.DataFrame:
    """Combines every dataset configured for one model into a single manifest."""
    config = load_config()
    raw_dir = config["paths"]["raw_dir"]
    frames = []
    for dataset, subdir in config["datasets"][name].items():
        frame = BUILDERS[dataset](raw_dir / subdir)
        if frame.empty:
            raise FileNotFoundError(f"No {dataset} images in {raw_dir / subdir}, run ./scripts/fetch-data.sh and ./scripts/prepare-data.sh.")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def audit(manifest: pd.DataFrame) -> None:
    for dataset, rows in manifest.groupby("dataset"):
        per_subject = rows.groupby("subject_id").size()
        balance = ", ".join(f"{label}={count}" for label, count in rows["label"].value_counts().items())
        print(
            f"[{dataset}] {len(rows)} images, {len(per_subject)} subjects "
            f"(per subject min={per_subject.min()} median={per_subject.median():.0f} max={per_subject.max()}), {balance}"
        )
        if len(per_subject) < 15:
            print(f"  warning: only {len(per_subject)} subjects, so its test results cover very few unseen people")
