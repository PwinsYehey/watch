from pathlib import Path

import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _iter_images(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def build_mrl_eye_manifest(raw_dir: str | Path) -> pd.DataFrame:
    """Parses <subjectID>_<imageID>_<gender>_<glasses>_<eyestate>_<reflections>_<lighting>_<sensor>.png, the documented MRL filename convention."""
    root = Path(raw_dir)
    rows = []
    skipped = 0

    for path in _iter_images(root):
        parts = path.stem.split("_")
        if len(parts) < 5:
            skipped += 1
            continue

        subject_id, eyestate = parts[0], parts[4]
        if eyestate not in ("0", "1"):
            skipped += 1
            continue

        rows.append(
            {
                "filepath": str(path),
                "dataset": "mrl_eye",
                "subject_id": subject_id,
                "label": "open" if eyestate == "1" else "closed",
            }
        )

    if skipped:
        print(f"[mrl_eye] skipped {skipped} file(s) that didn't match the expected filename pattern")

    return pd.DataFrame(rows)


def _strip_eye_side_suffix(stem: str) -> str:
    """Strips a trailing _L/_R so a person's left and right eye crops share one subject_id."""
    if stem.endswith("_L") or stem.endswith("_R"):
        return stem[:-2]
    return stem


def build_cew_manifest(raw_dir: str | Path) -> pd.DataFrame:
    """Reads the label from 'closed'/'open' in the file's path, grouping each person's left/right eye crops under one subject_id."""
    root = Path(raw_dir)
    rows = []
    skipped = 0

    for path in _iter_images(root):
        path_lower = str(path).lower()
        if "closed" in path_lower:
            label = "closed"
        elif "open" in path_lower:
            label = "open"
        else:
            skipped += 1
            continue

        rows.append(
            {
                "filepath": str(path),
                "dataset": "cew",
                "subject_id": _strip_eye_side_suffix(path.stem),
                "label": label,
            }
        )

    if skipped:
        print(f"[cew] skipped {skipped} file(s) with no 'closed'/'open' token in their path")

    return pd.DataFrame(rows)


def build_yawdd_manifest(raw_dir: str | Path) -> pd.DataFrame:
    """Parses '<subjectId>_<category>_<frameIdx>.jpg' (e.g. 'male7_yawning_00250.jpg'), a real subject id from the YawDD Mirror set."""
    root = Path(raw_dir)
    rows = []

    for label, subdir in [("yawn", "yawn"), ("no_yawn", "no_yawn")]:
        for path in sorted((root / subdir).glob("*.jpg")):
            subject_id = path.stem.split("_")[0]
            rows.append(
                {
                    "filepath": str(path),
                    "dataset": "yawdd",
                    "subject_id": subject_id,
                    "label": label,
                }
            )

    return pd.DataFrame(rows)


def build_nthu_eye_manifest(raw_dir: str | Path) -> pd.DataFrame:
    """Parses '<subjectId>_<glasses>_<scenario>_<frameIdx>_<side>.jpg', eye crops extracted via cli/extract_nthu.py from the real NTHU-DDD subject ids."""
    root = Path(raw_dir)
    rows = []

    for label, subdir in [("open", "open"), ("closed", "closed")]:
        for path in sorted((root / subdir).glob("*.jpg")):
            subject_id, glasses = path.stem.split("_")[0], path.stem.split("_")[1]
            rows.append(
                {
                    "filepath": str(path),
                    "dataset": "nthu_eye",
                    "subject_id": subject_id,
                    "label": label,
                    "glasses": glasses,
                }
            )

    return pd.DataFrame(rows)


BUILDERS = {
    "mrl_eye": build_mrl_eye_manifest,
    "cew": build_cew_manifest,
    "yawdd": build_yawdd_manifest,
    "nthu_eye": build_nthu_eye_manifest,
}


def build_manifest(dataset_name: str, raw_dir: str | Path) -> pd.DataFrame:
    if dataset_name not in BUILDERS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Known: {list(BUILDERS)}")
    return BUILDERS[dataset_name](raw_dir)
