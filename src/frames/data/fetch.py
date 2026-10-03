import os
import shutil
from pathlib import Path

import pandas as pd

_MRL_HANDLE = "tauilabdelilah/mrl-eye-dataset"
_NTHU_HANDLE = "ikhlaselhamly/nthu-ddd"
_YAWDD_HANDLE = "enider/yawdd-dataset"
_CEW_SOURCE = "https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html"
_CEW_DRIVE_URL = "https://drive.usercontent.google.com/download?id=1Z5hZZnkN4VycK-mOOzUX1Zuwty8BEePy&export=download&confirm=t"

# this Kaggle copy of MRL is foldered by eye state, not by the train/val/test/awake/sleepy layout manifest.csv points at
_MRL_FOLDER_TO_STATE = {"open eyes": "awake", "close eyes": "sleepy"}


def _place(source: Path, target: Path) -> bool:
    """Hardlinks source to target (falling back to a copy across filesystems), skipping targets that already exist."""
    if target.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return True


def _place_tree(source_dir: Path, target_dir: Path) -> int:
    placed = 0
    for path in source_dir.rglob("*"):
        if path.is_file():
            placed += _place(path, target_dir / path.relative_to(source_dir))
    return placed


def _download(handle: str) -> Path:
    import kagglehub

    print(f"\n[{handle}] downloading (cached after the first run)...")
    return Path(kagglehub.dataset_download(handle))


def fetch_mrl(raw_dir: Path, manifest_path: Path) -> None:
    """Rebuilds the original MRL layout from manifest.csv so the tracked split stays valid; images it doesn't list go under train/."""
    source = _download(_MRL_HANDLE) / "data"
    target = raw_dir / "eye" / "mrl_eye"

    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame(columns=["filepath", "dataset"])
    mrl_paths = manifest.loc[manifest["dataset"] == "mrl_eye", "filepath"]
    original_location = {Path(p).name: Path(p).relative_to(Path("data/raw/eye/mrl_eye")) for p in mrl_paths}

    placed = unlisted = 0
    for folder, state in _MRL_FOLDER_TO_STATE.items():
        for path in source.glob(f"*/{folder}/*.png"):
            relative = original_location.get(path.name)
            if relative is None:
                relative = Path("train") / state / path.name
                unlisted += 1
            placed += _place(path, target / relative)

    print(f"[mrl_eye] placed {placed} new image(s) in {target} ({unlisted} not listed in manifest.csv, put under train/)")


def fetch_nthu(raw_dir: Path) -> None:
    source = _download(_NTHU_HANDLE) / "NTHU-DDD"
    target = raw_dir / "eye" / "nthu_ddd" / "train"
    print(f"[nthu_ddd] placed {_place_tree(source, target)} new file(s) in {target}")


def fetch_yawdd(raw_dir: Path) -> None:
    """Keeps only the Mirror videos; extract_yawdd.py never reads the Dash set."""
    source = _download(_YAWDD_HANDLE) / "Mirror" / "Mirror"
    target = raw_dir / "yawn" / "yawdd_videos" / "Mirror"
    print(f"[yawdd] placed {_place_tree(source, target)} new file(s) in {target}")


def fetch_cew(raw_dir: Path) -> None:
    """Downloads the 24x24 eye-patch RAR linked from the official CEW page, since no correct Kaggle mirror exists."""
    import urllib.request

    from unrar.cffi import rarfile

    target = raw_dir / "eye" / "cew"
    archive = raw_dir / "eye" / "cew_eye_images.rar"
    if not archive.exists():
        print(f"\n[cew] downloading eye patches (linked from {_CEW_SOURCE})...")
        archive.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_CEW_DRIVE_URL, archive)

    placed = 0
    rar = rarfile.RarFile(str(archive))
    for info in rar.infolist():
        name = info.filename.replace("\\", "/")
        if info.is_dir() or not name.lower().endswith(".jpg"):
            continue
        # strip the archive's dataset_B_Eye_Images/ top folder so the four label folders sit directly in target
        output = target / Path(*Path(name).parts[1:])
        if output.exists():
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(rar.read(info.filename))
        placed += 1
    print(f"[cew] placed {placed} new image(s) in {target}")


def main():
    import argparse

    from frames.config import load_config

    config = load_config()
    raw_dir = Path(config["paths"]["raw_dir"])
    manifest_path = Path(config["paths"]["splits_dir"]) / "manifest.csv"

    fetchers = {
        "mrl_eye": lambda: fetch_mrl(raw_dir, manifest_path),
        "nthu_ddd": lambda: fetch_nthu(raw_dir),
        "yawdd": lambda: fetch_yawdd(raw_dir),
        "cew": lambda: fetch_cew(raw_dir),
    }
    parser = argparse.ArgumentParser(description="Download the Kaggle datasets and lay them out under data/raw/ where the pipeline expects them.")
    parser.add_argument("--only", nargs="+", choices=list(fetchers), help="Fetch just these datasets (default: all)")
    args = parser.parse_args()

    for name in args.only or fetchers:
        fetchers[name]()
