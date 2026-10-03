import os
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

from watch.config import load_config, manifest_path

CEW_PAGE = "https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html"
CEW_ARCHIVE = "https://drive.usercontent.google.com/download?id=1Z5hZZnkN4VycK-mOOzUX1Zuwty8BEePy&export=download&confirm=t"
MRL_FOLDERS = {"open eyes": "awake", "close eyes": "sleepy"}


def _place(source: Path, target: Path) -> bool:
    """Hardlinks source to target, copying instead across filesystems, and skips existing targets."""
    if target.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return True


def _place_tree(source_dir: Path, target_dir: Path) -> int:
    return sum(_place(path, target_dir / path.relative_to(source_dir)) for path in source_dir.rglob("*") if path.is_file())


def _kaggle(handle: str) -> Path:
    import kagglehub

    print(f"[{handle}] downloading, cached after the first run")
    return Path(kagglehub.dataset_download(handle))


def fetch_mrl(raw_dir: Path) -> int:
    """Restores the train/val/test layout eye_manifest.csv expects, since this Kaggle copy is foldered by eye state."""
    source = _kaggle("tauilabdelilah/mrl-eye-dataset") / "data"
    target = raw_dir / "eye" / "mrl_eye"
    manifest = manifest_path("eye")
    listed = pd.read_csv(manifest).query("dataset == 'mrl_eye'")["filepath"] if manifest.exists() else []
    locations = {Path(p).name: Path(p).relative_to("data/raw/eye/mrl_eye") for p in listed}
    return sum(
        _place(path, target / locations.get(path.name, Path("train") / state / path.name))
        for folder, state in MRL_FOLDERS.items()
        for path in source.glob(f"*/{folder}/*.png")
    )


def fetch_nthu(raw_dir: Path) -> int:
    return _place_tree(_kaggle("ikhlaselhamly/nthu-ddd") / "NTHU-DDD", raw_dir / "eye" / "nthu_ddd" / "train")


def fetch_yawdd(raw_dir: Path) -> int:
    return _place_tree(_kaggle("enider/yawdd-dataset") / "Mirror" / "Mirror", raw_dir / "yawn" / "yawdd_videos" / "Mirror")


def fetch_cew(raw_dir: Path) -> int:
    """Downloads the eye-patch archive linked from the official CEW page, since Kaggle has no correct copy."""
    from unrar.cffi import rarfile

    archive = raw_dir / "eye" / "cew_eye_images.rar"
    if not archive.exists():
        print(f"[cew] downloading from the link on {CEW_PAGE}")
        archive.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CEW_ARCHIVE, archive)

    rar, placed = rarfile.RarFile(str(archive)), 0
    for info in rar.infolist():
        parts = Path(info.filename.replace("\\", "/")).parts
        output = raw_dir / "eye" / "cew" / Path(*parts[1:])
        if info.is_dir() or output.suffix.lower() != ".jpg" or output.exists():
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(rar.read(info.filename))
        placed += 1
    return placed


FETCHERS = {"mrl_eye": fetch_mrl, "cew": fetch_cew, "nthu_ddd": fetch_nthu, "yawdd": fetch_yawdd}


def fetch(datasets: list[str] | None = None) -> None:
    raw_dir = load_config()["paths"]["raw_dir"]
    for name in datasets or FETCHERS:
        print(f"[{name}] placed {FETCHERS[name](raw_dir)} new file(s)")
