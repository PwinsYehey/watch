import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from watch.config import load_config
from watch.face import FaceTracker

YAWDD_LABELS = {"normal": "no_yawn", "talking": "no_yawn", "yawning": "yawn"}
YAWDD_FOLDERS = {"Female_mirror": "female", "Male_mirror Avi Videos": "male"}


def _fresh_dir(path: Path) -> Path:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


def _remove_overlay(image: np.ndarray) -> np.ndarray:
    """Inpaints the coloured detection boxes NTHU-DDD burns into its near-grayscale IR frames."""
    saturation = image.max(axis=2).astype(np.int16) - image.min(axis=2).astype(np.int16)
    mask = cv2.dilate((saturation > 30).astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


def parse_nthu(path: Path) -> tuple[str, str, str, str] | None:
    """Parses <subject>_<glasses>_<scenario>_<frame>_<coarseLabel>.jpg into (subject, glasses, scenario, frame)."""
    parts = path.stem.split("_")
    return tuple(parts[:4]) if len(parts) == 5 else None


def parse_yawdd(path: Path) -> tuple[str, str, str] | None:
    """Parses <num>-<Gender><Condition>-<Category>.avi into (num, condition, category)."""
    parts = path.stem.split("-")
    if len(parts) < 3:
        return None
    condition = parts[1].removeprefix("Female").removeprefix("Male")
    return parts[0], condition, parts[-1].lower()


def extract_nthu() -> None:
    """Saves eye crops labelled by their own EAR, since NTHU's drowsy label is per clip, not per frame."""
    config = load_config()
    raw_dir = config["paths"]["raw_dir"]
    settings, ear = config["extraction"]["nthu"], config["ear"]
    output_dir = _fresh_dir(raw_dir / config["datasets"]["eye"]["nthu_eye"])
    seen, counts = Counter(), Counter()

    with FaceTracker(static_image_mode=True) as tracker:
        for path in sorted((raw_dir / "eye" / "nthu_ddd" / "train").rglob("*.jpg")):
            parsed = parse_nthu(path)
            if parsed is None:
                continue
            subject, glasses, scenario, frame_idx = parsed
            index = seen[(subject, glasses, scenario)]
            seen[(subject, glasses, scenario)] += 1
            if index % settings["frame_stride"] or index // settings["frame_stride"] >= settings["max_frames_per_group"]:
                continue

            frame = cv2.imread(str(path))
            regions = tracker.extract(_remove_overlay(frame)) if frame is not None else None
            for side, eye in (regions.eyes.items() if regions else ()):
                label = "open" if eye.ratio >= ear["open_threshold"] else "closed" if eye.ratio <= ear["closed_threshold"] else None
                if label is None:
                    continue
                out_path = output_dir / label / f"{subject}_{glasses}_{scenario}_{frame_idx}_{side}.jpg"
                out_path.parent.mkdir(exist_ok=True)
                if cv2.imwrite(str(out_path), eye.crop):
                    counts[label] += 1
    print(f"NTHU eye crops: {dict(counts)}")


def extract_yawdd() -> None:
    """Saves mouth crops from the YawDD Mirror videos, keeping only wide-open frames from yawning clips."""
    config = load_config()
    raw_dir = config["paths"]["raw_dir"]
    settings, yawn_mar = config["extraction"]["yawdd"], config["mar"]["open_threshold"]
    mirror_dir = raw_dir / "yawn" / "yawdd_videos" / "Mirror"
    output_dir = _fresh_dir(raw_dir / config["datasets"]["yawn"]["yawdd"])
    counts = Counter()

    with FaceTracker(static_image_mode=True) as tracker:
        for folder, gender in YAWDD_FOLDERS.items():
            for video in sorted((mirror_dir / folder).glob("*.avi")):
                parsed = parse_yawdd(video)
                if parsed is None or parsed[2] not in YAWDD_LABELS:
                    print(f"skipping {video.name}")
                    continue
                number, condition, category = parsed
                label = YAWDD_LABELS[category]
                (output_dir / label).mkdir(exist_ok=True)

                capture = cv2.VideoCapture(str(video))
                frame_idx = saved = 0
                while saved < settings["max_frames_per_video"]:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if frame_idx % settings["frame_stride"][category] == 0:
                        regions = tracker.extract(frame)
                        mouth = regions.mouth if regions else None
                        if mouth is not None and (category != "yawning" or mouth.ratio >= yawn_mar):
                            out_path = output_dir / label / f"{gender}{number}_{condition}_{category}_{frame_idx:05d}.jpg"
                            saved += cv2.imwrite(str(out_path), mouth.crop)
                    frame_idx += 1
                capture.release()
                counts[label] += saved
    print(f"YawDD mouth crops: {dict(counts)}")
