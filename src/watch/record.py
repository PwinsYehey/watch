"""Records labelled eye crops on the device for fine-tuning, cued by the buzzer so the eyes can stay closed."""

import re
import time
from pathlib import Path

import cv2

from watch.camera import open_camera
from watch.config import load_config
from watch.face import FaceTracker
from watch.live import Alarm, remove_glare

FOLDERS = {"open": "open", "closed": "closed"}


def clean_name(value: str) -> str:
    """Lowercase letters and digits only, since the first underscore in a filename separates the subject."""
    name = re.sub(r"[^a-z0-9]", "", value.lower())
    if not name:
        raise ValueError(f"{value!r} has no letters or digits to use as a name.")
    return name


def crop_path(root: Path, label: str, subject: str, take: str, side: str, index: int) -> Path:
    return root / FOLDERS[label] / f"{subject}_{take}_{side}_{index:05d}.png"


def _cue(alarm: Alarm, label: str):
    for _ in range(1 if label == "open" else 2):
        alarm.beep(0.15)
        time.sleep(0.15)


def record(camera, subject: str, take: str) -> dict[str, int]:
    """Runs the configured open/closed segments and saves each readable eye crop under data/raw/<record.dir>."""
    config = load_config()
    settings = config["record"]
    root = config["paths"]["raw_dir"] / settings["dir"]
    subject, take = clean_name(subject), clean_name(take)
    if any(root.glob(f"*/{subject}_{take}_*")):
        raise FileExistsError(f"{subject} already has a take named {take}, pick another take name.")
    alarm = Alarm(config["live"].get("buzzer_gpio"))
    capture = open_camera(camera)
    if not capture.isOpened():
        raise ConnectionError(f"Could not open camera {camera}.")
    counts = {label: 0 for label in FOLDERS}
    frames = faces = 0
    try:
        with FaceTracker() as tracker:
            for label, seconds in settings["segments"]:
                print(f"{label} for {seconds}s", flush=True)
                _cue(alarm, label)
                start = time.time()
                while (elapsed := time.time() - start) < seconds:
                    ok, frame = capture.read()
                    if not ok or elapsed < settings["settle_sec"]:
                        continue
                    frames += 1
                    regions = tracker.extract(remove_glare(cv2.flip(frame, 1)))
                    if regions is None:
                        continue
                    faces += 1
                    for side, eye in regions.eyes.items():
                        path = crop_path(root, label, subject, take, side, counts[label])
                        path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(path), eye.crop)
                        counts[label] += 1
            alarm.beep(0.8)
    finally:
        alarm.close()
        capture.release()
    print(f"saved {counts['open']} open and {counts['closed']} closed crops, face found in {faces} of {frames} frames")
    return counts
