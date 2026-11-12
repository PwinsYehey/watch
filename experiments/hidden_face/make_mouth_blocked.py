import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from watch.config import PROJECT_ROOT
from watch.face import FaceTracker
from watch.live import remove_glare

OUT = PROJECT_ROOT / "data" / "raw" / "yawn" / "yawdd" / "blocked"


def folds(h, w, rng):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    angle, freq = rng.uniform(-0.6, 0.6), rng.uniform(0.15, 0.45)
    return np.sin((ys + xs * angle) * freq) * rng.uniform(4, 14)


def shade(h, w, rng):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return ((xs - w * rng.uniform(0.2, 0.8)) * rng.uniform(-1, 1) + (ys - h * rng.uniform(0.2, 0.8)) * rng.uniform(-1, 1)) / max(h, w) * rng.uniform(20, 70)


def skin(image):
    border = np.concatenate([image[:3].reshape(-1, 3), image[-3:].reshape(-1, 3), image[:, :3].reshape(-1, 3), image[:, -3:].reshape(-1, 3)])
    return np.median(border, axis=0)


def cover(image, rng):
    h, w = image.shape[:2]
    kind = rng.choice(["mask", "hand", "dark"], p=[0.5, 0.25, 0.25])
    if kind == "mask":
        colour = np.array(rng.choice([[235, 235, 235], [200, 170, 120], [40, 40, 40], [150, 150, 150], [230, 220, 245]]), np.float32)
        layer = colour + (folds(h, w, rng) + shade(h, w, rng))[..., None]
    elif kind == "hand":
        tone = skin(image).astype(np.float32) * rng.uniform(0.85, 1.15)
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        gaps = (np.abs(np.sin((xs * rng.uniform(0.8, 1.2) + ys * rng.uniform(-0.4, 0.4)) * np.pi / (w / rng.uniform(3, 5)))) < 0.12)
        layer = tone + shade(h, w, rng)[..., None] - gaps[..., None] * rng.uniform(25, 60)
    else:
        layer = np.full((h, w, 3), rng.uniform(15, 60), np.float32) + rng.normal(0, rng.uniform(4, 12), (h, w, 1)) + shade(h, w, rng)[..., None] * 0.5
        if rng.random() < 0.4:
            ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
            layer += (np.exp(-((xs - w * rng.uniform(0.2, 0.8)) ** 2) / (2 * (w * 0.12) ** 2)) * rng.uniform(60, 140))[..., None]
    top = int(h * rng.uniform(0.0, 0.3)) if kind == "mask" and rng.random() < 0.6 else 0
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[top:] = 1.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(1.0, h * 0.03))[..., None] if alpha.ndim == 2 else cv2.GaussianBlur(alpha[..., 0], (0, 0), max(1.0, h * 0.03))[..., None]
    out = image.astype(np.float32) * (1 - alpha) + layer * alpha + rng.normal(0, rng.uniform(1, 5), image.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def synthetic(manifest, rng, cap):
    written = 0
    for subject, rows in manifest.groupby("subject_id"):
        k = min(rows.label.value_counts().max() - 1, cap)
        for i, path in enumerate(rows.sample(n=k, random_state=int(rng.integers(1 << 31))).filepath):
            image = cv2.imread(str(PROJECT_ROOT / path))
            if image is None:
                continue
            cv2.imwrite(str(OUT / f"{subject}_blocked_syn_{i:05d}.jpg"), cover(image, rng))
            written += 1
    return written


def person_of(video: Path):
    m = re.match(r"(\d+)-(Fe)?[Mm]ale", video.stem)
    return f"{'female' if m.group(2) else 'male'}{m.group(1)}"


def overlay_crops(videos, stride):
    written = 0
    with FaceTracker() as tracker:
        for video in videos:
            capture, index = cv2.VideoCapture(str(video)), 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if index % stride == 0 and (regions := tracker.extract(remove_glare(frame))) and regions.mouth is not None:
                    cv2.imwrite(str(OUT / f"{person_of(video)}_blocked_mask_{video.stem}_{index:05d}.jpg"), regions.mouth.crop)
                    written += 1
                index += 1
            capture.release()
            print("overlay", video.name, written, flush=True)
    return written


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "yawn_manifest.csv", dtype={"subject_id": str})
    if "synthetic" in sys.argv[1:]:
        print("synthetic", synthetic(manifest, rng, cap=40), flush=True)
    if "overlays" in sys.argv[1:]:
        print("overlays", overlay_crops(sorted(p for p in Path(sys.argv[-1]).glob("*.avi") if ".part" not in p.name), stride=6), flush=True)
