import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from watch.config import PROJECT_ROOT
from watch.face import MOUTH_AXIS, MOUTH_IDS, FaceTracker, aligned_crop
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


CHEEKS, CHIN, FACE_EDGES = (205, 425), 152, (234, 454)


def take(frame, x0, y0, pw, ph):
    h, w = frame.shape[:2]
    x0, y0 = int(x0), int(y0)
    if x0 < 0 or y0 < 0 or x0 + pw > w or y0 + ph > h:
        return None
    return frame[y0:y0 + ph, x0:x0 + pw].astype(np.float32)


def cover_mouth(frame, landmarks, rng, pool):
    h, w = frame.shape[:2]
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in MOUTH_IDS])
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    bw, bh = (x1 - x0) * rng.uniform(1.6, 2.2), (y1 - y0) * 2.6 + (x1 - x0) * rng.uniform(0.25, 0.5)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 + bh * rng.uniform(-0.05, 0.1)
    box = [int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2)]
    if min(box[0], box[1]) < 0 or box[2] > w or box[3] > h:
        return None, None
    pw, ph = box[2] - box[0], box[3] - box[1]
    kind = rng.choice(["hand", "cloth", "background", "object"], p=[0.35, 0.2, 0.2, 0.25])
    if kind == "hand":
        cheek = landmarks[CHEEKS[int(rng.integers(2))]]
        sw = (x1 - x0) * rng.uniform(0.35, 0.55)
        src = take(frame, cheek.x * w - sw / 2, cheek.y * h - sw / 3, int(sw), int(sw * 2 / 3))
        patch = None if src is None else cv2.GaussianBlur(cv2.resize(src, (pw, ph)), (0, 0), max(1.0, pw * 0.015))
        if patch is not None:
            ys, xs = np.mgrid[0:ph, 0:pw].astype(np.float32)
            gaps = (np.abs(np.sin((xs * rng.uniform(0.9, 1.1) + ys * rng.uniform(-0.5, 0.5)) * np.pi / (pw / rng.uniform(3.5, 5)))) < 0.18).astype(np.float32)
            patch = patch - cv2.GaussianBlur(gaps, (0, 0), max(1.0, pw * 0.02))[..., None] * rng.uniform(8, 25)
    elif kind == "cloth":
        chin = landmarks[CHIN]
        patch = take(frame, cx - pw / 2 + rng.uniform(-0.3, 0.3) * pw, chin.y * h + ph * rng.uniform(0.6, 1.4), pw, ph)
    elif kind == "background":
        edge = landmarks[FACE_EDGES[int(rng.integers(2))]]
        side = -1 if edge.x * w < cx else 1
        patch = take(frame, edge.x * w + side * pw * rng.uniform(0.3, 0.9) - (pw if side < 0 else 0), cy - ph / 2 + rng.uniform(-0.5, 0.5) * ph, pw, ph)
    else:
        if not pool:
            return None, None
        other = pool[int(rng.integers(len(pool)))]
        oh, ow = other.shape[:2]
        patch = take(other, rng.uniform(0, max(1, ow - pw)), rng.uniform(0, max(1, oh - ph)), pw, ph)
    if patch is None:
        return None, None
    patch = patch * rng.uniform(0.75, 1.2) + shade(ph, pw, rng)[..., None] * 0.4
    mask = cv2.GaussianBlur(np.ones((ph, pw), np.float32), (0, 0), max(1.0, pw * 0.04))[..., None]
    out = frame.astype(np.float32).copy()
    region = out[box[1]:box[3], box[0]:box[2]]
    out[box[1]:box[3], box[0]:box[2]] = region * (1 - mask) + patch * mask
    cropped = aligned_crop(np.clip(out, 0, 255).astype(np.uint8), landmarks, MOUTH_IDS, MOUTH_AXIS, 0.3, False)
    return (cropped[0] if cropped is not None else None), kind


def covers(videos, per_video, rng):
    written, pool = 0, []
    with FaceTracker() as tracker:
        for video in videos:
            capture = cv2.VideoCapture(str(video))
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            for index in np.linspace(30, max(31, total - 1), per_video * 2).astype(int):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if not ok or not (regions := tracker.extract(remove_glare(frame))):
                    continue
                crop, kind = cover_mouth(frame, regions.landmarks, rng, pool)
                if crop is not None:
                    cv2.imwrite(str(OUT / f"{person_of(video)}_blocked_cover_{kind}_{video.stem}_{index:05d}.jpg"), crop)
                    written += 1
                if len([f for f in OUT.glob(f"{person_of(video)}_blocked_cover_*_{video.stem}_*")]) >= per_video:
                    break
            if ok:
                pool = (pool + [frame])[-20:]
            capture.release()
            print("covers", video.name, written, flush=True)
    return written


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "yawn_manifest.csv", dtype={"subject_id": str})
    if "synthetic" in sys.argv[1:]:
        print("synthetic", synthetic(manifest, rng, cap=40), flush=True)
    if "covers" in sys.argv[1:]:
        clear = sorted(p for p in (PROJECT_ROOT / "data" / "raw" / "yawn" / "yawdd_videos").rglob("*.avi") if "SunGlasses" not in p.name)
        print("covers", covers(clear, per_video=12, rng=rng), flush=True)
    if "overlays" in sys.argv[1:]:
        print("overlays", overlay_crops(sorted(p for p in Path(sys.argv[-1]).glob("*.avi") if ".part" not in p.name), stride=6), flush=True)
