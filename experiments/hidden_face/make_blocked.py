import re
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from watch.config import PROJECT_ROOT
from watch.face import FaceTracker
from watch.live import remove_glare
from watch.model import Classifier

RAW = PROJECT_ROOT / "data" / "raw" / "eye"
YAWDD = PROJECT_ROOT / "data" / "raw" / "yawn" / "yawdd_videos"


def cover(image, rng):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) if image.ndim == 3 else image.astype(np.float32)
    h, w = gray.shape
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(gray, (0, 0), rng.uniform(0.5, 2.0)) * rng.uniform(0.04, 0.22) + rng.uniform(0, 25)
    else:
        noise = cv2.resize(rng.uniform(0, 255, (4, 4)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        ys, xs = np.mgrid[0:h, 0:w]
        gradient = (xs * rng.uniform(-1, 1) + ys * rng.uniform(-1, 1)) / max(w, h) * 120 + rng.uniform(60, 200)
        reflection = 0.5 * noise + 0.5 * gradient
        out = gray * rng.uniform(0.05, 0.2) + reflection * rng.uniform(0.6, 0.9)
    if rng.random() < 0.25:
        ys, xs = np.mgrid[0:h, 0:w]
        band = (np.abs((xs * rng.uniform(0.3, 1.5) + ys) % max(w, h) - max(w, h) / 2) < max(w, h) * rng.uniform(0.06, 0.2)).astype(np.float32)
        band = cv2.GaussianBlur(band, (0, 0), max(w, h) * 0.06) * rng.uniform(0.3, 0.7)
        out = out * (1 - band) + 230 * band
    out = out + rng.normal(0, rng.uniform(1, 6), out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def synthetic_from_manifest(manifest, rng, per_subject_cap):
    written = []
    for (dataset, subject), rows in manifest[manifest.dataset.isin(["mrl_eye", "cew"])].groupby(["dataset", "subject_id"]):
        k = min(rows.label.value_counts().max() - 1, per_subject_cap if dataset == "mrl_eye" else 1)
        if k <= 0 or (dataset == "cew" and rng.random() < 0.5):
            continue
        for i, path in enumerate(rows.sample(n=k, random_state=int(rng.integers(1 << 31))).filepath):
            image = cv2.imread(str(PROJECT_ROOT / path))
            if image is None:
                continue
            if dataset == "mrl_eye":
                out = RAW / "mrl_eye" / "blocked" / f"{subject}_b{i:05d}_0_0_b_0_0_01.png"
            else:
                out = RAW / "cew" / "blockedEyes" / f"{subject}_L.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), cover(image, rng))
            written.append(out)
    return written


def person_of(video: Path):
    m = re.match(r"(\d+)-(Fe)?[Mm]ale", video.stem)
    return f"{'female' if m.group(2) else 'male'}{m.group(1)}"


def video_crops(video, stride, tracker):
    capture = cv2.VideoCapture(str(video))
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            regions = tracker.extract(remove_glare(frame))
            if regions:
                yield index, regions
        index += 1
    capture.release()


def yawdd_crops(eye_model, overlay_dir, parts):
    out_root = RAW / "yawdd_eye"
    counts = {"blocked": 0, "open": 0, "closed": 0}
    videos = sorted(YAWDD.rglob("*.avi"))
    sunglasses = [v for v in videos if "SunGlasses" in v.name] if "sunglasses" in parts else []
    overlays = sorted(p for p in overlay_dir.glob("*/*.avi") if ".part" not in p.name) if "overlays" in parts else []
    clear = [v for v in videos if "SunGlasses" not in v.name] if "clear" in parts else []
    with FaceTracker() as tracker:
        for video, stride in [(v, 6) for v in sunglasses] + [(v, 10) for v in overlays]:
            tag = video.parent.name if video in overlays else "sun"
            for index, regions in video_crops(video, stride, tracker):
                for side, eye in regions.eyes.items():
                    path = out_root / "blocked" / f"{person_of(video)}_{tag}_{video.stem}_{index:05d}_{side}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(path), eye.crop)
                    counts["blocked"] += 1
            print("blocked", video.name, counts, flush=True)
        for video in clear:
            samples = list(video_crops(video, 15, tracker))
            ears = [eye.ratio for _, r in samples for eye in r.eyes.values()]
            if len(ears) < 10:
                continue
            baseline = statistics.median(ears)
            for index, regions in samples:
                if not regions.eyes:
                    continue
                predictions = eye_model.predict_many([e.crop for e in regions.eyes.values()])
                for (side, eye), (_, conf, closed) in zip(regions.eyes.items(), predictions):
                    if closed < 0.05 and conf > 0.95 and eye.ratio > baseline - 0.03:
                        label = "open"
                    elif closed > 0.95 and eye.ratio <= baseline - 0.08:
                        label = "closed"
                    else:
                        continue
                    path = out_root / label / f"{person_of(video)}_clear_{video.stem}_{index:05d}_{side}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(path), eye.crop)
                    counts[label] += 1
            print("clear", video.name, counts, flush=True)
    return counts


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "eye_manifest.csv", dtype={"subject_id": str})
    if "synthetic" in sys.argv[1:]:
        print("synthetic", len(synthetic_from_manifest(manifest, rng, per_subject_cap=400)), flush=True)
    parts = {"sunglasses", "overlays", "clear"} & set(sys.argv[1:])
    if parts:
        print("yawdd", yawdd_crops(Classifier("eye"), Path(sys.argv[-1]), parts), flush=True)
