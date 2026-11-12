import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import HELMET_DIR, ROOT, serif_fonts
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from watch.face import FaceTracker

serif_fonts()

MIRROR = ROOT / "data/raw/yawn/yawdd_videos/Mirror"
OVERLAY = ROOT / "results/hidden_face/overlay"
OUT = ROOT / "docs/paper/figures"
FEMALE, MALE = MIRROR / "Female_mirror", MIRROR / "Male_mirror Avi Videos"


def read(video, index):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, img = cap.read()
    cap.release()
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def mouth_frames(video, tracker, stride=5):
    cap, i, out = cv2.VideoCapture(str(video)), 0, []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            r = tracker.extract(f)
            if r and r.mouth is not None:
                out.append((r.mouth.ratio, i))
        i += 1
    cap.release()
    return sorted(out)


def pick(video, tracker, kind):
    frames = mouth_frames(video, tracker)
    if kind == "Yawning":
        return frames[-1][1]
    if kind == "Talking":
        return frames[int(len(frames) * 0.8)][1]
    return frames[len(frames) // 2][1]


def crop(img, box=(0.12, 0.0, 0.88, 0.85)):
    h, w = img.shape[:2]
    return img[int(box[1] * h):int(box[3] * h), int(box[0] * w):int(box[2] * w)]


columns = [("No glasses", FEMALE, "1-FemaleNoGlasses"), ("Glasses", MALE, "11-MaleGlasses"), ("Sunglasses", FEMALE, "17-FemaleSunGlasses")]
rows = ["Normal", "Talking", "Yawning"]
with FaceTracker() as tracker:
    grid = [[None] * 3 for _ in rows]
    for c, (_, folder, stem) in enumerate(columns):
        for r, kind in enumerate(rows):
            video = folder / f"{stem}-{kind}.avi"
            grid[r][c] = read(video, pick(video, tracker, kind))
    source = MALE / "1-MaleNoGlasses-Yawning.avi"
    yawn_index = pick(source, tracker, "Yawning")

fig, axes = plt.subplots(3, 3, figsize=(9, 7.2), gridspec_kw={"hspace": 0.05, "wspace": 0.04})
for r, kind in enumerate(rows):
    for c, (title, _, _) in enumerate(columns):
        ax = axes[r][c]
        ax.imshow(grid[r][c])
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(title, fontsize=13)
        if c == 0:
            ax.set_ylabel(kind, fontsize=13)
fig.savefig(OUT / "yawdd.png", dpi=200, bbox_inches="tight", facecolor="white")

panels = [("Original", source), ("Dark visor", OVERLAY / "dark" / source.name), ("Mirror visor", OVERLAY / "mirror" / source.name), ("Dark visor and mask", OVERLAY / "dark_mask" / source.name)]
fig, axes = plt.subplots(1, 4, figsize=(11, 2.9), gridspec_kw={"wspace": 0.04})
for ax, (title, video) in zip(axes, panels):
    ax.imshow(crop(read(video, yawn_index), (0.2, 0.0, 0.8, 0.8)))
    ax.set_title(title, fontsize=13)
    ax.axis("off")
fig.savefig(OUT / "overlays.png", dpi=200, bbox_inches="tight", facecolor="white")
print("ok", yawn_index)
