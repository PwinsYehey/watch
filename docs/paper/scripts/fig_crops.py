import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import HELMET_DIR, ROOT, serif_fonts
from pathlib import Path
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

serif_fonts()
OUT = ROOT / "docs/paper/figures"


eye = pd.read_csv(ROOT / "data/splits/eye_manifest.csv")
yawn = pd.read_csv(ROOT / "data/splits/yawn_manifest.csv")


def sample(df, mask, k=1, seed=5):
    return [cv2.cvtColor(cv2.imread(str(ROOT / p)), cv2.COLOR_BGR2RGB) for p in df[mask].sample(k, random_state=seed).filepath]


e = eye.filepath
eye_cols = [
    ("Open", sample(eye, (eye.label == "eyes_opened") & (eye.dataset == "nthu_eye"), 1) + sample(eye, (eye.label == "eyes_opened") & (eye.dataset == "yawdd_eye"), 1)),
    ("Closed", sample(eye, (eye.label == "eyes_closed") & (eye.dataset == "nthu_eye"), 1) + sample(eye, (eye.label == "eyes_closed") & (eye.dataset == "yawdd_eye"), 1)),
    ("Blocked: tint", sample(eye, (eye.label == "blocked") & (eye.dataset == "mrl_eye"), 2)),
    ("Blocked: sunglasses", sample(eye, e.str.contains("_sun_"), 2)),
    ("Blocked: visor", sample(eye, e.str.contains("_mirror_"), 1) + sample(eye, e.str.contains("_dark_mask_|_dark_"), 1, 9)),
]
y = yawn.filepath
mouth_cols = [
    ("No yawn", sample(yawn, (yawn.label == "no_yawn") & y.str.contains("_normal_"), 1) + sample(yawn, (yawn.label == "no_yawn") & y.str.contains("_talking_"), 1)),
    ("Yawn", sample(yawn, yawn.label == "yawn", 2)),
    ("Blocked: synthetic", sample(yawn, y.str.contains("_blocked_syn_"), 2)),
    ("Blocked: cover", sample(yawn, y.str.contains("_blocked_cover_"), 2, 23)),
    ("Blocked: mask", sample(yawn, y.str.contains("_blocked_mask_"), 2)),
]

fig = plt.figure(figsize=(11, 8.6))
grid = fig.add_gridspec(5, 5, height_ratios=[1, 1, 0.12, 0.8, 0.8], hspace=0.1, wspace=0.06)
for block, cols in enumerate([eye_cols, mouth_cols]):
    for c, (title, images) in enumerate(cols):
        for r, img in enumerate(images):
            ax = fig.add_subplot(grid[block * 3 + r, c])
            ax.imshow(cv2.resize(img, (128, 128) if block == 0 else (160, 112)))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=13)
            if c == 0 and r == 0:
                ax.set_ylabel("Eye crops" if block == 0 else "Mouth crops", fontsize=13)
fig.savefig(OUT / "crops.png", dpi=170, bbox_inches="tight", facecolor="white")
print("ok")
