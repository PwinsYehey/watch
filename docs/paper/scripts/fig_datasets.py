import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import HELMET_DIR, ROOT, serif_fonts
import random
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RAW = ROOT / "data/raw"
OUT = ROOT / "docs/paper/figures"
random.seed(3)
from matplotlib import font_manager
serif_fonts()

def frame(video, at=0.5):
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * at))
    ok, img = cap.read()
    cap.release()
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def image(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

mirror = RAW / "yawn/yawdd_videos/Mirror/Female_mirror"
nthu = sorted((RAW / "eye/nthu_ddd/train/drowsy").glob("002_noglasses_sleepyCombination_*"))[300]
row1 = [
    (image(nthu), "NTHU-DDD (infrared)"),
    (frame(mirror / "1-FemaleNoGlasses-Normal.avi"), "YawDD, normal"),
    (frame(mirror / "1-FemaleNoGlasses-Yawning.avi", 125 / 387), "YawDD, yawning"),
    (frame(next(mirror.glob("*-FemaleSunGlasses-Normal.avi"))), "YawDD, sunglasses"),
]
m = pd.read_csv(ROOT / "data/splits/eye_manifest.csv")
def pick(ds, label, k):
    rows = m[(m.dataset == ds) & (m.label == label)].sample(k, random_state=7)
    return [image(ROOT / p) for p in rows.filepath]
row2 = [(i, "MRL Eye, open") for i in pick("mrl_eye", "eyes_opened", 2)] + [(i, "MRL Eye, closed") for i in pick("mrl_eye", "eyes_closed", 2)]
row3 = [(i, "CEW, open") for i in pick("cew", "eyes_opened", 2)] + [(i, "CEW, closed") for i in pick("cew", "eyes_closed", 2)]

fig = plt.figure(figsize=(10, 6.4))
grid = fig.add_gridspec(3, 4, height_ratios=[1.35, 1, 1], hspace=0.28, wspace=0.08)
for r, row in enumerate([row1, row2, row3]):
    for c, (img, title) in enumerate(row):
        ax = fig.add_subplot(grid[r, c])
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=12)
        ax.axis("off")
fig.savefig(OUT / "datasets.png", dpi=200, bbox_inches="tight", facecolor="white")
print("ok")
