import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import HELMET_DIR, ROOT, serif_fonts
import re
from pathlib import Path
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

serif_fonts()

ROOT = Path(str(HELMET_DIR))
people = pd.read_csv(f"{ROOT}/results/hidden_face/helmet_sessions.csv")
reads = pd.read_csv(f"{ROOT}/results/hidden_face/helmet/frames_current.csv")
WANT = {"alert": "eyes_opened", "drowsy": "eyes_closed"}


def chosen(session, label, count):
    r = reads[(reads.session == session) & (reads.left_label == WANT[label]) & (reads.right_label == WANT[label])]
    if r.empty:
        return count // 2
    return int(r["index"].iloc[len(r) // 2])
files = {}
for p in ROOT.rglob("*.jpg"):
    m = re.match(r"(.+?)_(\d+)$", p.stem)
    if m:
        files.setdefault(m.group(1), []).append((int(m.group(2)), p))

riders = [
    ("P01_rxr_white_glasses", "Full-face, glasses"),
    ("P02_silver_balaclava", "Full-face, neck gaiter"),
    ("P03_greenscreen_black", "Full-face"),
    ("P05_clear_visor_blue", "Clear visor down"),
    ("P07_green_blue_jordan", "Full-face, side view"),
    ("P08_purple_open", "Open-face, visor up"),
]
fig, axes = plt.subplots(2, len(riders), figsize=(12, 5.6), gridspec_kw={"hspace": 0.04, "wspace": 0.04})
for c, (person, title) in enumerate(riders):
    for r, label in enumerate(["alert", "drowsy"]):
        session = people[(people.person == person) & (people.label_given == label)].session.iloc[0]
        frames = sorted(files[session])
        index = dict(frames)
        img = cv2.cvtColor(cv2.imread(str(index.get(chosen(session, label, len(frames)), frames[len(frames) // 2][1]))), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        ch = min(h, int(w * 4 / 3))
        img = img[: ch]
        ax = axes[r][c]
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(title, fontsize=13)
        if c == 0:
            ax.set_ylabel(label.capitalize(), fontsize=13)
fig.savefig(f"{ROOT}/docs/paper/figures/helmet.png", dpi=170, bbox_inches="tight", facecolor="white")
print("ok")
