import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import HELMET_DIR, ROOT, serif_fonts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

serif_fonts()

INPUT, CONV, HEAD, OUT, FIX = "#e8f1fb", "#ffffff", "#f1eefb", "#eaf7f0", "#fff6e0"
EDGE = {INPUT: "#2a78d6", CONV: "#52514e", HEAD: "#4a3aa7", OUT: "#1baf7a", FIX: "#c98500"}


def box(ax, x, y, w, h, text, colour, size=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=colour, ec=EDGE[colour], lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color="#52514e", lw=1.1, shrinkA=0, shrinkB=0))


fig, ax = plt.subplots(figsize=(13, 6.6))
ax.set_xlim(0, 13); ax.set_ylim(0, 6.6); ax.axis("off")

ax.text(0.05, 6.35, "(a) Eye model: 111,043 parameters", fontsize=13, weight="bold")
y, h = 4.9, 1.0
xs = [0.05, 1.75, 3.45, 5.15, 6.85, 8.55, 10.45]
labels = [("Eye crop\n64×64, grayscale", INPUT), ("Conv 3×3, 32\nBatchNorm\nMaxPool", CONV), ("Conv 3×3, 64\nBatchNorm\nMaxPool", CONV),
          ("Conv 3×3, 128\nBatchNorm\nMaxPool", CONV), ("Global avg. pool\nDropout 0.5", HEAD), ("Dense 128, ReLU\nDropout 0.5", HEAD),
          ("Softmax, 3 classes\nclosed · open · blocked", OUT)]
widths = [1.45, 1.45, 1.45, 1.45, 1.45, 1.65, 2.45]
for x, w, (t, c) in zip(xs, widths, labels):
    box(ax, x, y, w, h, t, c)
for i in range(len(xs) - 1):
    arrow(ax, xs[i] + widths[i] + 0.03, y + h / 2, xs[i + 1] - 0.03, y + h / 2)

ax.text(0.05, 3.85, "(b) Yawn model: two branches, 14,737 parameters", fontsize=13, weight="bold")
top, bottom, h = 2.55, 0.55, 1.0
box(ax, 0.05, 1.55, 1.45, 1.0, "Mouth crop\n64×64, colour", INPUT)
bx = [1.95, 3.65, 5.35, 7.05]
bw = 1.45
branch = [("Conv 3×3, 16\nMaxPool", CONV), ("Conv 3×3, 32\nMaxPool", CONV), ("Global avg. pool\nDense 64, ReLU\nDropout 0.5", HEAD)]
for row, (yy, name, out) in enumerate([(top, "Yawn branch", "Softmax, 2\nno yawn · yawn"), (bottom, "Blocked branch", "Softmax, 3\nP(blocked) only")]):
    for x, (t, c) in zip(bx, branch):
        box(ax, x, yy, bw, h, t, c)
    box(ax, bx[3], yy, bw, h, out, OUT)
    for i in range(3):
        arrow(ax, bx[i] + bw + 0.03, yy + h / 2, bx[i + 1] - 0.03, yy + h / 2)
    ax.text(bx[0], yy + h + 0.08, name, fontsize=11, style="italic")
arrow(ax, 1.53, 2.2, 1.92, top + 0.5)
arrow(ax, 1.53, 1.9, 1.92, bottom + 0.5)
box(ax, 8.95, 1.3, 3.95, 1.5, "Combine (fixed layers)\n$p_b$ = P(blocked)\n[ (1−$p_b$)·P(no yawn),  (1−$p_b$)·P(yawn),  $p_b$ ]", FIX, 10.5)
arrow(ax, bx[3] + bw + 0.03, top + 0.5, 8.92, 2.4)
arrow(ax, bx[3] + bw + 0.03, bottom + 0.5, 8.92, 1.7)
fig.savefig(f"{ROOT}/docs/paper/figures/models.png", dpi=200, bbox_inches="tight", facecolor="white")
print("ok")
