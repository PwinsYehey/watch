import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import HELMET_DIR, ROOT, serif_fonts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

serif_fonts()
FILL = {"io": ("#eaf7f0", "#1baf7a"), "proc": ("#ffffff", "#52514e"), "model": ("#f1eefb", "#4a3aa7"), "sign": ("#fff6e0", "#c98500"),
        "ok": ("#e6f6e6", "#0ca30c"), "warn": ("#fff3d6", "#c98500"), "crit": ("#fde8e8", "#d03b3b"), "hold": ("#f1f0ec", "#8a8984")}
OUT = f"{ROOT}/docs/paper/figures/"


def box(ax, x, y, w, h, text, kind, size=10.5, bold_first=True):
    fc, ec = FILL[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.3))
    lines = text.split("\n")
    ax.text(x + w / 2, y + h / 2, "\n".join(lines), ha="center", va="center", fontsize=size, linespacing=1.3)
    if bold_first:
        pass


def arrow(ax, a, b, text=None, rad=0.0, size=9.5, offset=(0, 0)):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, color="#52514e", lw=1.1, connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if text:
        ax.text((a[0] + b[0]) / 2 + offset[0], (a[1] + b[1]) / 2 + offset[1], text, ha="center", va="center", fontsize=size, color="#2b2a27",
                bbox=dict(fc="white", ec="none", pad=1.2))


fig, ax = plt.subplots(figsize=(13, 4.4))
ax.set_xlim(0, 13.1); ax.set_ylim(0, 4.4); ax.axis("off")
cols = [0.05, 2.05, 4.45, 6.85, 9.35, 11.25]
W = [1.75, 2.15, 2.15, 2.2, 1.65, 1.75]
box(ax, cols[0], 1.6, W[0], 1.3, "Camera frame\nUSB webcam or\nPi Camera Module 3", "io")
box(ax, cols[1], 1.6, W[1], 1.3, "Preprocess\nremove glare\non the face,\nbrighten dark frames", "proc")
box(ax, cols[2], 2.55, W[2], 1.05, "Face Mesh landmarks\n(every frame)", "model")
box(ax, cols[2], 0.85, W[2], 1.05, "Pose landmarks\n(every 3rd frame)", "model")
feats = [("Eyes: eye CNN + EAR", 3.55), ("Mouth: yawn CNN + MAR", 2.75), ("Head: pitch and roll", 1.95), ("Body: shoulders and head", 0.75)]
for t, y in feats:
    box(ax, cols[3], y, W[3], 0.62, t, "proc", 10)
box(ax, cols[4], 1.2, W[4], 2.1, "Sign monitors\nsustained or\nhigh average,\nnods, stillness,\ncheck-in", "sign")
box(ax, cols[5], 1.2, W[5], 2.1, "Escalation\nnormal, warning,\ncritical\n→ alarm, dashboard,\nsession log", "warn")
arrow(ax, (cols[0] + W[0], 2.25), (cols[1], 2.25))
arrow(ax, (cols[1] + W[1], 2.4), (cols[2], 3.05))
arrow(ax, (cols[1] + W[1], 2.1), (cols[2], 1.4))
for t, y in feats[:3]:
    arrow(ax, (cols[2] + W[2], 3.07), (cols[3], y + 0.31))
arrow(ax, (cols[2] + W[2], 1.37), (cols[3], 0.75 + 0.31))
for t, y in feats:
    arrow(ax, (cols[3] + W[3], y + 0.31), (cols[4], 2.25))
arrow(ax, (cols[4] + W[4], 2.25), (cols[5], 2.25))
ax.text(cols[3] + W[3] / 2, 0.35, "each score smoothed by a rolling median of 5", ha="center", fontsize=9.5, style="italic", color="#52514e")
fig.savefig(OUT + "live_pipeline.png", dpi=200, bbox_inches="tight", facecolor="white")

fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.set_xlim(0, 6.4); ax.set_ylim(0.4, 5.2); ax.axis("off")
box(ax, 0.2, 4.0, 2.1, 1.0, "NORMAL", "ok", 12.5)
box(ax, 4.1, 4.0, 2.1, 1.0, "WARNING\n30 s watch", "warn", 12)
box(ax, 4.1, 0.6, 2.1, 1.0, "CRITICAL\nalarm sounds", "crit", 12)
arrow(ax, (2.3, 4.75), (4.1, 4.75), "one sign")
arrow(ax, (4.1, 4.25), (2.3, 4.25), "watch ends\nwith no signs", offset=(0, -0.42))
arrow(ax, (4.75, 4.0), (4.75, 1.6), "the sign repeats,\na second sign appears,\nor the sign lasts", offset=(-1.25, 0))
arrow(ax, (5.6, 1.6), (5.6, 4.0), "5 s with\nno signs", offset=(0.48, 0))
fig.savefig(OUT + "escalation.png", dpi=200, bbox_inches="tight", facecolor="white")


def placeholder(ax, text):
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0,rounding_size=0.02", fc="#f1f0ec", ec="#8a8984", lw=1.4, ls=(0, (6, 4)), transform=ax.transAxes))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12, color="#52514e", transform=ax.transAxes, linespacing=1.4)
    ax.axis("off")


fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"wspace": 0.06})
placeholder(axes[0], "PLACEHOLDER\n\nPilot in helmet, face visible\nLive dashboard on the Raspberry Pi: NORMAL\n(to be captured on the device)")
placeholder(axes[1], "PLACEHOLDER\n\nPilot with tinted visor and oxygen mask\nDashboard: FACE COVERED, body tracked\n(to be captured on the device)")
fig.savefig(OUT + "pilot_placeholder.png", dpi=200, bbox_inches="tight", facecolor="white")
print("ok")
