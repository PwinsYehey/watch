import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import FIGURES, serif_fonts

serif_fonts()


def placeholder(ax, text):
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0,rounding_size=0.02", fc="#f1f0ec", ec="#8a8984", lw=1.4, ls=(0, (6, 4)), transform=ax.transAxes))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12, color="#52514e", transform=ax.transAxes, linespacing=1.4)
    ax.axis("off")


fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"wspace": 0.06})
placeholder(axes[0], "PLACEHOLDER\n\nLive dashboard at WARNING\n(for example eyes closed or a yawn)\nscripts\\live.ps1, screenshot with Win + Shift + S")
placeholder(axes[1], "PLACEHOLDER\n\nLive dashboard at CRITICAL\n(the sign repeats or lasts)\nscripts\\live.ps1, screenshot with Win + Shift + S")
fig.savefig(FIGURES / "escalation_placeholder.png", dpi=200, bbox_inches="tight", facecolor="white")

fig, ax = plt.subplots(figsize=(11, 5.0))
placeholder(ax, "PLACEHOLDER\n\nFull live monitor: video with face markers and the shoulder line,\ndashboard with alert level, signs and timers, Body, Body drop, Body lean and Head drop rows,\nlighting, sharpness and recent events\n(head and both shoulders in view, about 1 m from the camera)")
fig.savefig(FIGURES / "dashboard_placeholder.png", dpi=200, bbox_inches="tight", facecolor="white")
print("ok")
