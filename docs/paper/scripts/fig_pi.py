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
FILL = {"io": ("#eaf7f0", "#1baf7a"), "proc": ("#ffffff", "#52514e"), "model": ("#f1eefb", "#4a3aa7"), "warn": ("#fff3d6", "#c98500"), "user": ("#e8f1fb", "#2a78d6")}
OUT = f"{ROOT}/docs/paper/figures/"


def box(ax, x, y, w, h, text, kind, size=11):
    fc, ec = FILL[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, linespacing=1.3)


def link(ax, a, b, text, offset=(0, 0.18), both=False, style="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="<|-|>" if both else "-|>", mutation_scale=11, color="#52514e", lw=1.2, ls=style, shrinkA=2, shrinkB=2))
    ax.text((a[0] + b[0]) / 2 + offset[0], (a[1] + b[1]) / 2 + offset[1], text, ha="center", va="center", fontsize=9.5, color="#2b2a27", bbox=dict(fc="white", ec="none", pad=1))


fig, ax = plt.subplots(figsize=(6.6, 5.4))
ax.set_xlim(0, 6.6); ax.set_ylim(0, 5.4); ax.axis("off")
box(ax, 2.1, 2.0, 2.4, 1.6, "Raspberry Pi 4 Model B\n8 GB, 32 GB microSD\nheatsink\nwatch.service at boot", "model")
box(ax, 0.1, 4.2, 2.2, 1.0, "Camera Module 3 NoIR\nimx708_noir, 640×480", "io", 10.5)
box(ax, 4.3, 4.2, 2.2, 1.0, "5 V 3 A USB-C\npower supply", "proc", 10.5)
box(ax, 0.1, 0.2, 2.2, 1.0, "Buzzer module\nGPIO 4, active low", "warn", 10.5)
box(ax, 4.3, 0.2, 2.2, 1.0, "Laptop browser\nvideo, dashboard,\nbuttons", "user", 10.5)
link(ax, (1.2, 4.2), (2.5, 3.6), "ribbon to\nCAMERA port", offset=(-0.45, -0.05))
link(ax, (5.4, 4.2), (4.1, 3.6), "USB-C", offset=(0.35, 0.0))
link(ax, (2.5, 2.0), (1.2, 1.2), "alarm while\ncritical", offset=(-0.45, 0.05))
link(ax, (4.1, 2.0), (5.4, 1.2), "Wi-Fi or Ethernet\nport 8766", offset=(0.55, 0.12), both=True)
fig.savefig(OUT + "pi_hardware.png", dpi=200, bbox_inches="tight", facecolor="white")


def placeholder(ax, text):
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0,rounding_size=0.02", fc="#f1f0ec", ec="#8a8984", lw=1.4, ls=(0, (6, 4)), transform=ax.transAxes))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12, color="#52514e", transform=ax.transAxes, linespacing=1.4)
    ax.axis("off")


fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"wspace": 0.06})
placeholder(axes[0], "PLACEHOLDER\n\nPhoto of the Raspberry Pi setup:\nPi 4, Camera Module 3 NoIR, buzzer\n(to be captured on the device)")
placeholder(axes[1], "PLACEHOLDER\n\nSession report exported on the Pi:\nalertness level over time\n(Export report on the device)")
fig.savefig(OUT + "pi_placeholder.png", dpi=200, bbox_inches="tight", facecolor="white")
print("ok")
