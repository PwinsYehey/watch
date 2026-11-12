import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import HELMET_DIR, ROOT, serif_fonts
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

serif_fonts()
plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": "#8a8984", "axes.linewidth": 0.8,
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#2b2a27"})
TRAIN, VAL, GRID, INK = "#2a78d6", "#eb6834", "#e6e5e0", "#2b2a27"
RUNS = [("eye", "(a) Eye model"), ("yawn_branch", "(b) Yawn branch"), ("blocked_branch", "(c) Blocked branch")]
fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.2), sharex="col", gridspec_kw={"hspace": 0.12, "wspace": 0.28})
for c, (tag, title) in enumerate(RUNS):
    h = pd.read_csv(f"{ROOT}/results/paper_training/{tag}_history.csv")
    epochs = range(1, len(h) + 1)
    best = int(h.val_loss.idxmin()) + 1
    drops = [i + 1 for i in range(1, len(h)) if h.learning_rate[i] < h.learning_rate[i - 1] * 0.9]
    for r, (metric, label) in enumerate([("loss", "Loss"), ("accuracy", "Accuracy")]):
        ax = axes[r][c]
        ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for d in drops:
            ax.axvline(d, color="#b8b7b0", lw=0.7, ls=(0, (1, 2)))
        ax.axvline(best, color=INK, lw=0.9, ls=(0, (4, 2)))
        ax.plot(epochs, h[metric], color=TRAIN, lw=1.8, label="Training")
        ax.plot(epochs, h[f"val_{metric}"], color=VAL, lw=1.8, ls=(0, (5, 1.5)), marker="o", ms=3.2, label="Validation")
        if r == 0:
            ax.set_title(title, fontsize=12)
            ax.text(best + 0.3, ax.get_ylim()[1] * 0.97, f"best epoch {best}", fontsize=9.5, color=INK, va="top")
        if c == 0:
            ax.set_ylabel(label)
        if r == 1:
            ax.set_xlabel("Epoch")
            ax.set_xticks([1, 5, 10, 15] + ([len(h)] if len(h) - 15 >= 3 else []))
handles, labels = axes[0][0].get_legend_handles_labels()
handles += [plt.Line2D([], [], color=INK, lw=0.9, ls=(0, (4, 2))), plt.Line2D([], [], color="#b8b7b0", lw=0.7, ls=(0, (1, 2)))]
labels += ["Best epoch (weights kept)", "Learning rate reduced"]
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=10.5)
fig.savefig(f"{ROOT}/docs/paper/figures/training_curves.png", dpi=200, bbox_inches="tight", facecolor="white")
print("ok")
