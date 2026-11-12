import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import HELMET_DIR, ROOT, serif_fonts
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

serif_fonts()
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#8a8984", "xtick.color": "#2b2a27", "ytick.color": "#2b2a27"})
OUT = f"{ROOT}/docs/paper/figures/"
ev = json.load(open(f"{ROOT}/results/paper_training/eval_rule.json"))
blues = LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#86b6ef", "#2a78d6", "#184f95"])
NAMES = {"eye": ["eyes_closed", "eyes_opened", "blocked"], "yawn": ["no_yawn", "yawn", "blocked"]}
LABEL = {"eyes_closed": "Closed", "eyes_opened": "Open", "blocked": "Blocked", "no_yawn": "No yawn", "yawn": "Yawn"}
fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), gridspec_kw={"wspace": 0.45})
for ax, (name, title) in zip(axes, [("eye", "(a) Eye model"), ("yawn", "(b) Yawn model")]):
    cls = NAMES[name]
    cm = np.array([[ev[name]["confusion"][p][t] for p in cls] for t in cls], float)
    share = cm / cm.sum(1, keepdims=True)
    ax.imshow(share, cmap=blues, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{share[i, j] * 100:.1f}%\n({int(cm[i, j]):,})", ha="center", va="center", fontsize=10.5, color="white" if share[i, j] > 0.55 else "#2b2a27")
    ax.set_xticks(range(3), [LABEL[c] for c in cls]); ax.set_yticks(range(3), [LABEL[c] for c in cls])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title, fontsize=12.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
fig.savefig(OUT + "confusion.png", dpi=200, bbox_inches="tight", facecolor="white")

frames = pd.read_csv(f"{ROOT}/results/hidden_face/helmet/frames_current.csv")
people = pd.read_csv(f"{ROOT}/results/hidden_face/helmet_sessions.csv")
people = people[people.label_check != "WRONG"]
f = frames[frames.session.isin(people.session)].copy()
f["person"] = f.session.map(dict(zip(people.session, people.person))).str[:3]
f["label"] = f.session.map(dict(zip(people.session, people.label_given)))
doubt = set(people[people.label_check == "check"].person.str[:3])
eyes = pd.concat([f[["person", "label", s]].rename(columns={s: "e"}) for s in ("left_label", "right_label")]).dropna()
closed = eyes.assign(c=eyes.e == "eyes_closed").groupby(["person", "label"]).c.mean().unstack() * 100
closed.to_csv(f"{ROOT}/results/paper_training/helmet_closed_by_rider.csv")
order = list(closed.index)
fig, ax = plt.subplots(figsize=(5.2, 3.9))
ALERT, DROWSY = "#2a78d6", "#eb6834"
for i, p in enumerate(order):
    a, d = closed.loc[p, "alert"], closed.loc[p, "drowsy"]
    if not np.isnan(a):
        ax.plot([a, d], [i, i], color="#c3c2b7", lw=2, zorder=1)
        ax.scatter(a, i, s=46, color=ALERT, zorder=3, edgecolor="white", linewidth=1.5)
    ax.scatter(d, i, s=46, color=DROWSY, zorder=3, marker="D", edgecolor="white", linewidth=1.5)
labels = [f"{p}*" if p in doubt else p for p in order]
labels = [l + " (no face when alert)" if closed.loc[p, "alert"] != closed.loc[p, "alert"] else l for l, p in zip(labels, order)]
ax.set_yticks(range(len(order)), labels); ax.invert_yaxis()
ax.set_xlabel("Eye readings classed closed (%)"); ax.set_xlim(-3, 100)
ax.grid(axis="x", color="#e6e5e0", lw=0.7); ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.scatter([], [], s=46, color=ALERT, label="Alert session"); ax.scatter([], [], s=46, color=DROWSY, marker="D", label="Drowsy session")
ax.legend(loc="lower right", frameon=False, fontsize=10.5)
fig.savefig(OUT + "helmet_closed.png", dpi=200, bbox_inches="tight", facecolor="white")
print(closed.round(1))
