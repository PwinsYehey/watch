import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

FPS = 29.97
runs = Path(sys.argv[1])
d = pd.concat([pd.read_csv(p) for p in sorted(runs.glob("*.csv"))], ignore_index=True)
parts = d.video.str.extract(r"(?P<subject>\d+-(?:Fe)?[Mm]ale)(?P<glasses>NoGlasses|SunGlasses|Glasses)-(?P<behaviour>\w+)\.avi")
d = pd.concat([d, parts], axis=1)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 300)


def per_video(g):
    warn = g.level >= 1
    reasons = g.reason.fillna("")
    return pd.Series(
        {
            "frames": len(g),
            "face_%": 100 * (g.face == 1).mean(),
            "eye_cue_%": 100 * g.eyes_score.notna().mean(),
            "eye_closed_mean": g.eyes_score.mean(),
            "eye_conf_mean": g[["left_conf", "right_conf"]].mean(axis=1).mean(),
            "mouth_cue_%": 100 * g.mouth_score.notna().mean(),
            "yawn_detected": int((g.mouth_state == "not_alert").any()),
            "roll_flip_%": 100 * (g.roll > 90).mean(),
            "warn_%": 100 * warn.mean(),
            "warn_eyes_%": 100 * (warn & reasons.str.contains("eyes")).mean(),
            "warn_head_%": 100 * (warn & reasons.str.contains("head")).mean(),
            "warn_yawn_%": 100 * (warn & reasons.str.contains("yawn")).mean(),
            "critical_%": 100 * (g.level == 2).mean(),
        }
    )


def blinks_truth(g):
    closed = g[["left_cnn_closed", "right_cnn_closed"]].max(axis=1).fillna(0).to_numpy() > 0.5
    events, start = [], None
    for i, c in enumerate(closed):
        if c and start is None:
            start = i
        if not c and start is not None:
            if i - start <= 0.6 * FPS:
                events.append((start + i) / 2)
            start = None
    return np.array(events)


def flicker(g, k):
    sig = g[["left_patch_mean", "right_patch_mean"]].mean(axis=1).interpolate(limit_direction="both").to_numpy()
    if len(sig) == 0 or np.isnan(sig).all():
        return np.array([])
    hp = sig - median_filter(sig, size=int(0.7 * FPS))
    mad = np.median(np.abs(hp - np.median(hp))) + 1e-6
    peaks, _ = find_peaks(np.abs(hp), height=k * mad, distance=int(0.25 * FPS), width=(1, int(0.4 * FPS)))
    return peaks.astype(float)


def match(truth, found, tol=0.25 * FPS):
    hit = sum(any(abs(t - f) <= tol for f in found) for t in truth)
    good = sum(any(abs(t - f) <= tol for t in truth) for f in found)
    return hit, good


if __name__ == "__main__":
    videos = d.groupby(["glasses", "behaviour", "subject"]).apply(per_video, include_groups=False).reset_index()
    print("== Per condition (mean over videos) ==")
    print(videos.groupby(["glasses", "behaviour"]).mean(numeric_only=True).round(1))
    print("\n== Per video ==")
    print(videos.round(1).to_string())

    print("\n== Blink flicker vs eye model on clear eyes (NoGlasses, Glasses) ==")
    for k in (4, 6, 8):
        tot = np.zeros(4)
        for _, g in d[d.glasses != "SunGlasses"].groupby("video"):
            g = g.reset_index(drop=True)
            truth, found = blinks_truth(g), flicker(g, k)
            tot += (len(truth), len(found), *match(truth, found))
        print(f"k={k}: model blinks {tot[0]:.0f}, flicker events {tot[1]:.0f}, recall {tot[2] / max(tot[0], 1):.0%}, precision {tot[3] / max(tot[1], 1):.0%}")

    print("\n== Blinks per minute by person (flicker k=6 / eye model) ==")
    rows = []
    for (subject, glasses), g in d.groupby(["subject", "glasses"]):
        minutes = len(g) / FPS / 60
        f = sum(len(flicker(v.reset_index(drop=True), 6)) for _, v in g.groupby("video"))
        t = sum(len(blinks_truth(v.reset_index(drop=True))) for _, v in g.groupby("video"))
        rows.append({"subject": subject, "glasses": glasses, "flicker": round(f / minutes, 1), "model": round(t / minutes, 1)})
    print(pd.DataFrame(rows).pivot(index="subject", columns="glasses", values=["flicker", "model"]).to_string())
