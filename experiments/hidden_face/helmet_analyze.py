import sys

import numpy as np
import pandas as pd

frames = pd.read_csv(sys.argv[1])
people = pd.read_csv(sys.argv[2])
people = people[people.label_check != "WRONG"]
person_of = dict(zip(people.session, people.person))
label_of = dict(zip(people.session, people.label_given))
doubt_of = dict(zip(people.session, people.label_check))
d = frames[frames.session.isin(person_of)].copy()
d["person"] = d.session.map(person_of)
d["label"] = d.session.map(label_of)
d["doubtful"] = d.session.map(doubt_of) == "check"
d["closed"] = d[["left_closed", "right_closed"]].mean(axis=1)
d["ear"] = d[["left_ear", "right_ear"]].mean(axis=1)
d["roll_fixed"] = np.where(d.roll > 90, 180 - d.roll, d.roll)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)

print("== Can WATCH see it? (share of photos) ==")
seen = d.groupby("label").agg(
    photos=("face", "size"),
    face=("face", "mean"),
    eyes=("left_closed", lambda s: s.notna().mean()),
    mouth=("mouth_open", lambda s: s.notna().mean()),
    head_pose=("pitch", lambda s: s.notna().mean()),
    body_pose=("pose", "mean"),
    roll_flipped=("roll", lambda s: (s > 90).mean()),
)
print((seen * [1, 100, 100, 100, 100, 100, 100]).round(1))


def person_row(g):
    ear_base = g[g.label == "alert"].ear.median()
    out = {}
    for label, s in g.groupby("label"):
        out[f"{label}_closed_%"] = 100 * (s.closed > 0.5).mean()
        out[f"{label}_ear_low_%"] = 100 * (s.ear <= ear_base - 0.1).mean()
        out[f"{label}_yawn_%"] = 100 * (s.mouth_open > 0.5).mean()
        out[f"{label}_face_%"] = 100 * s.face.mean()
    out["doubtful"] = bool(g.doubtful.any())
    return pd.Series(out)


per = d.groupby("person").apply(person_row, include_groups=False)
per["eyes_say_drowsier"] = per["drowsy_closed_%"] > per["alert_closed_%"]
per["ear_says_drowsier"] = per["drowsy_ear_low_%"] > per["alert_ear_low_%"]
print("\n== Per person: alert vs drowsy session ==")
print(per.round(1).to_string())
clean = per[~per.doubtful]
print(f"\nEye model says drowsy session is drowsier: {clean.eyes_say_drowsier.sum()} of {len(clean)} people (doubtful labels left out)")
print(f"Eye openness (EAR) says drowsy session is drowsier: {clean.ear_says_drowsier.sum()} of {len(clean)} people")

print("\n== Per session ==")
sess = d.groupby(["person", "label", "session"]).agg(
    photos=("face", "size"),
    face_pct=("face", lambda s: 100 * s.mean()),
    closed_pct=("closed", lambda s: 100 * (s > 0.5).mean()),
    eye_conf=("left_conf", "mean"),
    yawn_pct=("mouth_open", lambda s: 100 * (s > 0.5).mean()),
    pose_pct=("pose", lambda s: 100 * s.mean()),
)
print(sess.round(1).to_string())
