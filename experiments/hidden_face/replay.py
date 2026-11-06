import csv
import statistics
import sys
from pathlib import Path

import cv2

from watch.alertness import AlertnessMonitor, EscalationMonitor, RollingMedian
from watch.body import BodyPose, BodyTracker
from watch.config import load_config
from watch.face import LEFT_EYE_IDS, RIGHT_EYE_IDS, FaceTracker
from watch.live import LiveMonitor, face_box, glare_fraction, remove_glare
from watch.model import Classifier


def eye_patch_stats(gray, landmarks, ids):
    h, w = gray.shape
    xs = [landmarks[i].x * w for i in ids]
    ys = [landmarks[i].y * h for i in ids]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half = max(max(xs) - min(xs), 1) * 0.6
    x0, x1 = int(max(cx - half, 0)), int(min(cx + half, w))
    y0, y1 = int(max(cy - half * 0.6, 0)), int(min(cy + half * 0.6, h))
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return None, None, None
    return float(patch.mean()), float(patch.std()), (x0, y0, x1, y1)


def monitor_for(config, eye_model, yawn_model):
    m = object.__new__(LiveMonitor)
    m.config, m.settings = config, config["live"]
    m.eye_model, m.yawn_model = eye_model, yawn_model
    m.ear, m.head = config["ear"], config["head_pose"]
    return m


def calibrate(m, config, frames, fps):
    calib = config["calibration"]
    n_cal = int(calib["duration_sec"] * fps)
    ears, pitches, rolls, poses = [], [], [], []
    body_tracker = BodyTracker(config["body"])
    with FaceTracker() as tracker:
        for frame in frames[:n_cal]:
            if pose := body_tracker.update(frame):
                poses.append(pose)
            regions = tracker.extract(remove_glare(frame))
            if regions:
                ears += [eye.ratio for eye in regions.eyes.values()]
                if regions.pitch is not None:
                    pitches.append(regions.pitch)
                    rolls.append(regions.roll)
    if len(ears) >= 10:
        baseline = statistics.median(ears)
        m.ear = {"open_threshold": baseline, "closed_threshold": baseline - calib["closed_offset"]}
    if len(pitches) >= 10:
        pitch, roll = statistics.median(pitches), statistics.median(rolls)
        m.head = {
            "upright_threshold": pitch,
            "drooped_threshold": pitch + calib["drooped_offset"],
            "roll_upright_threshold": roll,
            "roll_drooped_threshold": roll + calib["roll_drooped_offset"],
        }
    body_tracker.close()
    heads = [p.head for p in poses if p.head is not None]
    m.body_base = BodyPose(*(statistics.median(getattr(p, f) for p in poses) for f in ("center_x", "center_y", "width", "tilt")), head=statistics.median(heads) if heads else None) if len(poses) >= 5 else None
    return n_cal


def read_frames(video, start=0, stop=None):
    capture = cv2.VideoCapture(str(video))
    index = 0
    while stop is None or index < stop:
        ok, frame = capture.read()
        if not ok:
            break
        if index >= start:
            yield index, frame
        index += 1
    capture.release()


def run(video, config, eye_model, yawn_model, writer):
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()

    m = monitor_for(config, eye_model, yawn_model)
    n_cal = calibrate(m, config, [frame for _, frame in read_frames(video, stop=int(config["calibration"]["duration_sec"] * fps))], fps)
    m.smoothers = {key: RollingMedian(m.settings["smoothing_window"]) for key in ("left", "right", "eyes", "mouth", "head", "body")}
    monitors = {key: AlertnessMonitor(**config[section]) for key, section in (("eyes", "alertness"), ("mouth", "yawn"), ("head", "nod"), ("body", "slump"))}
    body_tracker = BodyTracker(config["body"])
    escalation = EscalationMonitor(**config["escalation"])
    glare_box = None
    with FaceTracker() as tracker:
        for index, frame in read_frames(video, start=n_cal):
            now = index / fps
            regions = tracker.extract(remove_glare(frame, glare_box))
            glare_box = face_box(regions, frame.shape) if regions else None
            body = m._body(body_tracker.update(frame))
            row = {"video": video.name, "frame": index, "t": round(now, 3), "face": int(regions is not None), "person": int(regions is not None or body is not None)}
            cues = {}
            if regions:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for side, ids in (("left", LEFT_EYE_IDS), ("right", RIGHT_EYE_IDS)):
                    mean, std, box = eye_patch_stats(gray, regions.landmarks, ids)
                    row[f"{side}_patch_mean"], row[f"{side}_patch_std"] = mean, std
                    row[f"{side}_patch_glare"] = glare_fraction(frame, box) if box else None
                    row[f"{side}_region"] = int(side in regions.eyes)
                if regions.eyes:
                    predictions = eye_model.predict_many([eye.crop for eye in regions.eyes.values()])
                    for (side, eye), prediction in zip(regions.eyes.items(), predictions):
                        _, conf, closed = prediction
                        row[f"{side}_cnn_closed"], row[f"{side}_conf"], row[f"{side}_ear"] = closed, conf, eye.ratio
                        cues[side] = m._eye(frame, side, eye, prediction)
                if regions.mouth:
                    _, conf, opened = yawn_model.predict(regions.mouth.crop)
                    row["mouth_cnn_open"], row["mouth_conf"], row["mar"] = opened, conf, regions.mouth.ratio
                    cues["mouth"] = m._mouth(regions.mouth)
                else:
                    cues["mouth"] = None
                row["pitch"], row["roll"] = regions.pitch, regions.roll
                cues["head"] = m._head(regions.pitch, regions.roll)
            eyes = [(cues[side]["score"], regions.eyes[side].width) for side in ("left", "right") if cues.get(side)]
            weight = sum(width for _, width in eyes)
            scores = {
                "eyes": sum(s * w for s, w in eyes) / weight if weight else None,
                "mouth": cues["mouth"]["score"] if cues.get("mouth") else None,
                "head": cues["head"]["score"] if cues.get("head") else None,
                "body": body["score"] if regions is None and body and m.body_base else None,
            }
            statuses = {key: monitors[key].update(m.smoothers[key].add(score), now) for key, score in scores.items()}
            level = escalation.update(statuses, regions is not None or body is not None, now)
            for key, score in scores.items():
                row[f"{key}_score"] = score
                row[f"{key}_state"] = statuses[key].state
            row["level"], row["reason"] = level.level, level.reason
            row["screen"] = LiveMonitor._idle_state(regions, {"body": body}) if regions is None and level.level == 0 else level.level
            writer.writerow(row)


FIELDS = ["video", "frame", "t", "face", "person"]
for side in ("left", "right"):
    FIELDS += [f"{side}_{k}" for k in ("patch_mean", "patch_std", "patch_glare", "region", "cnn_closed", "conf", "ear")]
FIELDS += ["mouth_cnn_open", "mouth_conf", "mar", "pitch", "roll"]
FIELDS += [f"{k}_{s}" for k in ("eyes", "mouth", "head", "body") for s in ("score", "state")]
FIELDS += ["level", "reason", "screen"]

if __name__ == "__main__":
    out = Path(sys.argv[1])
    config = load_config()
    eye_model, yawn_model = Classifier("eye"), Classifier("yawn")
    for video in map(Path, sys.argv[2:]):
        path = out / f"{video.stem}.csv"
        if path.exists():
            continue
        partial = path.with_suffix(".part")
        with partial.open("w", newline="") as f:
            writer = csv.DictWriter(f, FIELDS)
            writer.writeheader()
            run(video, config, eye_model, yawn_model, writer)
        partial.rename(path)
        print("done", video.name, flush=True)
