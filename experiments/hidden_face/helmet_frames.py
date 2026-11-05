import csv
import math
import re
import sys
from pathlib import Path

import cv2
import mediapipe as mp

from watch.body import shoulders
from watch.face import LEFT_EYE_IDS, RIGHT_EYE_IDS, FaceTracker
from watch.live import glare_fraction, remove_glare
from watch.model import Classifier

POSE = mp.solutions.pose.PoseLandmark
FIELDS = [
    "folder", "session", "index", "face", "pose", "nose_vis", "ear_l_vis", "ear_r_vis", "ear_line_deg", "nose_drop",
    "shoulder_tilt", "left_closed", "left_conf", "left_ear", "left_glare", "left_mean", "left_std",
    "right_closed", "right_conf", "right_ear", "right_glare", "right_mean", "right_std",
    "mouth_open", "mouth_conf", "mar", "pitch", "roll",
]


def session_of(path):
    m = re.match(r"(.+?)_(\d+)$", path.stem)
    return (m.group(1), int(m.group(2))) if m else (path.stem, -1)


def resize(image, width=640):
    h, w = image.shape[:2]
    return image if w <= width else cv2.resize(image, (width, int(h * width / w)))


def eye_stats(frame, gray, landmarks, ids):
    h, w = gray.shape
    xs = [landmarks[i].x * w for i in ids]
    ys = [landmarks[i].y * h for i in ids]
    x0, x1 = int(max(min(xs), 0)), int(min(max(xs), w))
    y0, y1 = int(max(min(ys) - (max(xs) - min(xs)) * 0.2, 0)), int(min(max(ys) + (max(xs) - min(xs)) * 0.2, h))
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return None, None, None
    return float(patch.mean()), float(patch.std()), glare_fraction(frame, (x0, y0, x1, y1))


def pose_row(pose_result, w, h):
    if not pose_result.pose_landmarks:
        return {"pose": 0}
    lm = pose_result.pose_landmarks.landmark
    nose, ear_l, ear_r = lm[POSE.NOSE], lm[POSE.LEFT_EAR], lm[POSE.RIGHT_EAR]
    row = {"pose": 1, "nose_vis": nose.visibility, "ear_l_vis": ear_l.visibility, "ear_r_vis": ear_r.visibility}
    row["ear_line_deg"] = math.degrees(math.atan2((ear_r.y - ear_l.y) * h, abs(ear_r.x - ear_l.x) * w + 1e-6))
    body = shoulders(lm, w, h, 0.5)
    if body:
        row["shoulder_tilt"] = body.tilt
        row["nose_drop"] = (nose.y * h - body.center_y) / body.width
    return row


if __name__ == "__main__":
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    skip = set(sys.argv[3:])
    eye_model, yawn_model = Classifier("eye"), Classifier("yawn")
    images = sorted((session_of(p), p) for p in root.rglob("*.jpg"))
    with out.open("w", newline="") as f, FaceTracker(static_image_mode=True) as tracker, mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        for n, ((session, index), path) in enumerate(images):
            if session in skip:
                continue
            frame = resize(cv2.imread(str(path)))
            h, w = frame.shape[:2]
            row = {"folder": path.parent.name, "session": session, "index": index}
            row.update(pose_row(pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), w, h))
            regions = tracker.extract(remove_glare(frame))
            row["face"] = int(regions is not None)
            if regions:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for side, ids in (("left", LEFT_EYE_IDS), ("right", RIGHT_EYE_IDS)):
                    row[f"{side}_mean"], row[f"{side}_std"], row[f"{side}_glare"] = eye_stats(frame, gray, regions.landmarks, ids)
                if regions.eyes:
                    for (side, eye), (_, conf, closed) in zip(regions.eyes.items(), eye_model.predict_many([e.crop for e in regions.eyes.values()])):
                        row[f"{side}_closed"], row[f"{side}_conf"], row[f"{side}_ear"] = closed, conf, eye.ratio
                if regions.mouth:
                    _, conf, opened = yawn_model.predict(regions.mouth.crop)
                    row["mouth_open"], row["mouth_conf"], row["mar"] = opened, conf, regions.mouth.ratio
                row["pitch"], row["roll"] = regions.pitch, regions.roll
            writer.writerow(row)
            if n % 500 == 0:
                print(n, len(images), flush=True)
    print("done", flush=True)
