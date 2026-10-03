import statistics
import sys
import threading
import time

import cv2
import numpy as np

from watch.alertness import AlertnessMonitor, EscalationMonitor, RollingMedian
from watch.config import load_config
from watch.face import FACE_MESH, FaceTracker
from watch.model import Classifier

WINDOW = "WATCH - Live Alertness Monitoring (press q to quit)"
GREEN, RED, AMBER, MAGENTA, ORANGE = (0, 200, 0), (0, 0, 220), (255, 200, 0), (255, 0, 200), (0, 165, 255)
YELLOW, GRAY = (0, 215, 255), (120, 120, 120)
BANNERS = {0: ("NORMAL", GREEN), 1: ("WARNING", YELLOW), 2: ("CRITICAL", RED), "no_face": ("NO FACE DETECTED", GRAY)}
EYE_CONTOURS = {"left": FACE_MESH.FACEMESH_RIGHT_EYE, "right": FACE_MESH.FACEMESH_LEFT_EYE}


def _glare_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return (hsv[:, :, 2] > 235) & (hsv[:, :, 1] < 60)


def remove_glare(image: np.ndarray) -> np.ndarray:
    """Inpaints bright washed-out highlights such as visor reflections before landmark tracking."""
    mask = _glare_mask(image).astype(np.uint8) * 255
    if not mask.any():
        return image
    return cv2.inpaint(image, cv2.dilate(mask, np.ones((3, 3), np.uint8)), 3, cv2.INPAINT_TELEA)


def glare_fraction(image: np.ndarray, box) -> float:
    region = image[box[1] : box[3], box[0] : box[2]]
    return float(_glare_mask(region).mean()) if region.size else 0.0


def linear_score(value: float, low: float, high: float) -> float:
    return 0.0 if high == low else min(1.0, max(0.0, (value - low) / (high - low)))


class Alarm:
    """Repeating beep while active, using winsound on Windows and the terminal bell elsewhere."""

    def __init__(self):
        self._active = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def set(self, active: bool):
        self._active.set() if active else self._active.clear()

    def _loop(self):
        while True:
            self._active.wait()
            if sys.platform == "win32":
                import winsound

                winsound.Beep(880, 250)
            else:
                print("\a", end="", flush=True)
            time.sleep(0.25)


class LiveMonitor:
    def __init__(self, camera: int):
        config = load_config()
        self.config = config
        self.settings = config["live"]
        self.eye_model, self.yawn_model = Classifier("eye"), Classifier("yawn")
        self.ear, self.head = config["ear"], config["head_pose"]
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(camera, backend)
        if not self.capture.isOpened():
            raise ConnectionError(f"Could not open camera {camera}, try another --camera index.")
        self.tracker = FaceTracker()
        self.alarm = Alarm()
        self._reset()

    def _reset(self):
        window = self.settings["smoothing_window"]
        self.smoothers = {key: RollingMedian(window) for key in ("left", "right", "eyes", "mouth", "head")}
        self.monitors = {key: AlertnessMonitor(**self.config[section]) for key, section in (("eyes", "alertness"), ("mouth", "yawn"), ("head", "nod"))}
        self.escalation = EscalationMonitor(**self.config["escalation"])

    def _read(self):
        ok, frame = self.capture.read()
        return cv2.flip(frame, 1) if ok else None

    def calibrate(self):
        """Measures this user's resting EAR and head pose so thresholds fit them rather than an average face."""
        settings = self.config["calibration"]
        ears, pitches, rolls = [], [], []
        start = time.time()
        while (remaining := settings["duration_sec"] - (time.time() - start)) > 0:
            frame = self._read()
            if frame is None:
                break
            regions = self.tracker.extract(remove_glare(frame))
            if regions:
                ears += [eye.ratio for eye in regions.eyes.values()]
                if regions.pitch is not None:
                    pitches.append(regions.pitch)
                    rolls.append(regions.roll)
            for y, text in ((40, "Calibrating, look at the camera normally"), (80, f"{remaining:.1f}s")):
                cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ORANGE, 2)
            cv2.imshow(WINDOW, frame)
            cv2.waitKey(1)

        self.ear, self.head = self.config["ear"], self.config["head_pose"]
        if len(ears) < 10:
            print("Calibration found no face, using default thresholds.")
        else:
            baseline = statistics.median(ears)
            self.ear = {"open_threshold": baseline, "closed_threshold": baseline - settings["closed_offset"]}
        if len(pitches) >= 10:
            pitch, roll = statistics.median(pitches), statistics.median(rolls)
            self.head = {
                "upright_threshold": pitch,
                "drooped_threshold": pitch + settings["drooped_offset"],
                "roll_upright_threshold": roll,
                "roll_drooped_threshold": roll + settings["roll_drooped_offset"],
            }
        print(f"Thresholds: EAR closed<={self.ear['closed_threshold']:.2f}, pitch drooped>={self.head['drooped_threshold']:.1f}")
        self._reset()

    def _eye(self, frame, side, eye):
        _, confidence, cnn_closed = self.eye_model.predict(eye.crop)
        if confidence < self.settings["min_cnn_confidence"]:
            return None
        ear_closed = eye.ratio <= self.ear["closed_threshold"] and glare_fraction(frame, eye.box) < self.settings["ear_glare_tolerance"]
        closedness = self.smoothers[side].add(max(float(ear_closed), cnn_closed))
        return {"score": closedness, "label": "closed" if closedness >= 0.5 else "open", "ratio": eye.ratio}

    def _mouth(self, mouth):
        _, confidence, cnn_open = self.yawn_model.predict(mouth.crop)
        if confidence < self.settings["min_cnn_confidence"]:
            return None
        mar = self.config["mar"]
        openness = max(cnn_open, linear_score(mouth.ratio, mar["closed_threshold"], mar["open_threshold"]))
        return {"score": openness, "label": "yawn" if openness >= 0.5 else "no yawn", "ratio": mouth.ratio}

    def _head(self, pitch, roll):
        if pitch is None:
            return None
        head = self.head
        droop = max(
            linear_score(pitch, head["upright_threshold"], head["drooped_threshold"]),
            linear_score(roll, head["roll_upright_threshold"], head["roll_drooped_threshold"]),
        )
        label = "nodding" if droop >= self.config["nod"]["closedness_threshold"] else "upright"
        return {"score": droop, "label": label, "pitch": pitch, "roll": roll}

    def _draw(self, frame, regions, cues, escalation, statuses, fps):
        h, w = frame.shape[:2]

        def point(i):
            return int(regions.landmarks[i].x * w), int(regions.landmarks[i].y * h)

        def brackets(connections, margin, color, thickness=2):
            points = np.array([point(i) for i in {i for pair in connections for i in pair}])
            (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)
            pad_x, pad_y = int((x1 - x0) * margin), int((y1 - y0) * margin)
            x0, y0, x1, y1 = x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y
            length = max(6, int(min(x1 - x0, y1 - y0) * 0.25))
            for x, y, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
                cv2.line(frame, (x, y), (x + dx * length, y), color, thickness, cv2.LINE_AA)
                cv2.line(frame, (x, y), (x, y + dy * length), color, thickness, cv2.LINE_AA)

        lines = []
        for side in ("left", "right"):
            if cue := cues.get(side):
                color = GREEN if cue["label"] == "open" else RED
                brackets(EYE_CONTOURS[side], 0.25, color)
                lines.append((f"{side} eye: {cue['label']} {cue['score']:.0%} | EAR {cue['ratio']:.2f}", color))
        if cue := cues.get("mouth"):
            color = RED if cue["label"] == "yawn" else AMBER
            brackets(FACE_MESH.FACEMESH_LIPS, 0.2, color)
            lines.append((f"mouth: {cue['label']} {cue['score']:.0%} | MAR {cue['ratio']:.2f}", color))
        if cue := cues.get("head"):
            color = MAGENTA if cue["label"] == "upright" else RED
            brackets(FACE_MESH.FACEMESH_FACE_OVAL, 0.05, color, 3)
            # Forehead to chin, so sideways tilt is visible.
            cv2.line(frame, point(10), point(152), color, 2, cv2.LINE_AA)
            for i in (10, 152):
                cv2.circle(frame, point(i), 4, color, -1, cv2.LINE_AA)
            lines.append((f"head: {cue['label']} | pitch {cue['pitch']:.1f} | roll {cue['roll']:.1f} deg", color))

        if lines:
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 88), (280, 98 + len(lines) * 22), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            for i, (text, color) in enumerate(lines):
                cv2.putText(frame, text, (18, 88 + 22 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        text, color = BANNERS["no_face" if regions is None and escalation.level == 0 else escalation.level]
        if escalation.reason:
            text += f": {escalation.reason}"
        cv2.rectangle(frame, (0, 0), (w, 50), color, -1)
        cv2.putText(frame, text, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0) if color == YELLOW else (255, 255, 255), 2)
        detail = " | ".join(f"{label} {statuses[key].closed_duration:.1f}s" for key, label in (("eyes", "eyes closed"), ("mouth", "mouth open"), ("head", "head drooped")))
        if escalation.level == 1:
            detail += f" | watching {escalation.window_left:.0f}s"
        cv2.putText(frame, detail, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    def run(self):
        self.calibrate()
        no_face_since, previous = None, time.time()
        try:
            while (frame := self._read()) is not None:
                regions = self.tracker.extract(remove_glare(frame))
                if regions is None:
                    no_face_since = no_face_since or time.time()
                elif no_face_since is not None:
                    # A face returning after a real gap is likely a new driver, unless a warning is being held.
                    gap, no_face_since = time.time() - no_face_since, None
                    if gap >= self.settings["recalibrate_gap_sec"] and self.escalation.level == 0:
                        self.calibrate()
                        continue

                cues = {}
                if regions:
                    cues = {side: self._eye(frame, side, eye) for side, eye in regions.eyes.items()}
                    cues["mouth"] = self._mouth(regions.mouth) if regions.mouth else None
                    cues["head"] = self._head(regions.pitch, regions.roll)
                # The eye nearer the camera looks wider, so it counts more.
                eyes = [(cues[side]["score"], regions.eyes[side].width) for side in ("left", "right") if cues.get(side)]
                eye_weight = sum(width for _, width in eyes)
                scores = {
                    "eyes": sum(score * width for score, width in eyes) / eye_weight if eye_weight else None,
                    "mouth": cues["mouth"]["score"] if cues.get("mouth") else None,
                    "head": cues["head"]["score"] if cues.get("head") else None,
                }

                now = time.time()
                statuses = {key: self.monitors[key].update(self.smoothers[key].add(score), now) for key, score in scores.items()}
                escalation = self.escalation.update(statuses, regions is not None, now)
                self.alarm.set(escalation.level == 2)

                self._draw(frame, regions, cues, escalation, statuses, 1.0 / max(now - previous, 1e-6))
                previous = now
                cv2.imshow(WINDOW, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.alarm.set(False)
            self.capture.release()
            self.tracker.close()
            cv2.destroyAllWindows()
