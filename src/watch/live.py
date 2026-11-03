import statistics
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

from watch import dashboard as ui
from watch.alertness import AlertnessMonitor, EscalationMonitor, RollingMedian, head_droop
from watch.camera import open_camera
from watch.config import load_config
from watch.report import LEVEL_NAMES, SessionLog
from watch.face import FACE_MESH, FaceTracker, brighten, face_sharpness, mean_brightness
from watch.model import Classifier
from watch.stream import BrowserStream, Window

WINDOW = "WATCH - Live Alertness Monitoring"
# Marker colours on the video, as BGR.
GREEN, RED, AMBER, MAGENTA = (0, 200, 0), (0, 0, 220), (255, 200, 0), (255, 0, 200)
EYE_CONTOURS = {"left": FACE_MESH.FACEMESH_RIGHT_EYE, "right": FACE_MESH.FACEMESH_LEFT_EYE}


def _glare_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return (hsv[:, :, 2] > 235) & (hsv[:, :, 1] < 60)


def remove_glare(image: np.ndarray, box=None) -> np.ndarray:
    """Inpaints bright washed-out highlights such as visor reflections before landmark tracking, only inside box when given."""
    x0, y0, x1, y1 = box or (0, 0, image.shape[1], image.shape[0])
    region = image[y0:y1, x0:x1]
    mask = _glare_mask(region).astype(np.uint8) * 255
    if not mask.any():
        return image
    cleaned = image.copy()
    cleaned[y0:y1, x0:x1] = cv2.inpaint(region, cv2.dilate(mask, np.ones((3, 3), np.uint8)), 3, cv2.INPAINT_TELEA)
    return cleaned


def face_box(regions, shape, margin: float = 0.3):
    """Landmark bounds grown by margin, where glare matters on the next frame."""
    h, w = shape[:2]
    xs = [p.x * w for p in regions.landmarks]
    ys = [p.y * h for p in regions.landmarks]
    pad_x, pad_y = (max(xs) - min(xs)) * margin, (max(ys) - min(ys)) * margin
    box = (max(int(min(xs) - pad_x), 0), max(int(min(ys) - pad_y), 0), min(int(max(xs) + pad_x), w), min(int(max(ys) + pad_y), h))
    return box if box[2] > box[0] and box[3] > box[1] else None


def glare_fraction(image: np.ndarray, box) -> float:
    region = image[box[1] : box[3], box[0] : box[2]]
    return float(_glare_mask(region).mean()) if region.size else 0.0


def linear_score(value: float, low: float, high: float) -> float:
    return 0.0 if high == low else min(1.0, max(0.0, (value - low) / (high - low)))


class _Quit(Exception):
    pass


class Alarm:
    """Repeating beep while active, using winsound on Windows and the terminal bell elsewhere."""

    def __init__(self):
        self._active = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def set(self, active: bool):
        self._active.set() if active else self._active.clear()

    def is_active(self) -> bool:
        return self._active.is_set()

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
    def __init__(self, camera, stream_port: int | None = None, allow_shutdown: bool = False):
        config = load_config()
        self.config = config
        self.settings = config["live"]
        self.eye_model, self.yawn_model = Classifier("eye"), Classifier("yawn")
        self.ear, self.head = config["ear"], config["head_pose"]
        self._camera = camera
        self.capture = open_camera(camera)
        if not self.capture.isOpened():
            raise ConnectionError(f"Could not open camera {camera}, try another --camera index.")
        self.tracker = FaceTracker()
        self.alarm = Alarm()
        self.display = Window(WINDOW) if stream_port is None else BrowserStream(stream_port, self.alarm.is_active, allow_shutdown)
        self.view = ui.DashboardView()
        self.events = deque(maxlen=50)
        self.calibrated = False
        self.muted = False
        self.session = SessionLog()
        self._notice, self._notice_until = "", 0.0
        self._last_frame = None
        self._reset()

    def _reset(self):
        window = self.settings["smoothing_window"]
        self.smoothers = {key: RollingMedian(window) for key in ("left", "right", "eyes", "mouth", "head")}
        self.monitors = {key: AlertnessMonitor(**self.config[section]) for key, section in (("eyes", "alertness"), ("mouth", "yawn"), ("head", "nod"))}
        self.escalation = EscalationMonitor(**self.config["escalation"])
        self._level = 0

    def _read(self):
        ok, frame = self.capture.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)
        self._last_frame = frame.copy()
        return frame

    def _handle_key(self, key):
        if key == ord("q"):
            raise _Quit
        if key == ord("m"):
            self.muted = not self.muted
        if key == ord("e"):
            self.export()

    def _camera_lost(self):
        lost_since, next_retry = time.time(), 0.0
        self.events.append((time.strftime("%H:%M:%S"), "camera_lost", ""))
        self.session.event("Camera lost")
        print("Camera lost, retrying.")
        while True:
            now = time.time()
            lost_for = now - lost_since
            alarming = lost_for >= self.settings["camera_lost_alarm_sec"]
            self.alarm.set(alarming and not self.muted)
            if now >= next_retry:
                next_retry = now + self.settings["camera_retry_sec"]
                self.capture.release()
                self.capture = open_camera(self._camera)
                if self.capture.isOpened() and (frame := self._read()) is not None:
                    self.alarm.set(False)
                    print("Camera back.")
                    return frame
            self.session.add(now, "Camera lost")
            name, color = ui.LEVELS["camera_lost"]
            level = ui.Level(name, color, "check the camera connection", f"lost {lost_for:.0f}s" + (" · alarm on" if alarming else ""))
            frame = self._last_frame if self._last_frame is not None else np.zeros((480, 640, 3), np.uint8)
            view = self.view.render((frame * 0.3).astype(np.uint8), level, [], [], [], self._event_lines(), live=True, muted=self.muted, readings=[])
            self._handle_key(self.display.show(view, 30))

    def calibrate(self):
        """Measures this user's resting EAR and head pose so thresholds fit them rather than an average face."""
        settings = self.config["calibration"]
        self.alarm.set(False)
        ears, pitches, rolls = [], [], []
        start = time.time()
        while (remaining := settings["duration_sec"] - (time.time() - start)) > 0:
            frame = self._read()
            if frame is None:
                frame = self._camera_lost()
                start = time.time()
            regions = self.tracker.extract(remove_glare(frame))
            self.session.add(time.time(), "Calibrating")
            if regions:
                ears += [eye.ratio for eye in regions.eyes.values()]
                if regions.pitch is not None:
                    pitches.append(regions.pitch)
                    rolls.append(regions.roll)
            name, color = ui.LEVELS["calibrating"]
            level = ui.Level(name, color, "look at the camera normally", f"{remaining:.1f}s left")
            view = self.view.render(frame, level, [], [], [], self._event_lines(), live=False, muted=self.muted, readings=[])
            self._handle_key(self.display.show(view, 1))

        self.ear, self.head = self.config["ear"], self.config["head_pose"]
        self.calibrated = len(ears) >= 10
        if not self.calibrated:
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

    def _eye(self, frame, side, eye, prediction):
        _, confidence, cnn_closed = prediction
        if confidence < self.settings["min_cnn_confidence"]:
            return None
        glare = glare_fraction(frame, eye.box) >= self.settings["ear_glare_tolerance"]
        shape_closed = eye.ratio <= self.ear["closed_threshold"]
        # An open eye shape overrules the CNN, which misreads soft IR crops as closed, unless glare makes the shape unreliable.
        cnn_score = cnn_closed if shape_closed or glare else 0.0
        closedness = self.smoothers[side].add(max(float(shape_closed and not glare), cnn_score))
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
        droop = head_droop(pitch, roll, self.head)
        label = "nodding" if droop >= self.config["nod"]["closedness_threshold"] else "upright"
        return {"score": droop, "label": label, "pitch": pitch, "roll": roll}

    def _draw(self, frame, regions, cues, escalation, statuses, fps, brightness, sharpness=None):
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

        for side in ("left", "right"):
            if cue := cues.get(side):
                brackets(EYE_CONTOURS[side], 0.25, GREEN if cue["label"] == "open" else RED)
        if cue := cues.get("mouth"):
            brackets(FACE_MESH.FACEMESH_LIPS, 0.2, RED if cue["label"] == "yawn" else AMBER)
        if cue := cues.get("head"):
            color = MAGENTA if cue["label"] == "upright" else RED
            brackets(FACE_MESH.FACEMESH_FACE_OVAL, 0.05, color, 3)
            # Forehead to chin, so sideways tilt is visible.
            cv2.line(frame, point(10), point(152), color, 2, cv2.LINE_AA)
            for i in (10, 152):
                cv2.circle(frame, point(i), 4, color, -1, cv2.LINE_AA)

        level, signs, checks, calibration, readings = self._dashboard(regions, cues, escalation, statuses, fps, brightness, sharpness)
        notice = self._notice if time.time() < self._notice_until else ""
        return self.view.render(frame, level, signs, checks, calibration, self._event_lines(), live=True, muted=self.muted, readings=readings, notice=notice)

    def _dashboard(self, regions, cues, escalation, statuses, fps, brightness, sharpness=None):
        """The dashboard contents, mirroring updateDashboard and showCalibration in web/app.js."""
        name, color = ui.LEVELS["no_face" if regions is None and escalation.level == 0 else escalation.level]
        timer = {1: f"watching {escalation.window_left:.0f}s", 2: "alarm on"}.get(escalation.level, "")
        level = ui.Level(name, color, escalation.reason, timer)

        eyes = [cues[side] for side in ("left", "right") if cues.get(side)]
        mouth, head = cues.get("mouth"), cues.get("head")
        signs = []
        for label, key, state, concerning in (
            ("Eyes", "eyes", eyes and ("closed" if np.mean([cue["score"] for cue in eyes]) >= 0.5 else "open"), "closed"),
            ("Yawn", "mouth", mouth and ("yawning" if mouth["label"] == "yawn" else "normal"), "yawning"),
            ("Head", "head", head and ("drooped" if head["label"] == "nodding" else "upright"), "drooped"),
        ):
            if not state:
                dot, state = ui.GRAY, "not visible"
            else:
                dot = ui.RED if statuses[key].state == "not_alert" else ui.YELLOW if state == concerning else ui.GREEN
            signs.append(ui.Row(dot, label, state, ui.MUTED if dot == ui.GRAY else dot, f"{statuses[key].closed_duration:.1f}s"))

        face, dark = regions is not None, brightness < self.settings["low_light_mean"]
        checks = [
            ui.Row(ui.GREEN if face else ui.RED, "Face", "found" if face else "not found"),
            ui.Row((ui.RED, ui.YELLOW, ui.GREEN)[len(eyes)] if face else ui.GRAY, "Eyes", f"{len(eyes)} of 2 readable" if face else "-"),
            ui.Row((ui.GREEN if mouth else ui.YELLOW) if face else ui.GRAY, "Mouth", ("readable" if mouth else "not readable") if face else "-"),
            ui.Row(ui.YELLOW if dark else ui.GREEN, "Lighting", "low, boosted" if dark else "good"),
            self._sharpness_row(sharpness),
            ui.Row(ui.GREEN if fps >= 15 else ui.YELLOW, "FPS", f"{fps:.0f}"),
            ui.Row(ui.GREEN, "Device", "CPU"),
        ]
        for row in checks:
            row.value_color = ui.MUTED if row.color == ui.GRAY else row.color

        calibration = [
            ui.Row(ui.GREEN if self.calibrated else ui.YELLOW, "Status", "personal" if self.calibrated else "default", ui.GREEN if self.calibrated else ui.YELLOW),
            ui.Row(None, "Open eye", f"EAR {self.ear['open_threshold']:.2f}"),
            ui.Row(None, "Closed eye", f"EAR \u2264 {self.ear['closed_threshold']:.2f}"),
            ui.Row(None, "Head down", f"\u2265 {self.head['drooped_threshold']:.1f}\u00b0"),
            ui.Row(None, "Head tilt", f"\u2265 {self.head['roll_drooped_threshold']:.1f}\u00b0"),
        ]
        # What is being measured right now, to compare with the calibrated limits; red once past them.
        ear = self._readings(cues)["ear"]
        readings = [
            ("Eyes (EAR)", "-" if ear is None else f"{ear:.2f}", ear is not None and ear <= self.ear["closed_threshold"]),
            ("Mouth (MAR)", f"{mouth['ratio']:.2f}" if mouth else "-", bool(mouth) and mouth["label"] == "yawn"),
            ("Head down", f"{head['pitch']:.1f}\u00b0" if head else "-", bool(head) and head["pitch"] >= self.head["drooped_threshold"]),
            ("Head tilt", f"{head['roll']:.1f}\u00b0" if head else "-", bool(head) and head["roll"] >= self.head["roll_drooped_threshold"]),
        ]
        readings = [ui.Row(None, label, value, ui.RED if past else ui.MUTED if value == "-" else ui.TEXT) for label, value, past in readings]
        return level, signs, checks, calibration, readings

    def _sharpness_row(self, sharpness):
        if sharpness is None:
            return ui.Row(ui.GRAY, "Sharpness", "-")
        if sharpness >= self.settings["sharp_min"]:
            return ui.Row(ui.GREEN, "Sharpness", "sharp")
        if sharpness >= self.settings["blurry_below"]:
            return ui.Row(ui.YELLOW, "Sharpness", "a bit blurry")
        return ui.Row(ui.RED, "Sharpness", "blurry")

    @staticmethod
    def _readings(cues):
        eyes = [cues[side]["ratio"] for side in ("left", "right") if cues.get(side)]
        mouth, head = cues.get("mouth"), cues.get("head")
        return {
            "ear": float(np.mean(eyes)) if eyes else None,
            "mar": mouth["ratio"] if mouth else None,
            "head_down": head["pitch"] if head else None,
            "head_tilt": head["roll"] if head else None,
        }

    def _limits(self):
        return {"ear_closed": self.ear["closed_threshold"], "head_down": self.head["drooped_threshold"], "head_tilt": self.head["roll_drooped_threshold"]}

    def export(self):
        """Saves the session so far as a PNG report and a CSV under results/sessions."""
        png, csv_path = self.session.save(self.config["paths"]["results_dir"] / "sessions")
        print(f"Saved {png} and {csv_path}")
        self._notice, self._notice_until = f"Saved {png.name}", time.time() + 4

    def _event_lines(self):
        """Recent level changes, newest first."""
        return [(f"{stamp} {ui.LEVELS[level][0]} {reason}", ui.LEVELS[level][1]) for stamp, level, reason in reversed(self.events)]

    def run(self):
        no_face_since, previous, glare_box = None, time.time(), None
        try:
            self.calibrate()
            while True:
                if (frame := self._read()) is None:
                    self._camera_lost()
                    self.calibrate()
                    self.events.append((time.strftime("%H:%M:%S"), 0, "recalibrated"))
                    self.session.event("Recalibrated")
                    no_face_since, glare_box = None, None
                    continue
                # Only glare on the face matters, so skip bright spots elsewhere such as windows.
                regions = self.tracker.extract(remove_glare(frame, glare_box))
                glare_box = face_box(regions, frame.shape) if regions else None
                if regions is None:
                    no_face_since = no_face_since or time.time()
                elif no_face_since is not None:
                    # The user left the camera, so calibrate again for whoever is in front of it now.
                    gap, no_face_since = time.time() - no_face_since, None
                    if gap >= self.settings["recalibrate_gap_sec"]:
                        self.calibrate()
                        self.events.append((time.strftime("%H:%M:%S"), 0, "recalibrated"))
                        self.session.event("Recalibrated")
                        continue

                cues = {}
                if regions:
                    predictions = self.eye_model.predict_many([eye.crop for eye in regions.eyes.values()]) if regions.eyes else []
                    cues = {side: self._eye(frame, side, eye, prediction) for (side, eye), prediction in zip(regions.eyes.items(), predictions)}
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
                self.alarm.set(escalation.level == 2 and not self.muted)
                if escalation.level != self._level:
                    self._level = escalation.level
                    self.events.append((time.strftime("%H:%M:%S"), escalation.level, escalation.reason))
                    self.session.event(f"{LEVEL_NAMES[escalation.level]}: {escalation.reason}".rstrip(": "))
                state = "No face" if regions is None and escalation.level == 0 else LEVEL_NAMES[escalation.level]
                self.session.add(now, state, self._readings(cues), self._limits())

                sharpness = face_sharpness(brighten(frame, self.settings["low_light_mean"], self.settings["low_light_target"]), regions.landmarks) if regions else None
                view = self._draw(frame, regions, cues, escalation, statuses, 1.0 / max(now - previous, 1e-6), mean_brightness(frame), sharpness)
                previous = now
                self._handle_key(self.display.show(view, 1))
        except _Quit:
            pass
        finally:
            self.alarm.set(False)
            self.capture.release()
            self.tracker.close()
            self.display.close()
