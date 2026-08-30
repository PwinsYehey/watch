import statistics
import time

import cv2
import mediapipe as mp

from frames.config import load_config
from frames.inference.alertness import AlertnessMonitor
from frames.inference.predict import predict_eye_state, predict_yawn_state
from frames.models.build_model import load_trained_model
from frames.preprocessing.facemesh import EyeRegion, EyeRegionTracker, MouthRegion

_OPEN_COLOR = (0, 200, 0)
_CLOSED_COLOR = (0, 0, 220)
_MOUTH_COLOR = (255, 200, 0)
_HEAD_COLOR = (255, 0, 200)
_NO_FACE_COLOR = (0, 165, 255)
_NO_FACE_RECALIBRATE_GAP_SEC = 3.0

_mp_face_mesh = mp.solutions.face_mesh
_LEFT_EYE_CONNECTIONS = _mp_face_mesh.FACEMESH_LEFT_EYE
_RIGHT_EYE_CONNECTIONS = _mp_face_mesh.FACEMESH_RIGHT_EYE
_LIPS_CONNECTIONS = _mp_face_mesh.FACEMESH_LIPS
_FACE_OVAL_CONNECTIONS = _mp_face_mesh.FACEMESH_FACE_OVAL

_BANNER_COLORS = {
    "alert": (0, 200, 0),
    "not_alert": (0, 0, 220),
    "no_face": (0, 165, 255),
}
_BANNER_TEXT = {
    "alert": "ALERT",
    "not_alert": "NOT ALERT",
    "no_face": "NO FACE DETECTED",
}


def _linear_score(value: float, low_at_zero: float, high_at_one: float) -> float:
    """Maps `value` to [0, 1], 0 at low_at_zero and 1 at high_at_one."""
    span = high_at_one - low_at_zero
    if span == 0:
        return 0.0
    score = (value - low_at_zero) / span
    return max(0.0, min(1.0, score))


def _build_monitor(cfg: dict) -> AlertnessMonitor:
    """Builds an AlertnessMonitor from a config section (alertness/yawn/nod all share this shape)."""
    return AlertnessMonitor(
        closed_duration_threshold=cfg["closed_duration_threshold_sec"],
        perclos_window=cfg["perclos_window_sec"],
        perclos_threshold=cfg["perclos_threshold"],
        closedness_threshold=cfg["closedness_threshold"],
        dip_grace_period=cfg.get("dip_grace_period_sec", 0.0),
    )


def _calibrate_ear(cap, tracker, duration_sec: float, closed_fraction: float) -> dict | None:
    """Measures this user's own open-eye EAR for a few seconds so thresholds fit their eye shape, not a population average."""
    samples = []
    start = time.time()
    while time.time() - start < duration_sec:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        regions = tracker.extract(frame)
        if regions is not None:
            samples.extend(eye.ear for eye in regions.eyes.values())

        remaining = duration_sec - (time.time() - start)
        cv2.putText(frame, "Calibrating -- look at the camera normally", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(frame, f"{remaining:.1f}s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.imshow("FRAMES - Live Alertness Monitoring (press q to quit)", frame)
        cv2.waitKey(1)

    if len(samples) < 10:
        return None

    baseline = statistics.median(samples)
    return {"open_threshold": baseline, "closed_threshold": baseline * closed_fraction}


def _draw_contour(frame, landmarks, connections, color):
    h, w = frame.shape[:2]
    for start_idx, end_idx in connections:
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        pt1 = (int(start.x * w), int(start.y * h))
        pt2 = (int(end.x * w), int(end.y * h))
        cv2.line(frame, pt1, pt2, color, 2)


def _draw_eye(frame, side: str, region: EyeRegion, landmarks, model, image_size, class_names, ear_cfg) -> dict:
    _cnn_label, _cnn_confidence, cnn_closedness = predict_eye_state(model, region.crop, image_size, class_names)
    ear_closedness = _linear_score(region.ear, ear_cfg["open_threshold"], ear_cfg["closed_threshold"])
    # the CNN can raise closedness when EAR is ambiguous, but can't override a clearly-open EAR reading
    closedness = max(cnn_closedness, ear_closedness) if ear_closedness > 0 else 0.0
    # the displayed label always matches this combined score, not just the CNN's own opinion
    label = "closed" if closedness >= 0.5 else "open"

    connections = _LEFT_EYE_CONNECTIONS if side == "left" else _RIGHT_EYE_CONNECTIONS
    _draw_contour(frame, landmarks, connections, _OPEN_COLOR if label == "open" else _CLOSED_COLOR)
    return {"closedness": closedness, "label": label, "ear": region.ear}


def _draw_mouth(frame, region: MouthRegion, landmarks, yawn_model, image_size, yawn_class_names, mar_cfg) -> dict:
    _cnn_label, _cnn_confidence, cnn_openness = predict_yawn_state(yawn_model, region.crop, image_size, yawn_class_names)
    mar_openness = _linear_score(region.mar, mar_cfg["closed_threshold"], mar_cfg["open_threshold"])
    # either signal suggesting a yawn is enough -- a safety monitor should not need both to agree
    openness = max(cnn_openness, mar_openness)
    # the displayed label always matches this combined score, not just the CNN's own opinion
    label = "yawn" if openness >= 0.5 else "no yawn"

    _draw_contour(frame, landmarks, _LIPS_CONNECTIONS, _CLOSED_COLOR if label == "yawn" else _MOUTH_COLOR)
    return {"openness": openness, "label": label, "mar": region.mar}


def _head_pose_status(frame, pitch: float | None, landmarks, head_pose_cfg) -> dict | None:
    if pitch is None:
        return None
    droop = _linear_score(pitch, head_pose_cfg["upright_threshold"], head_pose_cfg["drooped_threshold"])
    label = "nodding" if droop >= 0.5 else "upright"
    _draw_contour(frame, landmarks, _FACE_OVAL_CONNECTIONS, _HEAD_COLOR if label == "upright" else _CLOSED_COLOR)
    return {"droop": droop, "label": label, "pitch": pitch}


def _draw_banner(frame, state: str, eye_status, yawn_status, nod_status):
    color = _BANNER_COLORS[state]
    text = _BANNER_TEXT[state]
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), color, -1)
    cv2.putText(frame, text, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    detail = (
        f"eyes closed {eye_status.closed_duration:.1f}s | "
        f"mouth open {yawn_status.closed_duration:.1f}s | "
        f"head drooped {nod_status.closed_duration:.1f}s"
    )
    cv2.putText(frame, detail, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def _draw_status_panel(frame, left: dict | None, right: dict | None, mouth: dict | None, head: dict | None):
    """A single fixed panel for all per-cue readouts, so labels never overlap regardless of face size/position."""
    lines = []
    if left:
        lines.append((f"left eye: {left['label']} {left['closedness'] * 100:.0f}% | EAR {left['ear']:.2f}", _OPEN_COLOR if left["label"] == "open" else _CLOSED_COLOR))
    if right:
        lines.append((f"right eye: {right['label']} {right['closedness'] * 100:.0f}% | EAR {right['ear']:.2f}", _OPEN_COLOR if right["label"] == "open" else _CLOSED_COLOR))
    if mouth:
        lines.append((f"mouth: {mouth['label']} {mouth['openness'] * 100:.0f}% | MAR {mouth['mar']:.2f}", _CLOSED_COLOR if mouth["label"] == "yawn" else _MOUTH_COLOR))
    if head:
        lines.append((f"head: {head['label']} | pitch {head['pitch']:.1f} deg", _HEAD_COLOR if head["label"] == "upright" else _CLOSED_COLOR))
    if not lines:
        return

    panel_y = 88
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, panel_y), (280, panel_y + len(lines) * 22 + 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (18, panel_y + 22 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def main():
    config = load_config()
    image_size = tuple(config["preprocessing"]["image_size"])
    class_names = config["model"]["class_names"]
    yawn_class_names = config["yawn_model"]["class_names"]
    alertness_cfg = config["alertness"]
    ear_cfg = config["ear"]
    calibration_cfg = config["calibration"]
    mar_cfg = config["mar"]
    yawn_cfg = config["yawn"]
    head_pose_cfg = config["head_pose"]
    nod_cfg = config["nod"]

    model = load_trained_model(config)
    yawn_model = load_trained_model(config, checkpoint_name="best_yawn_model.keras")

    eye_monitor = _build_monitor(alertness_cfg)
    yawn_monitor = _build_monitor(yawn_cfg)
    nod_monitor = _build_monitor(nod_cfg)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError("Could not open the camera.")

    prev_time = time.time()

    def run_calibration():
        nonlocal ear_cfg
        calibrated = _calibrate_ear(cap, tracker, calibration_cfg["duration_sec"], calibration_cfg["closed_fraction"])
        if calibrated is not None:
            ear_cfg = calibrated
            print(f"Calibrated EAR thresholds: open>={ear_cfg['open_threshold']:.2f}, closed<={ear_cfg['closed_threshold']:.2f}")
        else:
            print("Calibration failed (no face detected) -- using default EAR thresholds.")

    with EyeRegionTracker(static_image_mode=False) as tracker:
        run_calibration()
        no_face_since = None

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)
                regions = tracker.extract(frame)

                if regions is None:
                    if no_face_since is None:
                        no_face_since = time.time()
                elif no_face_since is not None:
                    # a face reappearing after a real gap likely means a new driver, so recalibrate for them automatically
                    gap = time.time() - no_face_since
                    no_face_since = None
                    if gap >= _NO_FACE_RECALIBRATE_GAP_SEC:
                        run_calibration()
                        continue

                if regions is not None:
                    eye_results = {
                        side: _draw_eye(frame, side, eye, regions.landmarks, model, image_size, class_names, ear_cfg)
                        for side, eye in regions.eyes.items()
                    }
                    # average both eyes so one drooping eyelid still raises the score
                    eye_closedness = sum(r["closedness"] for r in eye_results.values()) / len(eye_results)
                    mouth_result = _draw_mouth(frame, regions.mouth, regions.landmarks, yawn_model, image_size, yawn_class_names, mar_cfg)
                    mouth_openness = mouth_result["openness"]
                    head_result = _head_pose_status(frame, regions.pitch, regions.landmarks, head_pose_cfg)
                    head_droop = head_result["droop"] if head_result else None
                    _draw_status_panel(frame, eye_results.get("left"), eye_results.get("right"), mouth_result, head_result)
                else:
                    eye_closedness = None
                    mouth_openness = None
                    head_droop = None

                now = time.time()
                eye_status = eye_monitor.update(eye_closedness, now)
                yawn_status = yawn_monitor.update(mouth_openness, now)
                nod_status = nod_monitor.update(head_droop, now)

                if eye_status.state == "no_face":
                    overall_state = "no_face"
                elif "not_alert" in (eye_status.state, yawn_status.state, nod_status.state):
                    overall_state = "not_alert"
                else:
                    overall_state = "alert"

                _draw_banner(frame, overall_state, eye_status, yawn_status, nod_status)

                fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
                prev_time = now
                cv2.putText(
                    frame, f"FPS: {fps:.1f}", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
                )

                cv2.imshow("FRAMES - Live Alertness Monitoring (press q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
