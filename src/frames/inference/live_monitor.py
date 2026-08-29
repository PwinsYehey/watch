"""Live webcam alertness monitoring: tracks both eyes per frame, overlays classification + confidence, and raises a drowsy alert from sustained closure."""

import time
from pathlib import Path

import cv2
import tensorflow as tf

from frames.config import load_config
from frames.inference.alertness import AlertnessMonitor
from frames.inference.predict import predict_eye_state
from frames.preprocessing.facemesh import EyeRegion, EyeRegionTracker

_OPEN_COLOR = (0, 200, 0)
_CLOSED_COLOR = (0, 0, 220)
_NO_FACE_COLOR = (0, 165, 255)

_BANNER_COLORS = {
    "alert": (0, 200, 0),
    "drowsy": (0, 0, 220),
    "no_face": (0, 165, 255),
}
_BANNER_TEXT = {
    "alert": "ALERT",
    "drowsy": "DROWSY - WAKE UP",
    "no_face": "NO FACE DETECTED",
}


def _ear_to_closedness(ear: float, open_threshold: float, closed_threshold: float) -> float:
    """Maps EAR to a [0, 1] closedness score, 0 at open_threshold and 1 at closed_threshold."""
    span = open_threshold - closed_threshold
    if span <= 0:
        return 0.0
    score = (open_threshold - ear) / span
    return max(0.0, min(1.0, score))


def _draw_eye(frame, side: str, region: EyeRegion, model, image_size, class_names, ear_cfg) -> float:
    label, confidence, cnn_closedness = predict_eye_state(model, region.crop, image_size, class_names)
    ear_closedness = _ear_to_closedness(region.ear, ear_cfg["open_threshold"], ear_cfg["closed_threshold"])
    # either signal suggesting closure is enough -- a safety monitor should not need both to agree
    closedness = max(cnn_closedness, ear_closedness)

    color = _OPEN_COLOR if label == "open" else _CLOSED_COLOR
    x_min, y_min, x_max, y_max = region.box
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    text = f"{side}: {label} {confidence * 100:.0f}% | EAR {region.ear:.2f}"
    cv2.putText(frame, text, (x_min, max(y_min - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return closedness


def _draw_banner(frame, status):
    color = _BANNER_COLORS[status.state]
    text = _BANNER_TEXT[status.state]
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), color, -1)
    cv2.putText(frame, text, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    detail = f"closed {status.closed_duration:.1f}s | PERCLOS {status.perclos * 100:.0f}%"
    cv2.putText(frame, detail, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def main():
    config = load_config()
    image_size = tuple(config["preprocessing"]["image_size"])
    class_names = config["model"]["class_names"]
    alertness_cfg = config["alertness"]
    ear_cfg = config["ear"]

    model_path = Path(config["paths"]["models_dir"]) / "best_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found -- run `python cli/train.py` first.")
    model = tf.keras.models.load_model(model_path)

    monitor = AlertnessMonitor(
        closed_duration_threshold=alertness_cfg["closed_duration_threshold_sec"],
        perclos_window=alertness_cfg["perclos_window_sec"],
        perclos_threshold=alertness_cfg["perclos_threshold"],
        closedness_threshold=alertness_cfg["closedness_threshold"],
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError("Could not open the camera.")

    prev_time = time.time()

    with EyeRegionTracker(static_image_mode=False) as tracker:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)
                eyes = tracker.extract(frame)

                if eyes is not None:
                    closed_scores = {
                        side: _draw_eye(frame, side, region, model, image_size, class_names, ear_cfg)
                        for side, region in eyes.items()
                    }
                    # average both eyes so one drooping eyelid still raises the score
                    closedness = sum(closed_scores.values()) / len(closed_scores)
                else:
                    closedness = None

                status = monitor.update(closedness, time.time())
                _draw_banner(frame, status)

                now = time.time()
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
