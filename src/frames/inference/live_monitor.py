"""Live webcam eye-state monitoring: tracks both eyes per frame and overlays classification + confidence."""

import time
from pathlib import Path

import cv2
import tensorflow as tf

from frames.config import load_config
from frames.inference.predict import predict_eye_state
from frames.preprocessing.facemesh import EyeRegionTracker

_OPEN_COLOR = (0, 200, 0)
_CLOSED_COLOR = (0, 0, 220)
_NO_FACE_COLOR = (0, 0, 220)


def _draw_eye(frame, side: str, crop, box, model, image_size, class_names):
    label, confidence = predict_eye_state(model, crop, image_size, class_names)
    color = _OPEN_COLOR if label == "open" else _CLOSED_COLOR

    x_min, y_min, x_max, y_max = box
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    text = f"{side}: {label} {confidence * 100:.0f}%"
    cv2.putText(frame, text, (x_min, max(y_min - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def main():
    config = load_config()
    image_size = tuple(config["preprocessing"]["image_size"])
    class_names = config["model"]["class_names"]

    model_path = Path(config["paths"]["models_dir"]) / "best_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found -- run `python cli/train.py` first.")
    model = tf.keras.models.load_model(model_path)

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
                    for side, (crop, box) in eyes.items():
                        _draw_eye(frame, side, crop, box, model, image_size, class_names)
                else:
                    cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _NO_FACE_COLOR, 2)

                now = time.time()
                fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
                prev_time = now
                cv2.putText(
                    frame, f"FPS: {fps:.1f}", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
                )

                cv2.imshow("FRAMES - Live Eye Monitoring (press q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
