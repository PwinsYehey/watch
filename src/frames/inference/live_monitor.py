import time

import cv2

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


def _draw_eye(frame, side: str, region: EyeRegion, model, image_size, class_names, ear_cfg) -> float:
    _cnn_label, _cnn_confidence, cnn_closedness = predict_eye_state(model, region.crop, image_size, class_names)
    ear_closedness = _linear_score(region.ear, ear_cfg["open_threshold"], ear_cfg["closed_threshold"])
    # either signal suggesting closure is enough -- a safety monitor should not need both to agree
    closedness = max(cnn_closedness, ear_closedness)
    # the displayed label always matches this combined score, not just the CNN's own opinion
    label = "closed" if closedness >= 0.5 else "open"

    color = _OPEN_COLOR if label == "open" else _CLOSED_COLOR
    x_min, y_min, x_max, y_max = region.box
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    text = f"{side}: {label} {closedness * 100:.0f}% | EAR {region.ear:.2f}"
    cv2.putText(frame, text, (x_min, max(y_min - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return closedness


def _draw_mouth(frame, region: MouthRegion, yawn_model, image_size, yawn_class_names, mar_cfg) -> float:
    _cnn_label, _cnn_confidence, cnn_openness = predict_yawn_state(yawn_model, region.crop, image_size, yawn_class_names)
    mar_openness = _linear_score(region.mar, mar_cfg["closed_threshold"], mar_cfg["open_threshold"])
    # either signal suggesting a yawn is enough -- a safety monitor should not need both to agree
    openness = max(cnn_openness, mar_openness)
    # the displayed label always matches this combined score, not just the CNN's own opinion
    label = "yawn" if openness >= 0.5 else "no yawn"

    color = _CLOSED_COLOR if label == "yawn" else _MOUTH_COLOR
    x_min, y_min, x_max, y_max = region.box
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    text = f"mouth: {label} {openness * 100:.0f}% | MAR {region.mar:.2f}"
    cv2.putText(frame, text, (x_min, max(y_min - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return openness


def _draw_head_pose(frame, pitch: float | None, head_pose_cfg) -> float | None:
    if pitch is None:
        return None
    droop = _linear_score(pitch, head_pose_cfg["upright_threshold"], head_pose_cfg["drooped_threshold"])
    label = "nodding" if droop >= 0.5 else "upright"
    color = _HEAD_COLOR if label == "upright" else _CLOSED_COLOR
    cv2.putText(
        frame, f"head: {label} | pitch {pitch:.1f} deg", (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
    )
    return droop


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


def main():
    config = load_config()
    image_size = tuple(config["preprocessing"]["image_size"])
    class_names = config["model"]["class_names"]
    yawn_class_names = config["yawn_model"]["class_names"]
    alertness_cfg = config["alertness"]
    ear_cfg = config["ear"]
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

    with EyeRegionTracker(static_image_mode=False) as tracker:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)
                regions = tracker.extract(frame)

                if regions is not None:
                    closed_scores = {
                        side: _draw_eye(frame, side, eye, model, image_size, class_names, ear_cfg)
                        for side, eye in regions.eyes.items()
                    }
                    # average both eyes so one drooping eyelid still raises the score
                    eye_closedness = sum(closed_scores.values()) / len(closed_scores)
                    mouth_openness = _draw_mouth(frame, regions.mouth, yawn_model, image_size, yawn_class_names, mar_cfg)
                    head_droop = _draw_head_pose(frame, regions.pitch, head_pose_cfg)
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
