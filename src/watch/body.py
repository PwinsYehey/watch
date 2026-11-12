"""Upper-body posture from the shoulders, relative to the posture measured during calibration."""

import math
from dataclasses import dataclass

import cv2
import mediapipe as mp

NOSE, LEFT_SHOULDER, RIGHT_SHOULDER = 0, 11, 12
# Nose, eyes, ears and mouth; under a helmet only some stay visible, so the head is their average.
HEAD_POINTS = range(0, 11)


@dataclass
class BodyPose:
    center_x: float  # shoulder midpoint, pixels
    center_y: float
    width: float  # shoulder to shoulder, pixels
    tilt: float  # shoulder line angle, degrees, positive when the right shoulder is lower
    left: tuple[float, float] | None = None  # shoulder points in pixels, for drawing
    right: tuple[float, float] | None = None
    head: float | None = None


class BodyTracker:
    """Runs MediaPipe Pose every few frames, since posture changes slowly and the model costs about as much as face tracking."""

    def __init__(self, settings: dict):
        self._pose = mp.solutions.pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self._every, self._min_visibility = settings["every_n_frames"], settings["min_visibility"]
        self._frame, self._last = 0, None

    def update(self, image_bgr) -> BodyPose | None:
        """The latest shoulder pose, or None when no person's shoulders are in view."""
        self._frame += 1
        if self._frame % self._every == 1 or self._every == 1:
            results = self._pose.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            self._last = shoulders(results.pose_landmarks.landmark, *image_bgr.shape[1::-1], self._min_visibility) if results.pose_landmarks else None
        return self._last

    def close(self):
        self._pose.close()


def shoulders(landmarks, w: int, h: int, min_visibility: float) -> BodyPose | None:
    left, right = landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER]
    if min(left.visibility, right.visibility) < min_visibility:
        return None
    lx, ly, rx, ry = left.x * w, left.y * h, right.x * w, right.y * h
    width = math.hypot(lx - rx, ly - ry)
    if width < 1:
        return None
    head_ys = [landmarks[i].y * h for i in HEAD_POINTS if landmarks[i].visibility >= min_visibility]
    head = (sum(head_ys) / len(head_ys) - (ly + ry) / 2) / width if head_ys else None
    return BodyPose((lx + rx) / 2, (ly + ry) / 2, width, math.degrees(math.atan2(ry - ly, abs(rx - lx))), (lx, ly), (rx, ry), head)


def body_slump(pose: BodyPose, baseline: BodyPose, settings: dict) -> tuple[float, float, float, float | None, float]:
    """(score from 0 upright to 1 slumped, shoulder drop in shoulder widths, extra lean in degrees, head drop toward the shoulders,
    forward lean as the fraction the shoulders look wider) against the calibrated posture."""
    drop = max(0.0, (pose.center_y - baseline.center_y) / baseline.width)
    lean = abs(pose.tilt - baseline.tilt)
    head_drop = max(0.0, pose.head - baseline.head) if pose.head is not None and baseline.head is not None else None
    # Slumping toward the camera brings the shoulders closer, so they look wider.
    forward = max(0.0, pose.width / baseline.width - 1.0)
    score = max(drop / settings["drop_limit"], lean / settings["lean_limit_deg"], (head_drop or 0.0) / settings["head_drop_limit"], forward / settings["forward_limit"])
    return min(1.0, score), drop, lean, head_drop, forward
