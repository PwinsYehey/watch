"""Upper-body posture from the shoulders, relative to the posture measured during calibration."""

import math
from dataclasses import dataclass

import cv2
import mediapipe as mp

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12


@dataclass
class BodyPose:
    center_x: float  # shoulder midpoint, pixels
    center_y: float
    width: float  # shoulder to shoulder, pixels
    tilt: float  # shoulder line angle, degrees, positive when the right shoulder is lower
    left: tuple[float, float] | None = None  # shoulder points in pixels, for drawing
    right: tuple[float, float] | None = None


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
    return BodyPose((lx + rx) / 2, (ly + ry) / 2, width, math.degrees(math.atan2(ry - ly, abs(rx - lx))), (lx, ly), (rx, ry))


def body_slump(pose: BodyPose, baseline: BodyPose, settings: dict) -> tuple[float, float, float]:
    """(score from 0 upright to 1 slumped, shoulder drop in shoulder widths, extra lean in degrees) against the calibrated posture."""
    drop = max(0.0, (pose.center_y - baseline.center_y) / baseline.width)
    lean = abs(pose.tilt - baseline.tilt)
    score = max(drop / settings["drop_limit"], lean / settings["lean_limit_deg"])
    return min(1.0, score), drop, lean
