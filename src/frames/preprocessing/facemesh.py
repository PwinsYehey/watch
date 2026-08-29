"""Locates and crops eye regions from a full face image via MediaPipe FaceMesh, bridging pre-cropped training data with real full-face deployment input."""

import cv2
import mediapipe as mp
import numpy as np

_mp_face_mesh = mp.solutions.face_mesh

# Landmark indices bounding each eye region (MediaPipe FaceMesh topology).
_LEFT_EYE_IDS = [33, 133, 159, 145, 160, 144, 158, 153]
_RIGHT_EYE_IDS = [362, 263, 386, 374, 385, 380, 387, 373]


def _crop_from_landmarks(image: np.ndarray, landmarks, ids: list[int], margin: float = 0.6):
    """Returns (crop, (x_min, y_min, x_max, y_max)), or None if the box is degenerate."""
    h, w = image.shape[:2]
    xs = [landmarks[i].x * w for i in ids]
    ys = [landmarks[i].y * h for i in ids]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    box_w, box_h = x_max - x_min, y_max - y_min

    x_min = max(int(x_min - box_w * margin), 0)
    x_max = min(int(x_max + box_w * margin), w)
    y_min = max(int(y_min - box_h * margin), 0)
    y_max = min(int(y_max + box_h * margin), h)

    if x_max <= x_min or y_max <= y_min:
        return None
    return image[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)


class EyeRegionTracker:
    """Reusable MediaPipe FaceMesh instance for per-frame eye tracking, avoiding the cost of re-initializing the model on every frame."""

    def __init__(self, static_image_mode: bool = False):
        self._face_mesh = _mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, image_bgr: np.ndarray):
        """Returns {'left': (crop, box), 'right': (crop, box)}, or None if no face/eyes could be located."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        left = _crop_from_landmarks(image_bgr, landmarks, _LEFT_EYE_IDS)
        right = _crop_from_landmarks(image_bgr, landmarks, _RIGHT_EYE_IDS)

        if left is None or right is None:
            return None
        return {"left": left, "right": right}

    def close(self):
        self._face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


def extract_eye_crops(image_bgr: np.ndarray):
    """Convenience wrapper for a single standalone image: returns {'left': (crop, box), 'right': (crop, box)}, or None."""
    with EyeRegionTracker(static_image_mode=True) as tracker:
        return tracker.extract(image_bgr)
