from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

_mp_face_mesh = mp.solutions.face_mesh

# Landmark indices bounding each eye region
_LEFT_EYE_IDS = [33, 133, 159, 145, 160, 144, 158, 153]
_RIGHT_EYE_IDS = [362, 263, 386, 374, 385, 380, 387, 373]

# The same eye landmarks organized into the classic 6-point EAR layout (Soukupova & Cech).
_LEFT_EYE_EAR_POINTS = {"horizontal": (33, 133), "vertical_a": (160, 144), "vertical_b": (158, 153)}
_RIGHT_EYE_EAR_POINTS = {"horizontal": (362, 263), "vertical_a": (385, 380), "vertical_b": (387, 373)}

# Landmark indices bounding the mouth region, and the MAR layout.
_MOUTH_IDS = [61, 291, 13, 14, 78, 308, 0, 17]
_MOUTH_MAR_POINTS = {"horizontal": (61, 291), "vertical": (13, 14)}

# Head-pose landmarks paired with a generic 3D face model for solvePnP.
_HEAD_POSE_IDS = [1, 152, 33, 263, 61, 291]
_HEAD_POSE_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),        # nose tip
        (0.0, -330.0, -65.0),   # chin
        (-225.0, 170.0, -135.0),  # eye corner
        (225.0, 170.0, -135.0),   # eye corner
        (-150.0, -150.0, -125.0),  # mouth corner
        (150.0, -150.0, -125.0),   # mouth corner
    ],
    dtype=np.float64,
)


@dataclass
class EyeRegion:
    crop: np.ndarray
    box: tuple[int, int, int, int]
    ear: float


@dataclass
class MouthRegion:
    crop: np.ndarray
    box: tuple[int, int, int, int]
    mar: float


@dataclass
class FaceRegions:
    eyes: dict[str, EyeRegion]
    mouth: MouthRegion
    pitch: float | None
    roll: float | None
    landmarks: object


def _crop_from_landmarks(image: np.ndarray, landmarks, ids: list[int], margin: float, square: bool = False):
    """Returns (crop, (x_min, y_min, x_max, y_max)), or None if the box is degenerate."""
    h, w = image.shape[:2]
    xs = [landmarks[i].x * w for i in ids]
    ys = [landmarks[i].y * h for i in ids]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    box_w, box_h = x_max - x_min, y_max - y_min

    x_min -= box_w * margin
    x_max += box_w * margin
    y_min -= box_h * margin
    y_max += box_h * margin

    if square:
        # pad the short side instead of stretching -- the eye CNN's training data is framed square
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        side = max(x_max - x_min, y_max - y_min)
        x_min, x_max = cx - side / 2, cx + side / 2
        y_min, y_max = cy - side / 2, cy + side / 2

    x_min = max(int(x_min), 0)
    x_max = min(int(x_max), w)
    y_min = max(int(y_min), 0)
    y_max = min(int(y_max), h)

    if x_max <= x_min or y_max <= y_min:
        return None
    return image[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)


def _point(landmarks, index: int, w: int, h: int) -> np.ndarray:
    return np.array([landmarks[index].x * w, landmarks[index].y * h])


def _eye_aspect_ratio(landmarks, w: int, h: int, ear_points: dict) -> float:
    """Lower EAR means more closed; ~0.30 wide open, ~0.20 or below closed (Soukupova & Cech)."""
    h1, h2 = ear_points["horizontal"]
    va1, va2 = ear_points["vertical_a"]
    vb1, vb2 = ear_points["vertical_b"]

    horizontal = np.linalg.norm(_point(landmarks, h1, w, h) - _point(landmarks, h2, w, h))
    vertical_a = np.linalg.norm(_point(landmarks, va1, w, h) - _point(landmarks, va2, w, h))
    vertical_b = np.linalg.norm(_point(landmarks, vb1, w, h) - _point(landmarks, vb2, w, h))

    if horizontal == 0:
        return 0.0
    return (vertical_a + vertical_b) / (2.0 * horizontal)


def _head_pose(landmarks, w: int, h: int) -> tuple[float, float] | None:
    """Head pitch (forward nod) and roll (sideways tilt) in degrees via solvePnP, or None if the pose couldn't be estimated."""
    image_points = np.array([_point(landmarks, i, w, h) for i in _HEAD_POSE_IDS], dtype=np.float64)
    focal_length = w
    camera_matrix = np.array(
        [[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]], dtype=np.float64
    )
    dist_coeffs = np.zeros((4, 1))

    success, rvec, _tvec = cv2.solvePnP(_HEAD_POSE_MODEL_POINTS, image_points, camera_matrix, dist_coeffs)
    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    raw_pitch = float(angles[0])
    # solvePnP's pitch reads near +-180 for a neutral head; roll has no such offset and reads near 0 already
    pitch = ((raw_pitch + 360.0) % 360.0) - 180.0
    roll = abs(float(angles[2]))
    return pitch, roll


def _mouth_aspect_ratio(landmarks, w: int, h: int) -> float:
    """Higher MAR means more open; a wide, sustained MAR is what distinguishes yawning from talking."""
    h1, h2 = _MOUTH_MAR_POINTS["horizontal"]
    v1, v2 = _MOUTH_MAR_POINTS["vertical"]

    horizontal = np.linalg.norm(_point(landmarks, h1, w, h) - _point(landmarks, h2, w, h))
    vertical = np.linalg.norm(_point(landmarks, v1, w, h) - _point(landmarks, v2, w, h))

    if horizontal == 0:
        return 0.0
    return vertical / horizontal


class EyeRegionTracker:
    """Reusable MediaPipe FaceMesh instance for per-frame eye/mouth tracking, avoiding the cost of re-initializing the model on every frame."""

    def __init__(self, static_image_mode: bool = False):
        self._face_mesh = _mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, image_bgr: np.ndarray) -> FaceRegions | None:
        """Returns a FaceRegions with both eyes, the mouth, and head pitch, or None if no face could be located."""
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        left = _crop_from_landmarks(image_bgr, landmarks, _LEFT_EYE_IDS, margin=0.15, square=True)
        right = _crop_from_landmarks(image_bgr, landmarks, _RIGHT_EYE_IDS, margin=0.15, square=True)
        mouth = _crop_from_landmarks(image_bgr, landmarks, _MOUTH_IDS, margin=0.3)

        if left is None or right is None or mouth is None:
            return None

        left_crop, left_box = left
        right_crop, right_box = right
        mouth_crop, mouth_box = mouth
        left_ear = _eye_aspect_ratio(landmarks, w, h, _LEFT_EYE_EAR_POINTS)
        right_ear = _eye_aspect_ratio(landmarks, w, h, _RIGHT_EYE_EAR_POINTS)
        mar = _mouth_aspect_ratio(landmarks, w, h)
        pose = _head_pose(landmarks, w, h)
        pitch, roll = pose if pose is not None else (None, None)

        return FaceRegions(
            eyes={
                "left": EyeRegion(crop=left_crop, box=left_box, ear=left_ear),
                "right": EyeRegion(crop=right_crop, box=right_box, ear=right_ear),
            },
            mouth=MouthRegion(crop=mouth_crop, box=mouth_box, mar=mar),
            pitch=pitch,
            roll=roll,
            landmarks=landmarks,
        )

    def close(self):
        self._face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


def extract_face_regions(image_bgr: np.ndarray) -> FaceRegions | None:
    """Convenience wrapper for a single standalone image: returns a FaceRegions, or None."""
    with EyeRegionTracker(static_image_mode=True) as tracker:
        return tracker.extract(image_bgr)
