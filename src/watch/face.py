from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

from watch.config import load_config

FACE_MESH = mp.solutions.face_mesh

LEFT_EYE_IDS = [33, 133, 159, 145, 160, 144, 158, 153]
RIGHT_EYE_IDS = [362, 263, 386, 374, 385, 380, 387, 373]
MOUTH_IDS = [61, 291, 13, 14, 78, 308, 0, 17]

# Each pair is (horizontal, vertical_a, vertical_b) for the 6-point EAR of Soukupova & Cech.
LEFT_EYE_EAR = ((33, 133), (160, 144), (158, 153))
RIGHT_EYE_EAR = ((362, 263), (385, 380), (387, 373))
MOUTH_MAR = ((61, 291), (13, 14))

HEAD_POSE_IDS = [1, 152, 33, 263, 61, 291]
HEAD_POSE_MODEL = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)


@dataclass
class Region:
    crop: np.ndarray
    box: tuple[int, int, int, int]
    ratio: float


@dataclass
class FaceRegions:
    eyes: dict[str, Region]
    mouth: Region | None
    pitch: float | None
    roll: float | None
    landmarks: object


def crop_box(image: np.ndarray, landmarks, ids: list[int], margin: float, square: bool = False):
    h, w = image.shape[:2]
    xs = [landmarks[i].x * w for i in ids]
    ys = [landmarks[i].y * h for i in ids]
    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
    pad_x, pad_y = (x_max - x_min) * margin, (y_max - y_min) * margin
    x_min, x_max, y_min, y_max = x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y

    if square:
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        half = max(x_max - x_min, y_max - y_min) / 2
        x_min, x_max, y_min, y_max = cx - half, cx + half, cy - half, cy + half

    box = (max(int(x_min), 0), max(int(y_min), 0), min(int(x_max), w), min(int(y_max), h))
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return image[box[1] : box[3], box[0] : box[2]], box


def _distance(landmarks, a: int, b: int, w: int, h: int) -> float:
    return float(np.hypot((landmarks[a].x - landmarks[b].x) * w, (landmarks[a].y - landmarks[b].y) * h))


def aspect_ratio(landmarks, w: int, h: int, points: tuple) -> float:
    """Mean vertical opening over horizontal width, i.e. EAR for eyes and MAR for the mouth."""
    horizontal, *verticals = points
    width = _distance(landmarks, *horizontal, w, h)
    if width == 0:
        return 0.0
    return sum(_distance(landmarks, *pair, w, h) for pair in verticals) / (len(verticals) * width)


def head_pose(landmarks, w: int, h: int) -> tuple[float, float] | None:
    image_points = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in HEAD_POSE_IDS], dtype=np.float64)
    camera = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(HEAD_POSE_MODEL, image_points, camera, np.zeros((4, 1)))
    if not ok:
        return None
    angles, *_ = cv2.RQDecomp3x3(cv2.Rodrigues(rvec)[0])
    # solvePnP reads a neutral head near +-180 pitch, so shift it to read near 0.
    pitch = ((float(angles[0]) + 360.0) % 360.0) - 180.0
    return pitch, abs(float(angles[2]))


def is_low_detail(crop: np.ndarray, threshold: float) -> bool:
    return float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).std()) < threshold


class FaceTracker:
    def __init__(self, static_image_mode: bool = False):
        self._mesh = FACE_MESH.FaceMesh(static_image_mode=static_image_mode, max_num_faces=1, refine_landmarks=True)
        self._low_detail_std = load_config()["live"]["low_detail_std"]

    def _region(self, image, landmarks, ids, margin, square, ratio_points) -> Region | None:
        cropped = crop_box(image, landmarks, ids, margin, square)
        if cropped is None or is_low_detail(cropped[0], self._low_detail_std):
            return None
        h, w = image.shape[:2]
        return Region(cropped[0], cropped[1], aspect_ratio(landmarks, w, h, ratio_points))

    def extract(self, image_bgr: np.ndarray) -> FaceRegions | None:
        results = self._mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        eyes = {
            side: region
            for side, ids, points in (("left", LEFT_EYE_IDS, LEFT_EYE_EAR), ("right", RIGHT_EYE_IDS, RIGHT_EYE_EAR))
            if (region := self._region(image_bgr, landmarks, ids, 0.15, True, points)) is not None
        }
        mouth = self._region(image_bgr, landmarks, MOUTH_IDS, 0.3, False, MOUTH_MAR)
        h, w = image_bgr.shape[:2]
        pitch, roll = head_pose(landmarks, w, h) or (None, None)
        return FaceRegions(eyes, mouth, pitch, roll, landmarks)

    def close(self):
        self._mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
