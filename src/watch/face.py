from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

from watch.config import load_config

FACE_MESH = mp.solutions.face_mesh
FACE_DETECTION = mp.solutions.face_detection

LEFT_EYE_IDS = [33, 133, 159, 145, 160, 144, 158, 153]
RIGHT_EYE_IDS = [362, 263, 386, 374, 385, 380, 387, 373]
MOUTH_IDS = [61, 291, 13, 14, 78, 308, 0, 17]

# Each pair is (horizontal, vertical_a, vertical_b) for the 6-point EAR of Soukupova & Cech.
LEFT_EYE_EAR = ((33, 133), (160, 144), (158, 153))
RIGHT_EYE_EAR = ((362, 263), (385, 380), (387, 373))
MOUTH_MAR = ((61, 291), (13, 14))
LEFT_EYE_AXIS, RIGHT_EYE_AXIS, MOUTH_AXIS = (33, 133), (362, 263), (61, 291)

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
class Landmark:
    x: float
    y: float
    z: float


@dataclass
class Region:
    crop: np.ndarray
    box: tuple[int, int, int, int]
    ratio: float
    width: float


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


def aligned_crop(image: np.ndarray, landmarks, ids: list[int], axis: tuple[int, int], margin: float, square: bool = False, min_angle: float = 0.0):
    """Like crop_box, but rotated so the axis landmarks are level."""
    cropped = crop_box(image, landmarks, ids, margin, square)
    if cropped is None:
        return None
    h, w = image.shape[:2]
    (ax, ay), (bx, by) = ((landmarks[i].x * w, landmarks[i].y * h) for i in axis)
    angle = float(np.degrees(np.arctan2(by - ay, bx - ax)))
    if abs(angle) < min_angle:
        return cropped
    points = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in ids], dtype=np.float64)
    rotation = cv2.getRotationMatrix2D(tuple(points.mean(axis=0)), angle, 1.0)
    rotated = points @ rotation[:, :2].T + rotation[:, 2]

    (x_min, y_min), (x_max, y_max) = rotated.min(axis=0), rotated.max(axis=0)
    pad_x, pad_y = (x_max - x_min) * margin, (y_max - y_min) * margin
    x_min, x_max, y_min, y_max = x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y
    if square:
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        half = max(x_max - x_min, y_max - y_min) / 2
        x_min, x_max, y_min, y_max = cx - half, cx + half, cy - half, cy + half

    size = (max(int(round(x_max - x_min)), 1), max(int(round(y_max - y_min)), 1))
    rotation[:, 2] -= (x_min, y_min)
    return cv2.warpAffine(image, rotation, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), cropped[1]


def _distance(landmarks, a: int, b: int, w: int, h: int) -> float:
    # 3D keeps ratios steady when the face turns.
    pa, pb = landmarks[a], landmarks[b]
    dz = (getattr(pa, "z", 0.0) - getattr(pb, "z", 0.0)) * w
    return float(np.sqrt(((pa.x - pb.x) * w) ** 2 + ((pa.y - pb.y) * h) ** 2 + dz**2))


def _distance2d(landmarks, a: int, b: int, w: int, h: int) -> float:
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


def brighten(image: np.ndarray, dark_below: float, target: float) -> np.ndarray:
    """Gamma-lifts frames darker than dark_below to a mean brightness of target."""
    mean = float(cv2.cvtColor(cv2.resize(image, (64, 48)), cv2.COLOR_BGR2GRAY).mean()) / 255.0
    if mean >= dark_below or mean <= 0.0:
        return image
    gamma = np.log(target) / np.log(mean)
    return cv2.LUT(image, (np.linspace(0.0, 1.0, 256) ** gamma * 255.0).astype(np.uint8))


def is_low_detail(crop: np.ndarray, threshold: float) -> bool:
    return float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).std()) < threshold


class FaceTracker:
    def __init__(self, static_image_mode: bool = False):
        settings = load_config()["live"]
        self._mesh = FACE_MESH.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=settings["face_detection_confidence"],
            min_tracking_confidence=settings["face_tracking_confidence"],
        )
        # Fallback for faces the tracker misses, such as one far from the camera.
        self._detector = FACE_DETECTION.FaceDetection(model_selection=1, min_detection_confidence=settings["face_detection_confidence"])
        self._fallback_mesh = FACE_MESH.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=settings["face_detection_confidence"])
        self._low_detail_std = settings["low_detail_std"]
        self._low_light = (settings["low_light_mean"], settings["low_light_target"])
        self._level_min_angle = settings["level_min_angle_deg"]

    def _landmarks(self, image_rgb: np.ndarray):
        results = self._mesh.process(image_rgb)
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0].landmark
        detections = self._detector.process(image_rgb).detections
        if not detections:
            return None

        h, w = image_rgb.shape[:2]
        box = detections[0].location_data.relative_bounding_box
        cx, cy = (box.xmin + box.width / 2) * w, (box.ymin + box.height / 2) * h
        half = max(box.width * w, box.height * h)
        x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
        x1, y1 = int(min(cx + half, w)), int(min(cy + half, h))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        crop = cv2.resize(image_rgb[y0:y1, x0:x1], (256, int(256 * (y1 - y0) / (x1 - x0))))
        results = self._fallback_mesh.process(crop)
        if not results.multi_face_landmarks:
            return None
        cw, ch = x1 - x0, y1 - y0
        return [Landmark((x0 + p.x * cw) / w, (y0 + p.y * ch) / h, p.z * cw / w) for p in results.multi_face_landmarks[0].landmark]

    def _region(self, image, landmarks, ids, axis, margin, square, ratio_points) -> Region | None:
        cropped = aligned_crop(image, landmarks, ids, axis, margin, square, self._level_min_angle)
        if cropped is None or is_low_detail(cropped[0], self._low_detail_std):
            return None
        h, w = image.shape[:2]
        return Region(cropped[0], cropped[1], aspect_ratio(landmarks, w, h, ratio_points), _distance2d(landmarks, *axis, w, h))

    def extract(self, image_bgr: np.ndarray) -> FaceRegions | None:
        image_bgr = brighten(image_bgr, *self._low_light)
        landmarks = self._landmarks(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if landmarks is None:
            return None

        eyes = {
            side: region
            for side, ids, axis, points in (("left", LEFT_EYE_IDS, LEFT_EYE_AXIS, LEFT_EYE_EAR), ("right", RIGHT_EYE_IDS, RIGHT_EYE_AXIS, RIGHT_EYE_EAR))
            if (region := self._region(image_bgr, landmarks, ids, axis, 0.15, True, points)) is not None
        }
        mouth = self._region(image_bgr, landmarks, MOUTH_IDS, MOUTH_AXIS, 0.3, False, MOUTH_MAR)
        h, w = image_bgr.shape[:2]
        pitch, roll = head_pose(landmarks, w, h) or (None, None)
        return FaceRegions(eyes, mouth, pitch, roll, landmarks)

    def close(self):
        self._mesh.close()
        self._detector.close()
        self._fallback_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
