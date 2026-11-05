from types import SimpleNamespace

import cv2
import numpy as np

from watch.face import aligned_crop, aspect_ratio, brighten, crop_box, eye_line_roll


def _landmarks(points: dict[int, tuple[float, float]]):
    return {i: SimpleNamespace(x=x, y=y) for i, (x, y) in points.items()}


def test_aspect_ratio_averages_vertical_openings():
    landmarks = _landmarks({0: (0.0, 0.5), 1: (1.0, 0.5), 2: (0.4, 0.4), 3: (0.4, 0.6), 4: (0.6, 0.45), 5: (0.6, 0.55)})
    assert np.isclose(aspect_ratio(landmarks, 100, 100, ((0, 1), (2, 3), (4, 5))), 0.15)


def test_aspect_ratio_handles_zero_width():
    landmarks = _landmarks({0: (0.5, 0.5), 1: (0.5, 0.5), 2: (0.5, 0.4), 3: (0.5, 0.6)})
    assert aspect_ratio(landmarks, 100, 100, ((0, 1), (2, 3))) == 0.0


def test_square_crop_and_clipping():
    image = np.zeros((100, 200, 3), np.uint8)
    landmarks = _landmarks({0: (0.4, 0.5), 1: (0.6, 0.5), 2: (0.5, 0.45), 3: (0.5, 0.55)})
    crop, box = crop_box(image, landmarks, [0, 1, 2, 3], margin=0.0, square=True)
    assert box == (80, 30, 120, 70) and crop.shape[:2] == (40, 40)
    assert crop_box(image, _landmarks({0: (2.0, 2.0), 1: (3.0, 3.0)}), [0, 1], 0.0) is None


def test_aligned_crop_levels_a_tilted_eye():
    image = np.zeros((200, 200, 3), np.uint8)
    angle = np.radians(30)
    a, b = (100 - 40 * np.cos(angle), 100 - 40 * np.sin(angle)), (100 + 40 * np.cos(angle), 100 + 40 * np.sin(angle))
    cv2.line(image, tuple(map(int, a)), tuple(map(int, b)), (255, 255, 255), 3)
    landmarks = _landmarks({0: (a[0] / 200, a[1] / 200), 1: (b[0] / 200, b[1] / 200)})
    crop, box = aligned_crop(image, landmarks, [0, 1], (0, 1), margin=0.0)
    assert crop.shape[1] > 3 * crop.shape[0]
    assert box == crop_box(image, landmarks, [0, 1], 0.0)[1]
    assert aligned_crop(image, landmarks, [0, 1], (0, 1), 0.0, min_angle=45)[0].shape == crop_box(image, landmarks, [0, 1], 0.0)[0].shape


def test_aspect_ratio_uses_depth_when_the_face_turns():
    upright = _landmarks({0: (0.0, 0.5), 1: (1.0, 0.5), 2: (0.5, 0.4), 3: (0.5, 0.6)})
    turned = {i: SimpleNamespace(x=0.5 + (p.x - 0.5) * 0.5, y=p.y, z=(p.x - 0.5) * np.sqrt(0.75)) for i, p in upright.items()}
    points = ((0, 1), (2, 3))
    assert np.isclose(aspect_ratio(turned, 100, 100, points), aspect_ratio(upright, 100, 100, points))


def test_brighten_lifts_only_dark_frames():
    dark, normal = np.full((48, 64, 3), 30, np.uint8), np.full((48, 64, 3), 150, np.uint8)
    assert abs(brighten(dark, 0.3, 0.45).mean() / 255 - 0.45) < 0.01
    assert brighten(normal, 0.3, 0.45) is normal


def test_eye_line_roll_reads_tilt_and_never_flips():
    level = _landmarks({33: (0.3, 0.5), 263: (0.7, 0.5)})
    tilted = _landmarks({33: (0.3, 0.4), 263: (0.7, 0.6)})
    assert np.isclose(eye_line_roll(level, 100, 100), 0.0)
    assert np.isclose(eye_line_roll(tilted, 100, 100), np.degrees(np.arctan2(20, 40)))
    turned = _landmarks({33: (0.48, 0.5), 263: (0.52, 0.51)})
    assert eye_line_roll(turned, 100, 100) < 90
