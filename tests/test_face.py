from types import SimpleNamespace

import numpy as np

from watch.face import aspect_ratio, crop_box


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
