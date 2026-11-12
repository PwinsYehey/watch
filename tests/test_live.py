from types import SimpleNamespace

import numpy as np

from watch.alertness import RollingMedian
from watch.config import load_config
from watch.live import LiveMonitor


class _Model:
    def __init__(self, positive):
        self.positive = positive

    def predict(self, crop):
        return "label", 0.9, self.positive


def _monitor(yawn_open=0.0, eye_decision="shape_gate"):
    config = load_config()
    config["live"]["eye_decision"] = eye_decision
    monitor = object.__new__(LiveMonitor)
    monitor.config, monitor.settings = config, config["live"]
    monitor.ear = {"open_threshold": 0.30, "closed_threshold": 0.20}
    monitor.yawn_model = _Model(yawn_open)
    monitor.smoothers = {key: RollingMedian(1) for key in ("left", "right")}
    return monitor


FRAME = np.full((100, 100, 3), 120, np.uint8)


def _eye(ratio):
    return SimpleNamespace(ratio=ratio, box=(10, 10, 30, 20))


def test_eye_read_closed_by_the_model_but_open_by_shape_is_unreadable():
    assert _monitor()._eye(FRAME, "left", _eye(0.29), ("eyes_closed", 0.9, 0.9)) is None


def test_eye_closed_by_model_and_shape_still_counts_as_closed():
    cue = _monitor()._eye(FRAME, "left", _eye(0.18), ("eyes_closed", 0.9, 0.9))
    assert cue["label"] == "closed"


def test_open_eye_reads_open():
    assert _monitor()._eye(FRAME, "left", _eye(0.30), ("eyes_opened", 0.9, 0.1))["label"] == "open"


def test_yawn_on_a_closed_mouth_means_the_mouth_is_covered():
    assert _monitor(yawn_open=0.95)._mouth(SimpleNamespace(crop=None, ratio=0.05)) is None


def test_real_yawn_still_counts():
    assert _monitor(yawn_open=0.95)._mouth(SimpleNamespace(crop=None, ratio=0.7))["label"] == "yawn"


def test_eye_the_model_calls_blocked_is_unreadable():
    assert _monitor()._eye(FRAME, "left", _eye(0.30), ("blocked", 0.9, 0.05)) is None


def test_screen_says_face_covered_only_when_a_person_is_still_in_view():
    monitor = _monitor()
    monitor.body_only = False
    assert monitor._idle_state(None, {"body": {"label": "upright"}}) == "face_covered"
    assert monitor._idle_state(None, {"body": None}) == "no_face"


def test_body_mode_shows_normal_with_a_person_and_no_person_without_one():
    monitor = _monitor()
    monitor.body_only = True
    assert monitor._idle_state(None, {"body": {"label": "upright"}}) == 0
    assert monitor._idle_state(None, {"body": None}) == "no_person"


def test_cnn_mode_trusts_the_eye_model_even_when_the_shape_looks_open():
    assert _monitor(eye_decision="cnn")._eye(FRAME, "left", _eye(0.29), ("eyes_closed", 0.9, 0.9))["label"] == "closed"


def test_mouth_the_model_calls_blocked_is_unreadable():
    monitor = _monitor()
    monitor.yawn_model = SimpleNamespace(predict=lambda crop: ("blocked", 0.95, 0.0))
    assert monitor._mouth(SimpleNamespace(crop=None, ratio=0.05)) is None
