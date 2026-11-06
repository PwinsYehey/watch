from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from watch.alertness import RollingMedian
from watch.config import load_config
from watch.data.manifest import build_own_eye
from watch.finetune import split_takes
from watch.live import LiveMonitor, ear_thresholds, eye_closedness
from watch.record import clean_name, crop_path

CALIBRATION = {"closed_offset": 0.10, "min_open_ear": 0.20}


def _record_crops(root, takes, per_label=3):
    for subject, take in takes:
        for label in ("open", "closed"):
            for i in range(per_label):
                path = crop_path(root, label, subject, take, "left", i)
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), np.zeros((8, 8, 3), np.uint8))


def test_clean_name_keeps_letters_and_digits_only():
    assert clean_name("Yeri_Shades 1!") == "yerishades1"
    with pytest.raises(ValueError):
        clean_name("__")


def test_own_eye_rows_carry_label_subject_and_take(tmp_path):
    _record_crops(tmp_path, [("yeri", "shades1")])
    rows = build_own_eye(tmp_path)
    assert len(rows) == 6
    assert set(rows["label"]) == {"eyes_opened", "eyes_closed"}
    assert set(rows["subject_id"]) == {"yeri"}
    assert set(rows["take_id"]) == {"yeri_shades1"}


def test_split_holds_out_whole_takes(tmp_path):
    _record_crops(tmp_path, [("yeri", "shades1"), ("yeri", "shades2"), ("ana", "shades1"), ("ana", "noshades1")])
    rows = split_takes(build_own_eye(tmp_path), test_fraction=0.25, seed=42)
    assert rows.groupby("take_id")["split"].nunique().max() == 1
    assert rows.loc[rows["split"] == "test", "take_id"].nunique() == 1
    assert set(rows.loc[rows["split"] == "test", "label"]) == {"eyes_opened", "eyes_closed"}


def test_split_can_name_the_test_takes(tmp_path):
    _record_crops(tmp_path, [("yeri", "shades1"), ("yeri", "shades2"), ("yeri", "noshades1")])
    rows = split_takes(build_own_eye(tmp_path), test_fraction=0.25, seed=42, test_takes=["yeri_shades2"])
    assert set(rows.loc[rows["split"] == "test", "take_id"]) == {"yeri_shades2"}
    with pytest.raises(ValueError):
        split_takes(build_own_eye(tmp_path), test_fraction=0.25, seed=42, test_takes=["yeri_missing"])


def test_split_needs_two_takes(tmp_path):
    _record_crops(tmp_path, [("yeri", "shades1")])
    with pytest.raises(ValueError):
        split_takes(build_own_eye(tmp_path), test_fraction=0.25, seed=42)


def test_shape_gate_lets_either_the_shape_or_the_cnn_close_the_eye():
    assert eye_closedness(shape_closed=True, glare=False, cnn_closed=0.2) == 1.0
    assert eye_closedness(shape_closed=False, glare=False, cnn_closed=0.3) == 0.3
    assert eye_closedness(shape_closed=True, glare=True, cnn_closed=0.2) == 0.2


def test_cnn_mode_trusts_the_cnn_alone():
    assert eye_closedness(shape_closed=False, glare=False, cnn_closed=0.9, mode="cnn") == 0.9
    assert eye_closedness(shape_closed=True, glare=False, cnn_closed=0.1, mode="cnn") == 0.1


def _monitor(mode):
    config = load_config()
    monitor = object.__new__(LiveMonitor)
    monitor.config, monitor.settings = config, {**config["live"], "eye_decision": mode}
    monitor.ear = {"open_threshold": 0.30, "closed_threshold": 0.20}
    monitor.smoothers = {"left": RollingMedian(1)}
    return monitor


def test_cnn_mode_still_reads_a_closed_eye_whose_shape_looks_open():
    eye, frame = SimpleNamespace(ratio=0.29, box=(10, 10, 30, 20)), np.full((100, 100, 3), 120, np.uint8)
    assert _monitor("shape_gate")._eye(frame, "left", eye, ("eyes_closed", 0.9, 0.9)) is None
    assert _monitor("cnn")._eye(frame, "left", eye, ("eyes_closed", 0.9, 0.9))["label"] == "closed"


def test_calibration_uses_open_eyes():
    thresholds = ear_thresholds([0.28] * 12, CALIBRATION)
    assert thresholds["open_threshold"] == pytest.approx(0.28)
    assert thresholds["closed_threshold"] == pytest.approx(0.18)


def test_calibration_ignores_hidden_eyes_and_too_few_readings():
    assert ear_thresholds([0.15] * 12, CALIBRATION) is None
    assert ear_thresholds([0.28] * 5, CALIBRATION) is None
