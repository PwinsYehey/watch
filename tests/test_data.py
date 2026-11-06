from pathlib import Path

from watch.data.extract import parse_nthu, parse_yawdd
from watch.data.manifest import build_cew, build_mrl_eye, build_nthu_eye, build_yawdd, build_yawdd_eye


def _touch(root: Path, *names: str) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def test_parse_nthu():
    assert parse_nthu(Path("001_glasses_sleepyCombination_1420_drowsy.jpg")) == ("001", "glasses", "sleepyCombination", "1420")
    assert parse_nthu(Path("bad_name.jpg")) is None


def test_parse_yawdd_keeps_condition_so_crops_do_not_collide():
    glasses = parse_yawdd(Path("9-MaleGlasses-Yawning.avi"))
    no_glasses = parse_yawdd(Path("9-MaleNoGlasses-Yawning.avi"))
    assert glasses == ("9", "Glasses", "yawning")
    assert no_glasses == ("9", "NoGlasses", "yawning")


def test_mrl_labels_from_filename(tmp_path):
    _touch(tmp_path, "train/awake/s0001_00001_0_0_1_0_0_01.png", "train/sleepy/s0002_00002_0_0_0_0_0_01.png", "junk.png")
    rows = build_mrl_eye(tmp_path).set_index("subject_id")["label"].to_dict()
    assert rows == {"s0001": "eyes_opened", "s0002": "eyes_closed"}


def test_cew_groups_left_and_right_eye(tmp_path):
    _touch(tmp_path, "closedLeftEyes/face_1_L.jpg", "closedRightEyes/face_1_R.jpg", "openLeftEyes/x_L.jpg")
    rows = build_cew(tmp_path)
    assert rows.groupby("subject_id")["label"].first().to_dict() == {"face_1": "eyes_closed", "x": "eyes_opened"}


def test_folder_labelled_datasets(tmp_path):
    _touch(tmp_path, "open/001_glasses_a_1_left.jpg", "closed/002_glasses_a_1_right.jpg")
    assert sorted(build_nthu_eye(tmp_path)["label"]) == ["eyes_closed", "eyes_opened"]
    _touch(tmp_path, "yawn/male9_Glasses_yawning_00003.jpg", "no_yawn/female1_NoGlasses_normal_00000.jpg")
    assert build_yawdd(tmp_path).set_index("subject_id")["label"].to_dict() == {"male9": "yawn", "female1": "no_yawn"}


def test_blocked_crops_keep_their_subject(tmp_path):
    _touch(tmp_path, "mrl/blocked/s0003_b00001_0_0_b_0_0_01.png", "cew/blockedEyes/face_2_L.jpg", "yawdd/blocked/female17_sun_x_00001_left.png")
    assert build_mrl_eye(tmp_path / "mrl").set_index("subject_id")["label"].to_dict() == {"s0003": "blocked"}
    assert build_cew(tmp_path / "cew").set_index("subject_id")["label"].to_dict() == {"face_2": "blocked"}
    assert build_yawdd_eye(tmp_path / "yawdd").set_index("subject_id")["label"].to_dict() == {"female17": "blocked"}
