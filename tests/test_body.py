from types import SimpleNamespace

from watch.body import BodyPose, body_slump, shoulders

SETTINGS = {"drop_limit": 0.5, "lean_limit_deg": 20.0, "head_drop_limit": 0.25}
BASE = BodyPose(center_x=320, center_y=400, width=200, tilt=2.0, head=-0.6)


def test_upright_and_small_movements_are_not_slumped():
    assert body_slump(BASE, BASE, SETTINGS)[0] == 0.0
    assert body_slump(BodyPose(320, 440, 200, 8.0), BASE, SETTINGS)[0] < 1.0


def test_dropped_or_leaning_shoulders_are_slumped():
    score, drop, _, _ = body_slump(BodyPose(320, 510, 200, 2.0), BASE, SETTINGS)
    assert score == 1.0 and drop > 0.5
    score, _, lean, _ = body_slump(BodyPose(320, 400, 200, 25.0), BASE, SETTINGS)
    assert score == 1.0 and lean == 23.0


def test_sitting_up_straighter_is_not_a_slump():
    assert body_slump(BodyPose(320, 300, 200, 2.0), BASE, SETTINGS)[0] == 0.0


def test_shoulders_need_both_visible():
    landmarks = [SimpleNamespace(x=0.5, y=0.5, visibility=1.0)] * 33
    landmarks = list(landmarks)
    landmarks[11] = SimpleNamespace(x=0.6, y=0.7, visibility=0.9)
    landmarks[12] = SimpleNamespace(x=0.4, y=0.7, visibility=0.2)
    assert shoulders(landmarks, 640, 480, 0.5) is None
    landmarks[12] = SimpleNamespace(x=0.4, y=0.7, visibility=0.9)
    pose = shoulders(landmarks, 640, 480, 0.5)
    assert round(pose.width) == 128 and round(pose.tilt) == 0


def test_head_dropping_toward_the_shoulders_is_slumped_even_with_level_shoulders():
    score, _, _, head_drop = body_slump(BodyPose(320, 400, 200, 2.0, head=-0.3), BASE, SETTINGS)
    assert score == 1.0 and round(head_drop, 2) == 0.3
    assert body_slump(BodyPose(320, 400, 200, 2.0, head=-0.55), BASE, SETTINGS)[0] < 1.0
    assert body_slump(BodyPose(320, 400, 200, 2.0), BASE, SETTINGS)[3] is None


def test_raised_arm_tilt_is_ignored_but_a_real_drop_still_counts():
    tilted = BodyPose(320, 400, 200, 25.0, arm_raised=True)
    score, _, lean, _ = body_slump(tilted, BASE, SETTINGS)
    assert lean == 23.0 and score < 1.0
    assert body_slump(BodyPose(320, 510, 200, 25.0, arm_raised=True), BASE, SETTINGS)[0] == 1.0


def test_arm_counts_as_raised_when_a_wrist_or_elbow_is_above_its_shoulder():
    landmarks = [SimpleNamespace(x=0.5, y=0.9, visibility=1.0) for _ in range(33)]
    landmarks[11] = SimpleNamespace(x=0.6, y=0.7, visibility=0.9)
    landmarks[12] = SimpleNamespace(x=0.4, y=0.7, visibility=0.9)
    assert not shoulders(landmarks, 640, 480, 0.5).arm_raised
    landmarks[16] = SimpleNamespace(x=0.4, y=0.3, visibility=0.9)
    assert shoulders(landmarks, 640, 480, 0.5).arm_raised
    landmarks[16] = SimpleNamespace(x=0.4, y=0.3, visibility=0.1)
    assert not shoulders(landmarks, 640, 480, 0.5).arm_raised
