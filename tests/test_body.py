from types import SimpleNamespace

from watch.body import BodyPose, body_slump, shoulders

SETTINGS = {"drop_limit": 0.5, "lean_limit_deg": 20.0, "head_drop_limit": 0.25, "forward_limit": 0.2}
BASE = BodyPose(center_x=320, center_y=400, width=200, tilt=2.0, head=-0.6)


def test_upright_and_small_movements_are_not_slumped():
    assert body_slump(BASE, BASE, SETTINGS)[0] == 0.0
    assert body_slump(BodyPose(320, 440, 200, 8.0), BASE, SETTINGS)[0] < 1.0


def test_dropped_or_leaning_shoulders_are_slumped():
    score, drop, *_ = body_slump(BodyPose(320, 510, 200, 2.0), BASE, SETTINGS)
    assert score == 1.0 and drop > 0.5
    score, _, lean, *_ = body_slump(BodyPose(320, 400, 200, 25.0), BASE, SETTINGS)
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
    score, _, _, head_drop, _ = body_slump(BodyPose(320, 400, 200, 2.0, head=-0.3), BASE, SETTINGS)
    assert score == 1.0 and round(head_drop, 2) == 0.3
    assert body_slump(BodyPose(320, 400, 200, 2.0, head=-0.55), BASE, SETTINGS)[0] < 1.0
    assert body_slump(BodyPose(320, 400, 200, 2.0), BASE, SETTINGS)[3] is None


def test_leaning_toward_the_camera_is_slumped():
    score, *_, forward = body_slump(BodyPose(320, 400, 250, 2.0), BASE, SETTINGS)
    assert score == 1.0 and forward == 0.25
    assert body_slump(BodyPose(320, 400, 220, 2.0), BASE, SETTINGS)[0] < 1.0
    assert body_slump(BodyPose(320, 400, 160, 2.0), BASE, SETTINGS)[4] == 0.0


def test_head_is_found_from_whatever_head_points_a_helmet_leaves_visible():
    landmarks = [SimpleNamespace(x=0.5, y=0.3, visibility=0.1) for _ in range(33)]
    landmarks[11] = SimpleNamespace(x=0.6, y=0.7, visibility=0.9)
    landmarks[12] = SimpleNamespace(x=0.4, y=0.7, visibility=0.9)
    assert shoulders(landmarks, 640, 480, 0.5).head is None
    landmarks[7] = SimpleNamespace(x=0.55, y=0.4, visibility=0.8)  # one ear, nose hidden
    assert shoulders(landmarks, 640, 480, 0.5).head < 0
