from watch.alertness import AlertnessMonitor, AlertnessStatus, CheckInMonitor, EscalationMonitor, NodCounter, RollingMedian, StillnessMonitor, head_droop
from watch.body import BodyPose
from watch.config import load_config


def _monitor(grace=0.0):
    return AlertnessMonitor(1.5, 5.0, 0.75, 0.85, grace)


def test_sustained_closure_triggers_not_alert():
    monitor = _monitor()
    states = [monitor.update(1.0, t / 10).state for t in range(20)]
    assert states[0] == "alert" and states[-1] == "not_alert"


def test_blink_does_not_trigger():
    monitor = _monitor()
    for t in range(5):
        monitor.update(1.0, t / 10)
    monitor.update(0.0, 0.5)
    assert monitor.update(0.0, 0.6).state == "alert"
    assert monitor.update(1.0, 0.7).closed_duration == 0.0


def test_grace_period_keeps_streak_through_short_dip():
    monitor = _monitor(grace=0.5)
    monitor.update(1.0, 0.0)
    monitor.update(0.0, 1.0)
    assert monitor.update(1.0, 1.2).closed_duration > 1.0


def test_perclos_needs_full_window():
    monitor = AlertnessMonitor(99.0, 2.0, 0.5, 0.9)
    statuses = [monitor.update(0.6, t / 10) for t in range(30)]
    assert statuses[5].state == "alert" and statuses[-1].state == "not_alert"


def test_no_face():
    assert _monitor().update(None, 0.0).state == "no_face"


def test_rolling_median_ignores_missing_values():
    smoother = RollingMedian(3)
    assert [smoother.add(v) for v in (1.0, None, 9.0, 2.0)] == [1.0, None, 5.0, 2.0]


def _escalation():
    return EscalationMonitor(30.0, 10.0, {"eyes": 3.0, "mouth": None, "head": 4.0}, {"eyes": 2, "mouth": 3, "head": 2})


def _statuses(eyes=0.0, mouth=0.0, head=0.0):
    """Each cue is concerning when its streak (seconds) is positive."""
    return {cue: AlertnessStatus("not_alert" if streak else "alert", streak, 0.0) for cue, streak in (("eyes", eyes), ("mouth", mouth), ("head", head))}


def test_single_event_warns_then_clears_after_window():
    monitor = _escalation()
    assert monitor.update(_statuses(eyes=1.5), True, 0.0).level == 1
    assert monitor.update(_statuses(), True, 1.0).level == 1
    assert monitor.update(_statuses(), True, 31.0).level == 0


def test_persisting_eye_closure_turns_critical():
    monitor = _escalation()
    monitor.update(_statuses(eyes=1.5), True, 0.0)
    status = monitor.update(_statuses(eyes=3.0), True, 1.5)
    assert status.level == 2 and status.reason == "eyes closed too long"


def test_second_cue_or_repeat_turns_critical():
    monitor = _escalation()
    monitor.update(_statuses(eyes=1.5), True, 0.0)
    monitor.update(_statuses(), True, 1.0)
    assert monitor.update(_statuses(mouth=3.0), True, 5.0).reason == "eyes closed + yawning"

    monitor = _escalation()
    monitor.update(_statuses(eyes=1.5), True, 0.0)
    monitor.update(_statuses(), True, 1.0)
    assert monitor.update(_statuses(eyes=1.5), True, 5.0).reason == "eyes closed again"


def test_yawns_alone_need_three():
    monitor = _escalation()
    levels = []
    for t in (0.0, 10.0, 20.0):
        levels.append(monitor.update(_statuses(mouth=3.0), True, t).level)
        monitor.update(_statuses(), True, t + 1.0)
    assert levels == [1, 1, 2]


def test_critical_calms_down_to_warning():
    monitor = _escalation()
    monitor.update(_statuses(eyes=1.5), True, 0.0)
    monitor.update(_statuses(eyes=3.0), True, 1.5)
    monitor.update(_statuses(), True, 2.0)
    assert monitor.update(_statuses(), True, 11.0).level == 2
    status = monitor.update(_statuses(), True, 12.0)
    assert status.level == 1 and status.window_left == 30.0
    # Any new event while recovering is critical again.
    assert monitor.update(_statuses(head=2.0), True, 15.0).level == 2


def test_face_lost_holds_level_and_pauses_window():
    monitor = _escalation()
    monitor.update(_statuses(eyes=1.5), True, 0.0)
    monitor.update(_statuses(), True, 1.0)
    assert monitor.update(_statuses(), False, 100.0).level == 1
    assert monitor.update(_statuses(), True, 101.0).level == 1


HEAD = {"upright_threshold": 5.0, "drooped_threshold": 30.0, "roll_upright_threshold": 2.0, "roll_drooped_threshold": 27.0}


def test_head_droop_straight_down_or_sideways():
    assert head_droop(30.0, 2.0, HEAD) == 1.0
    assert head_droop(5.0, 27.0, HEAD) == 1.0
    assert head_droop(17.5, 2.0, HEAD) == 0.5


def test_head_droop_counts_diagonal_tilts():
    # 18 degrees down and 18 to the side is drooped although neither reaches 25 alone.
    assert head_droop(5.0 + 18, 2.0 + 18, HEAD) == 1.0
    assert head_droop(5.0 + 10, 2.0 + 10, HEAD) < 1.0


def test_head_droop_ignores_looking_up():
    assert head_droop(-20.0, 2.0, HEAD) == 0.0


def test_check_in_is_asked_only_after_the_eyes_stay_unreadable():
    check_in = CheckInMonitor(blind_sec=60, answer_sec=5)
    assert check_in.update(False, True, False, 0).state == "alert" and not check_in.asking
    check_in.update(False, True, False, 59)
    assert not check_in.asking
    assert check_in.update(False, True, False, 60).state == "alert" and check_in.asking


def test_answered_check_in_resets_and_records_reaction_time():
    check_in = CheckInMonitor(blind_sec=60, answer_sec=5)
    check_in.update(False, True, False, 0)
    check_in.update(False, True, False, 60)
    status = check_in.update(False, True, True, 61.5)
    assert status.state == "alert" and not check_in.asking and check_in.just_answered and check_in.last_reaction == 1.5
    check_in.update(False, True, False, 120)
    assert not check_in.asking


def test_unanswered_check_in_is_not_alert():
    check_in = CheckInMonitor(blind_sec=60, answer_sec=5)
    check_in.update(False, True, False, 0)
    check_in.update(False, True, False, 60)
    status = check_in.update(False, True, False, 67)
    assert status.state == "not_alert" and status.closed_duration == 2


def test_readable_eyes_or_an_empty_seat_cancel_the_check_in():
    check_in = CheckInMonitor(blind_sec=60, answer_sec=5)
    check_in.update(False, True, False, 0)
    check_in.update(False, True, False, 60)
    check_in.update(True, True, False, 61)
    assert not check_in.asking
    assert check_in.update(False, False, False, 200).state == "no_face"


def test_unanswered_check_in_warns_then_turns_critical():
    monitor = EscalationMonitor(30.0, 10.0, {"check_in": 5.0}, {"check_in": 2})
    check_in = CheckInMonitor(blind_sec=60, answer_sec=5)
    for now in (0, 60, 64):
        assert monitor.update({"check_in": check_in.update(False, True, False, now)}, True, now).level == 0
    status = monitor.update({"check_in": check_in.update(False, True, False, 65)}, True, 65)
    assert status.level == 1 and status.reason == "no check-in answer"
    status = monitor.update({"check_in": check_in.update(False, True, False, 70)}, True, 70)
    assert status.level == 2 and status.reason == "no check-in answer too long"


def test_quick_head_drops_count_as_nods_but_a_held_drop_does_not():
    nods = NodCounter(nod_drop=0.12, nod_max_sec=2.5, window_sec=60.0, nods_limit=3)
    t = 0.0
    for _ in range(2):
        nods.update(0.0, t)
        nods.update(0.2, t + 1)
        status = nods.update(0.0, t + 2)
        t += 10
    assert nods.count == 2 and status.state == "alert"
    nods.update(0.2, t)
    nods.update(0.2, t + 5)  # held down too long: a slump, not a nod
    assert nods.update(0.0, t + 6).state == "alert" and nods.count == 2
    nods.update(0.2, t + 10)
    assert nods.update(0.0, t + 11).state == "not_alert"
    assert nods.update(0.0, t + 200).state == "alert"  # old nods leave the window


def test_a_body_that_barely_moves_for_the_whole_window_is_flagged():
    still = StillnessMonitor(window_sec=45.0, still_limit=0.05)
    assert still.update(BodyPose(320, 400, 200, 0.0, head=-0.6), 0.0).state == "alert"
    for t in range(1, 46):
        status = still.update(BodyPose(320 + (t % 2), 400, 200, 0.0, head=-0.6), float(t))
    assert status.state == "not_alert"
    assert still.update(BodyPose(360, 420, 200, 0.0, head=-0.6), 46.0).state == "alert"
    assert still.update(None, 47.0).state == "no_face" and still.spread is None


def test_escalation_knows_the_body_cues():
    config = load_config()["escalation"]
    for cue in ("nodding", "still"):
        assert cue in config["persist_sec"] and cue in config["repeat_count"]


def test_critical_clears_once_alert_even_while_nod_and_stillness_history_lingers():
    monitor = EscalationMonitor(30.0, 5.0, {"eyes": 3.0, "nodding": None, "still": 30.0}, {"eyes": 2, "nodding": 2, "still": 2})

    def statuses(eyes=0.0, nodding=False, still=0.0):
        return {
            "eyes": AlertnessStatus("not_alert" if eyes else "alert", eyes, 0.0),
            "nodding": AlertnessStatus("not_alert" if nodding else "alert", 0.0, 0.0),
            "still": AlertnessStatus("not_alert" if still else "alert", still, 0.0),
        }

    assert monitor.update(statuses(eyes=3.5, nodding=True, still=40.0), True, 0.0).level == 2
    assert monitor.update(statuses(nodding=True, still=41.0), True, 1.0).level == 2
    assert monitor.update(statuses(nodding=True, still=46.0), True, 6.0).reason == "recovering"
    assert monitor.update(statuses(nodding=True, still=50.0), True, 10.0).level == 1
    assert monitor.update(statuses(), True, 40.0).level == 0
