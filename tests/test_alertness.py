from watch.alertness import AlertnessMonitor, AlertnessStatus, EscalationMonitor, RollingMedian, head_droop


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
