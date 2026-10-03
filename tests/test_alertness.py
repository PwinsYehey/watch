from watch.alertness import AlertnessMonitor, RollingMedian


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
