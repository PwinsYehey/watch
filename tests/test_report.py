import csv

from watch.report import SessionLog


def _session():
    log = SessionLog(interval=0.25)
    for i in range(8):
        log.add(100.0 + i * 0.5, "Calibrating")
    log.event("Normal")
    for i in range(10):
        log.add(104.0 + i * 0.5, "Normal", {"ear": 0.3, "mar": 0.05, "head_down": 5.0, "head_tilt": 1.0}, {"ear_closed": 0.2, "head_down": 30.0, "head_tilt": 26.0})
    log.event("Warning: eyes closed")
    for i in range(4):
        log.add(109.0 + i * 0.5, "Warning", {"ear": 0.1}, {"ear_closed": 0.2})
    log.event("Critical: eyes closed too long")
    for i in range(4):
        log.add(111.0 + i * 0.5, "Critical", {"ear": 0.05}, {"ear_closed": 0.2})
    return log


def test_samples_are_throttled_but_keep_events():
    log = SessionLog(interval=0.25)
    log.add(0.0, "Normal")
    log.add(0.1, "Normal")
    log.event("Warning: yawning")
    log.add(0.15, "Warning")
    assert [s.state for s in log.samples] == ["Normal", "Warning"]
    assert log.samples[-1].event == "Warning: yawning"


def test_summary_counts_alerts():
    summary = _session().summary()
    assert "duration 0:12" in summary and "warnings 1" in summary and "critical alerts 1" in summary


def test_time_in_states():
    seconds = _session().time_in_states()
    assert seconds["Critical"] == 1.75 and seconds["Normal"] == 5.0 and seconds["Camera lost"] == 0.0


def test_log_lists_each_change_with_its_reason():
    log = _session()
    log.event("Recalibrated")
    log.add(113.0, "Critical")
    assert [(status, detail) for _, _, status, detail in log.log()] == [
        ("CALIBRATING", ""),
        ("NORMAL", ""),
        ("WARNING", "eyes closed"),
        ("CRITICAL", "eyes closed too long"),
        ("RECALIBRATED", ""),
    ]


def test_save_writes_report_and_csv(tmp_path):
    png, csv_path = _session().save(tmp_path)
    assert png.stat().st_size > 10_000
    rows = list(csv.DictReader(csv_path.open()))
    assert list(rows[0]) == ["time", "elapsed", "status", "detail"]
    assert [row["status"] for row in rows] == ["CALIBRATING", "NORMAL", "WARNING", "CRITICAL"]
    assert rows[2]["detail"] == "eyes closed" and rows[2]["elapsed"] == "0:09"
