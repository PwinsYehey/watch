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


def test_summary_counts_alerts_and_time_in_critical():
    summary = _session().summary()
    assert "duration 0:12" in summary and "warnings 1" in summary and "critical alerts 1" in summary
    assert "time in critical 0:02" in summary


def test_save_writes_report_and_csv(tmp_path):
    png, csv_path = _session().save(tmp_path)
    assert png.stat().st_size > 10_000
    rows = list(csv.DictReader(csv_path.open()))
    assert rows[0]["state"] == "Calibrating" and rows[-1]["state"] == "Critical"
    assert any(row["event"] == "Warning: eyes closed" for row in rows)
