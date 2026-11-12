import time

from watch import live
from watch.live import Alarm


def _fake_pinctrl(monkeypatch):
    calls = []
    monkeypatch.setattr(live.shutil, "which", lambda name: "/usr/bin/pinctrl")
    monkeypatch.setattr(live.subprocess, "run", lambda args, check: calls.append(args[-1]))
    return calls


def test_buzzer_starts_silent_beeps_while_active_and_ends_silent(monkeypatch):
    calls = _fake_pinctrl(monkeypatch)
    alarm = Alarm(buzzer_pin=4)
    assert calls == ["dh"]

    alarm.set(True)
    time.sleep(0.4)
    alarm.close()

    assert "dl" in calls
    assert calls[-1] == "dh"


def test_single_beep_switches_the_buzzer_on_then_off(monkeypatch):
    calls = _fake_pinctrl(monkeypatch)
    alarm = Alarm(buzzer_pin=4)
    alarm.beep(0.01)
    alarm.close()
    assert calls[:3] == ["dh", "dl", "dh"]


def test_no_buzzer_without_a_pin(monkeypatch):
    calls = _fake_pinctrl(monkeypatch)
    Alarm().close()
    assert calls == []
