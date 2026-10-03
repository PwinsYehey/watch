import statistics
from collections import deque
from dataclasses import dataclass


@dataclass
class AlertnessStatus:
    state: str  # "alert", "not_alert", or "no_face"
    closed_duration: float
    perclos: float


class RollingMedian:
    def __init__(self, size: int):
        self._values: deque[float] = deque(maxlen=size)

    def add(self, value: float | None) -> float | None:
        if value is None:
            return None
        self._values.append(value)
        return statistics.median(self._values)


class AlertnessMonitor:
    """Flags not_alert after a sustained streak above threshold, or a high mean score (PERCLOS) over a full window."""

    def __init__(self, closed_duration_threshold_sec, perclos_window_sec, perclos_threshold, closedness_threshold, dip_grace_period_sec=0.0):
        self.closed_duration_threshold = closed_duration_threshold_sec
        self.perclos_window = perclos_window_sec
        self.perclos_threshold = perclos_threshold
        self.closedness_threshold = closedness_threshold
        self.dip_grace_period = dip_grace_period_sec
        self._closed_since: float | None = None
        self._below_since: float | None = None
        self._history: deque[tuple[float, float]] = deque()

    def update(self, closedness: float | None, now: float) -> AlertnessStatus:
        if closedness is None:
            return AlertnessStatus("no_face", 0.0, self._perclos(now))

        if closedness >= self.closedness_threshold:
            self._below_since = None
            if self._closed_since is None:
                self._closed_since = now
        else:
            if self._below_since is None:
                self._below_since = now
            if now - self._below_since > self.dip_grace_period:
                self._closed_since = None

        self._history.append((now, closedness))
        perclos = self._perclos(now)
        closed_duration = now - self._closed_since if self._closed_since is not None else 0.0
        window_filled = now - self._history[0][0] >= self.perclos_window
        not_alert = closed_duration >= self.closed_duration_threshold or (window_filled and perclos >= self.perclos_threshold)
        return AlertnessStatus("not_alert" if not_alert else "alert", closed_duration, perclos)

    def _perclos(self, now: float) -> float:
        while self._history and now - self._history[0][0] > self.perclos_window:
            self._history.popleft()
        if not self._history:
            return 0.0
        return sum(score for _, score in self._history) / len(self._history)


CUE_LABELS = {"eyes": "eyes closed", "mouth": "yawning", "head": "head drooped"}


@dataclass
class EscalationStatus:
    level: int  # 0 normal, 1 warning, 2 critical
    reason: str
    window_left: float


class EscalationMonitor:
    """One cue event warns; a repeat, a second cue, or a persisting event within the window escalates to critical."""

    def __init__(self, window_sec, calm_sec, persist_sec, repeat_count):
        self.window = window_sec
        self.calm = calm_sec
        self.persist = persist_sec
        self.repeat = repeat_count
        self.level, self.reason = 0, ""
        self._events: list[str] = []
        self._active: dict[str, bool] = {}
        self._window_end = 0.0
        self._calm_since: float | None = None
        self._last: float | None = None

    def update(self, statuses: dict[str, AlertnessStatus], face_present: bool, now: float) -> EscalationStatus:
        elapsed = now - self._last if self._last is not None else 0.0
        self._last = now
        if not face_present:
            # Hold the level while the face is out of view: the window is paused and nothing counts as calm.
            self._window_end += elapsed
            self._calm_since = None
            return self._status(now)

        concerning = [cue for cue, status in statuses.items() if status.state == "not_alert"]
        new_events = [cue for cue in concerning if not self._active.get(cue)]
        for cue, status in statuses.items():
            if status.state != "no_face":
                self._active[cue] = status.state == "not_alert"
        self._events += new_events

        if self.level == 2:
            if concerning:
                self._calm_since = None
            elif self._calm_since is None:
                self._calm_since = now
            elif now - self._calm_since >= self.calm:
                self.level, self.reason = 1, "recovering"
                self._window_end = now + self.window
            return self._status(now)

        if new_events:
            self._window_end = now + self.window
            if self.level == 0:
                self.level, self.reason = 1, " + ".join(CUE_LABELS[cue] for cue in new_events)
            if reason := self._repeated_or_combined():
                return self._critical(reason, now)
        if self.level == 1:
            for cue in concerning:
                if self.persist[cue] is not None and statuses[cue].closed_duration >= self.persist[cue]:
                    return self._critical(f"{CUE_LABELS[cue]} too long", now)
            if not concerning and now >= self._window_end:
                self.level, self.reason = 0, ""
                self._events.clear()
        return self._status(now)

    def _repeated_or_combined(self) -> str:
        cues = sorted(set(self._events), key=list(CUE_LABELS).index)
        if len(cues) > 1:
            return " + ".join(CUE_LABELS[cue] for cue in cues)
        cue = cues[0]
        return f"{CUE_LABELS[cue]} again" if self._events.count(cue) >= self.repeat[cue] else ""

    def _critical(self, reason: str, now: float) -> EscalationStatus:
        self.level, self.reason = 2, reason
        self._calm_since = None
        return self._status(now)

    def _status(self, now: float) -> EscalationStatus:
        window_left = max(0.0, self._window_end - now) if self.level == 1 else 0.0
        return EscalationStatus(self.level, self.reason, window_left)
