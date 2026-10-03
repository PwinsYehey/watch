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
