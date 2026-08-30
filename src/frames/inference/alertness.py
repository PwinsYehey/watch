import statistics
from collections import deque
from dataclasses import dataclass


@dataclass
class AlertnessStatus:
    state: str  # "alert", "not_alert", or "no_face"
    closed_duration: float
    perclos: float


class RollingMedian:
    """Smooths out a noisy per-frame CNN score so a single bad frame can't sway the sustained-duration or PERCLOS checks."""

    def __init__(self, size: int):
        self._values: deque[float] = deque(maxlen=size)

    def add(self, value: float | None) -> float | None:
        if value is None:
            return None
        self._values.append(value)
        return statistics.median(self._values)


class AlertnessMonitor:
    def __init__(
        self,
        closed_duration_threshold: float,
        perclos_window: float,
        perclos_threshold: float,
        closedness_threshold: float,
        dip_grace_period: float = 0.0,
    ):
        self.closed_duration_threshold = closed_duration_threshold
        self.perclos_window = perclos_window
        self.perclos_threshold = perclos_threshold
        self.closedness_threshold = closedness_threshold
        self.dip_grace_period = dip_grace_period
        self._closed_since: float | None = None
        self._below_since: float | None = None
        self._history: deque[tuple[float, float]] = deque()

    def update(self, closedness: float | None, now: float) -> AlertnessStatus:
        """`closedness` is P(closed) in [0, 1], or None when no face was detected this frame."""
        if closedness is None:
            return AlertnessStatus(state="no_face", closed_duration=0.0, perclos=self._perclos(now))

        is_closed_now = closedness >= self.closedness_threshold
        if is_closed_now:
            self._below_since = None
            if self._closed_since is None:
                self._closed_since = now
        else:
            # a brief dip below threshold doesn't reset the streak until it outlasts dip_grace_period
            if self._below_since is None:
                self._below_since = now
            if now - self._below_since > self.dip_grace_period:
                self._closed_since = None

        self._history.append((now, closedness))
        self._trim_history(now)

        closed_duration = now - self._closed_since if self._closed_since is not None else 0.0
        perclos = self._perclos(now)
        # only trust PERCLOS once the rolling window has actually filled
        window_filled = now - self._history[0][0] >= self.perclos_window

        is_not_alert = closed_duration >= self.closed_duration_threshold or (window_filled and perclos >= self.perclos_threshold)
        return AlertnessStatus(state="not_alert" if is_not_alert else "alert", closed_duration=closed_duration, perclos=perclos)

    def _trim_history(self, now: float) -> None:
        while self._history and now - self._history[0][0] > self.perclos_window:
            self._history.popleft()

    def _perclos(self, now: float) -> float:
        """Mean closedness over the window -- a continuous average, not a binary closed/open count, so partial closure shows up smoothly."""
        self._trim_history(now)
        if not self._history:
            return 0.0
        return sum(score for _, score in self._history) / len(self._history)
