import math
import statistics
from collections import deque
from dataclasses import dataclass


def head_droop(pitch: float, roll: float, head: dict) -> float:
    """0 upright to 1 drooped. Looking down and tilting sideways add up, so a diagonal droop counts as well as a straight one."""

    def past(value, upright, drooped):
        return 0.0 if drooped == upright else max(0.0, (value - upright) / (drooped - upright))

    down = past(pitch, head["upright_threshold"], head["drooped_threshold"])
    side = past(roll, head["roll_upright_threshold"], head["roll_drooped_threshold"])
    return min(1.0, math.hypot(down, side))


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


CUE_LABELS = {
    "eyes": "eyes closed",
    "mouth": "yawning",
    "head": "head drooped",
    "body": "slumped",
    "nodding": "nodding off",
    "still": "no movement",
    "check_in": "no check-in answer",
}


class NodCounter:
    """Counts nods: the head dropping toward the shoulders and coming back up quickly, the jerk of someone nodding off.
    A drop that stays down is a slump instead, which the posture check catches."""

    def __init__(self, nod_drop, nod_max_sec, window_sec, nods_limit):
        self.nod_drop, self.nod_max = nod_drop, nod_max_sec
        self.window, self.limit = window_sec, nods_limit
        self._down_since: float | None = None
        self._nods: deque[float] = deque()

    @property
    def count(self) -> int:
        return len(self._nods)

    def update(self, head_drop: float | None, now: float) -> AlertnessStatus:
        while self._nods and now - self._nods[0] > self.window:
            self._nods.popleft()
        if head_drop is None:
            return AlertnessStatus("no_face", 0.0, 0.0)
        if head_drop >= self.nod_drop:
            if self._down_since is None:
                self._down_since = now
        elif head_drop < self.nod_drop / 2 and self._down_since is not None:
            if now - self._down_since <= self.nod_max:
                self._nods.append(now)
            self._down_since = None
        state = "not_alert" if len(self._nods) >= self.limit else "alert"
        return AlertnessStatus(state, 0.0, len(self._nods) / self.limit)


class StillnessMonitor:
    """Flags a body that has barely moved for a long stretch, a sign of zoning out. Movement is the spread of the shoulder
    and head positions over the window in shoulder widths, which the small frame-to-frame jitter of the tracker cannot fake."""

    def __init__(self, window_sec, still_limit):
        self.window, self.limit = window_sec, still_limit
        self._history: deque[tuple[float, tuple[float, ...]]] = deque()
        self._still_since: float | None = None
        self.spread: float | None = None

    def update(self, pose, now: float) -> AlertnessStatus:
        if pose is None:
            self._history.clear()
            self._still_since, self.spread = None, None
            return AlertnessStatus("no_face", 0.0, 0.0)
        head = pose.head if pose.head is not None else 0.0
        self._history.append((now, (pose.center_x / pose.width, pose.center_y / pose.width, head)))
        while now - self._history[0][0] > self.window:
            self._history.popleft()
        if now - self._history[0][0] < self.window * 0.9:
            self.spread = None
            return AlertnessStatus("alert", 0.0, 0.0)
        points = [p for _, p in self._history]
        self.spread = max(max(axis) - min(axis) for axis in zip(*points))
        if self.spread >= self.limit:
            self._still_since = None
            return AlertnessStatus("alert", 0.0, 0.0)
        self._still_since = self._still_since or now
        return AlertnessStatus("not_alert", now - self._still_since, 0.0)


class CheckInMonitor:

    def __init__(self, blind_sec, answer_sec):
        self.blind = blind_sec
        self.answer = answer_sec
        self._blind_since: float | None = None
        self._asked_at: float | None = None
        self.last_reaction: float | None = None
        self.just_answered = False

    @property
    def asking(self) -> bool:
        return self._asked_at is not None

    def update(self, eyes_readable: bool, person_present: bool, answered: bool, now: float) -> AlertnessStatus:
        self.just_answered = False
        if not person_present:
            self._blind_since = self._asked_at = None
            return AlertnessStatus("no_face", 0.0, 0.0)
        if eyes_readable:
            self._blind_since = self._asked_at = None
            return AlertnessStatus("alert", 0.0, 0.0)
        if self._blind_since is None:
            self._blind_since = now
        if self._asked_at is not None and answered:
            self.last_reaction, self.just_answered = now - self._asked_at, True
            self._asked_at, self._blind_since = None, now
            return AlertnessStatus("alert", 0.0, 0.0)
        if self._asked_at is None and now - self._blind_since >= self.blind:
            self._asked_at = now
        overdue = now - self._asked_at - self.answer if self._asked_at is not None else -1.0
        return AlertnessStatus("not_alert", overdue, 0.0) if overdue >= 0 else AlertnessStatus("alert", 0.0, 0.0)


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
