"""Session recording and export: a PNG line graph of alertness over time, and an event log as a CSV."""

import csv
import time
from dataclasses import dataclass
from pathlib import Path


# Report colours: status colours for alert levels, ink for text.
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
LEVEL_STYLE = {
    "Normal": "#0ca30c",
    "Warning": "#fab219",
    "Critical": "#d03b3b",
    "No face": MUTED,
    "Face covered": "#3d7bd6",
    "No person": MUTED,
    "Calibrating": AXIS,
    "Camera lost": "#7a1f1f",
}
LEVEL_NAMES = {0: "Normal", 1: "Warning", 2: "Critical"}
STATUS_TEXT = {
    "Normal": "NORMAL",
    "Warning": "WARNING",
    "Critical": "CRITICAL",
    "No face": "NO FACE DETECTED",
    "Face covered": "FACE COVERED",
    "No person": "NO PERSON DETECTED",
    "Calibrating": "CALIBRATING",
    "Camera lost": "CAMERA LOST",
}
LEVEL_VALUE = {"Normal": 0, "Warning": 1, "Critical": 2}


@dataclass
class Sample:
    clock: float
    elapsed: float
    state: str  # a key of LEVEL_STYLE
    ear: float | None
    mar: float | None
    head_down: float | None
    head_tilt: float | None
    ear_closed_limit: float | None
    head_down_limit: float | None
    head_tilt_limit: float | None
    event: str = ""


class SessionLog:
    """Records the monitoring session a few times a second, cheap enough to run for hours."""

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.samples: list[Sample] = []
        self._start: float | None = None
        self._pending_event = ""

    def event(self, text: str):
        """Attaches a level change or recalibration to the next sample."""
        self._pending_event = f"{self._pending_event}; {text}" if self._pending_event else text

    def add(self, now: float, state: str, readings: dict | None = None, limits: dict | None = None):
        if self._start is None:
            self._start = now
        if self.samples and now - self.samples[-1].clock < self.interval and not self._pending_event:
            return
        readings, limits = readings or {}, limits or {}
        self.samples.append(
            Sample(
                now, now - self._start, state,
                readings.get("ear"), readings.get("mar"), readings.get("head_down"), readings.get("head_tilt"),
                limits.get("ear_closed"), limits.get("head_down"), limits.get("head_tilt"),
                self._pending_event,
            )
        )
        self._pending_event = ""

    def save(self, directory: Path) -> tuple[Path, Path]:
        """Writes watch-session-<time>.csv and .png into directory and returns both paths."""
        directory.mkdir(parents=True, exist_ok=True)
        stem = directory / f"watch-session-{time.strftime('%Y%m%d-%H%M%S')}"
        csv_path, png_path = stem.with_suffix(".csv"), stem.with_suffix(".png")
        self._write_csv(csv_path)
        self._write_png(png_path)
        return png_path, csv_path

    def log(self) -> list[tuple[float, float, str, str]]:
        rows, previous = [], None
        for sample in self.samples:
            events = [part.strip() for part in sample.event.split(";") if part.strip()]
            reason = ""
            for event in events:
                head, _, tail = event.partition(":")
                if head == sample.state:
                    reason = tail.strip()
            if sample.state != previous:
                others = [event for event in events if event.partition(":")[0] != sample.state]
                rows.append((sample.clock, sample.elapsed, STATUS_TEXT[sample.state], reason or "; ".join(others).lower()))
                previous = sample.state
            else:
                for event in events:
                    if event.partition(":")[0] != sample.state:
                        rows.append((sample.clock, sample.elapsed, event.upper(), ""))
        return rows

    def _write_csv(self, path: Path):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "elapsed", "status", "detail"])
            for clock, elapsed, status, detail in self.log():
                writer.writerow([time.strftime("%H:%M:%S", time.localtime(clock)), _clock(elapsed), status, detail])

    def time_in_states(self) -> dict[str, float]:
        seconds = {state: 0.0 for state in LEVEL_STYLE}
        for current, following in zip(self.samples, self.samples[1:]):
            seconds[current.state] += following.elapsed - current.elapsed
        if self.samples:
            seconds[self.samples[-1].state] += self.interval
        return seconds

    def summary(self) -> str:
        if not self.samples:
            return "No data recorded"
        duration = self.samples[-1].elapsed
        warnings = criticals = 0
        for current, following in zip(self.samples, self.samples[1:]):
            warnings += following.state == "Warning" and current.state in ("Normal", "No face", "Face covered", "No person")
            criticals += following.state == "Critical" and current.state != "Critical"
        return (
            f"Started {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.samples[0].clock))}  ·  duration {_clock(duration)}"
            f"  ·  warnings {warnings}  ·  critical alerts {criticals}"
        )

    def _write_png(self, path: Path):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from matplotlib.ticker import FuncFormatter

        plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"], "font.size": 9})
        figure = plt.figure(figsize=(12, 5.2), dpi=150, facecolor=SURFACE)
        figure.text(0.06, 0.93, "WATCH session report", fontsize=16, fontweight="bold", color=INK)
        figure.text(0.06, 0.875, self.summary(), fontsize=9.5, color=INK_2)
        seconds = self.time_in_states()
        totals = "  ·  ".join(f"{STATUS_TEXT[state].capitalize()} {_clock(seconds[state])}" for state in LEVEL_STYLE if seconds[state] >= 0.5)
        figure.text(0.06, 0.83, "Time in each state:  " + (totals or "-"), fontsize=9.5, color=INK_2)
        figure.text(0.06, 0.74, "Alertness over time", fontsize=11, fontweight="bold", color=INK)

        ax = figure.add_axes([0.11, 0.15, 0.71, 0.53])
        ax.set_facecolor(SURFACE)
        end = (self.samples[-1].elapsed + self.interval) if self.samples else 1.0
        runs, start = [], 0
        for i in range(1, len(self.samples) + 1):
            if i == len(self.samples) or self.samples[i].state != self.samples[start].state:
                runs.append((self.samples[start].state, self.samples[start].elapsed, self.samples[i].elapsed if i < len(self.samples) else end))
                start = i
        previous = None
        for state, begin, finish in runs:
            if state in LEVEL_VALUE:
                y = LEVEL_VALUE[state]
                if previous is not None and previous[0] == begin:
                    ax.plot([begin, begin], [previous[1], y], color=AXIS, linewidth=1.5, zorder=2)
                ax.plot([begin, finish], [y, y], color=LEVEL_STYLE[state], linewidth=3, solid_capstyle="butt", zorder=3)
                previous = (finish, y)
            else:
                ax.axvspan(begin, finish, color=LEVEL_STYLE[state], alpha=0.25, linewidth=0)
                previous = None
        ax.set_xlim(0, max(end, 1.0))
        ax.set_ylim(-0.4, 2.4)
        ax.set_yticks(list(LEVEL_VALUE.values()), [STATUS_TEXT[state].capitalize() for state in LEVEL_VALUE])
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=0, pad=6)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _clock(value)))
        ax.set_xlabel("time since start (m:ss)", color=MUTED, fontsize=8.5)
        used = {state for state, _, _ in runs}
        handles = [Line2D([0], [0], color=LEVEL_STYLE[s], linewidth=3, label=STATUS_TEXT[s].capitalize()) for s in LEVEL_VALUE if s in used]
        handles += [Patch(color=LEVEL_STYLE[s], alpha=0.25, label=STATUS_TEXT[s].capitalize()) for s in LEVEL_STYLE if s not in LEVEL_VALUE and s in used]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8.5, labelcolor=INK_2, handlelength=1.6)
        figure.savefig(path, facecolor=SURFACE)
        plt.close(figure)

def _clock(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}" if seconds >= 3600 else f"{seconds // 60}:{seconds % 60:02d}"
