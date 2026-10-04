"""Session recording and export: a CSV of the readings and a PNG report with graphs."""

import csv
import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

# Report colours: the validated light palette (series), status colours for alert levels, ink for text.
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"
LEVEL_STYLE = {
    "Normal": "#0ca30c",
    "Warning": "#fab219",
    "Critical": "#d03b3b",
    "No face": MUTED,
    "Calibrating": AXIS,
    "Camera lost": "#7a1f1f",
}
LEVEL_NAMES = {0: "Normal", 1: "Warning", 2: "Critical"}


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

    def _write_csv(self, path: Path):
        names = [f.name for f in fields(Sample)]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", *names[1:]])
            for sample in self.samples:
                row = [getattr(sample, name) for name in names]
                row[0] = time.strftime("%H:%M:%S", time.localtime(sample.clock))
                writer.writerow(["" if value is None else round(value, 4) if isinstance(value, float) else value for value in row])

    def summary(self) -> str:
        if not self.samples:
            return "No data recorded"
        duration = self.samples[-1].elapsed
        seconds_in = {state: 0.0 for state in LEVEL_STYLE}
        warnings = criticals = 0
        for current, following in zip(self.samples, self.samples[1:]):
            seconds_in[current.state] += following.elapsed - current.elapsed
            warnings += following.state == "Warning" and current.state in ("Normal", "No face")
            criticals += following.state == "Critical" and current.state != "Critical"
        return (
            f"Started {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.samples[0].clock))}  ·  duration {_clock(duration)}"
            f"  ·  warnings {warnings}  ·  critical alerts {criticals}  ·  time in critical {_clock(seconds_in['Critical'])}"
        )

    def _write_png(self, path: Path):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.ticker import FuncFormatter

        plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"], "font.size": 9})
        t = np.array([s.elapsed for s in self.samples]) if self.samples else np.array([0.0])
        column = lambda name: np.array([np.nan if getattr(s, name) is None else getattr(s, name) for s in self.samples], dtype=float)

        figure = plt.figure(figsize=(12, 8.6), dpi=150, facecolor=SURFACE)
        grid = figure.add_gridspec(4, 1, height_ratios=[0.32, 1, 1, 1.25], hspace=0.55, left=0.07, right=0.88, top=0.86, bottom=0.07)
        axes = [figure.add_subplot(grid[i]) for i in range(4)]
        for ax in axes[1:]:
            ax.sharex(axes[0])
        figure.text(0.07, 0.95, "WATCH session report", fontsize=16, fontweight="bold", color=INK)
        figure.text(0.07, 0.915, self.summary(), fontsize=9.5, color=INK_2)

        def style(ax, title, unit=""):
            ax.set_facecolor(SURFACE)
            ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=6)
            if unit:
                ax.set_ylabel(unit, color=MUTED, fontsize=8.5)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(AXIS)
            ax.tick_params(colors=MUTED, labelsize=8, length=0, pad=4)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _clock(value)))

        # 1. Alert level over time, one coloured band per run of the same state.
        level = axes[0]
        style(level, "Alert level")
        level.grid(False)
        level.set_yticks([])
        runs, start = [], 0
        for i in range(1, len(self.samples) + 1):
            if i == len(self.samples) or self.samples[i].state != self.samples[start].state:
                end = self.samples[i].elapsed if i < len(self.samples) else self.samples[-1].elapsed + self.interval
                runs.append((self.samples[start].state, self.samples[start].elapsed, end))
                start = i
        for state, begin, end in runs:
            level.broken_barh([(begin, end - begin)], (0, 1), color=LEVEL_STYLE[state], linewidth=0)
        level.set_ylim(0, 1)
        used = [state for state in LEVEL_STYLE if any(run[0] == state for run in runs)]
        level.legend(
            handles=[Patch(color=LEVEL_STYLE[state], label=state) for state in used],
            loc="upper left", bbox_to_anchor=(1.005, 1.25), frameon=False, fontsize=8.5, labelcolor=INK_2, handlelength=1.2,
        )

        def series(ax, values, color, label=None):
            ax.plot(t, values, color=color, linewidth=1.5, solid_joinstyle="round", solid_capstyle="round", label=label)

        def limit(ax, values, text):
            """A labelled reference line for a calibrated limit, drawn as steps since recalibration can move it."""
            if np.all(np.isnan(values)):
                return
            ax.plot(t, values, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), drawstyle="steps-post")
            last = values[~np.isnan(values)][-1]
            ax.annotate(text.format(last), (1.005, last), xycoords=("axes fraction", "data"), color=INK_2, fontsize=8, va="center")

        # 2. Eyes: openness against the calibrated "closed" limit.
        eyes = axes[1]
        style(eyes, "Eyes", "EAR")
        series(eyes, column("ear"), SERIES_1)
        limit(eyes, column("ear_closed_limit"), "closed limit {:.2f}")
        eyes.set_ylim(bottom=0)

        # 3. Mouth: opening, with the level at which the opening alone reads as a yawn.
        mouth = axes[2]
        style(mouth, "Mouth", "MAR")
        series(mouth, column("mar"), SERIES_1)
        yawn_level = np.full_like(t, 0.45, dtype=float)
        limit(mouth, yawn_level, "yawn level {:.2f}")
        mouth.set_ylim(bottom=0)

        # 4. Head: looking down and tilting sideways, each against its calibrated limit.
        head = axes[3]
        style(head, "Head", "degrees")
        series(head, column("head_down"), SERIES_1, "Down")
        series(head, column("head_tilt"), SERIES_2, "Tilt")
        limit(head, column("head_down_limit"), "down limit {:.1f}°")
        limit(head, column("head_tilt_limit"), "tilt limit {:.1f}°")
        head.legend(loc="upper left", bbox_to_anchor=(1.005, 1.25), frameon=False, fontsize=8.5, labelcolor=INK_2, handlelength=1.6)
        head.set_xlabel("time since start (m:ss)", color=MUTED, fontsize=8.5)

        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)
        axes[0].set_xlim(0, max(t[-1], 1.0))
        figure.savefig(path, facecolor=SURFACE)
        plt.close(figure)


def _clock(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}" if seconds >= 3600 else f"{seconds // 60}:{seconds % 60:02d}"
