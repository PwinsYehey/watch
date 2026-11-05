// Browser port of src/watch/report.py: records the session and exports a line graph of alertness over time, and an event log as a CSV.

const REPORT = {
  surface: "#fcfcfb", grid: "#e1e0d9", axis: "#c3c2b7",
  ink: "#0b0b0b", ink2: "#52514e", muted: "#898781",
  levels: { Normal: "#0ca30c", Warning: "#fab219", Critical: "#d03b3b", "No face": "#898781", "Face covered": "#3d7bd6", Calibrating: "#c3c2b7", "Camera lost": "#7a1f1f" },
  font: 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
};
const REPORT_LEVEL_NAMES = { 0: "Normal", 1: "Warning", 2: "Critical" };
const REPORT_STATUS = { Normal: "NORMAL", Warning: "WARNING", Critical: "CRITICAL", "No face": "NO FACE DETECTED", "Face covered": "FACE COVERED", Calibrating: "CALIBRATING", "Camera lost": "CAMERA LOST" };
const REPORT_LEVEL_VALUE = { Normal: 0, Warning: 1, Critical: 2 };

function reportClock(seconds) {
  const s = Math.max(0, Math.round(seconds));
  const pad = (n) => String(n).padStart(2, "0");
  return s >= 3600 ? `${Math.floor(s / 3600)}:${pad(Math.floor((s % 3600) / 60))}:${pad(s % 60)}` : `${Math.floor(s / 60)}:${pad(s % 60)}`;
}

class SessionLog {
  constructor(interval = 0.25) {
    this.interval = interval;
    this.samples = [];
    this.start = null;
    this.pendingEvent = "";
  }

  event(text) {
    this.pendingEvent = this.pendingEvent ? `${this.pendingEvent}; ${text}` : text;
  }

  add(now, state, readings = {}, limits = {}) {
    if (this.start === null) this.start = now;
    const last = this.samples[this.samples.length - 1];
    if (last && now - last.clock < this.interval && !this.pendingEvent) return;
    this.samples.push({
      clock: now, elapsed: now - this.start, state,
      ear: readings.ear ?? null, mar: readings.mar ?? null, headDown: readings.headDown ?? null, headTilt: readings.headTilt ?? null,
      earClosedLimit: limits.earClosed ?? null, headDownLimit: limits.headDown ?? null, headTiltLimit: limits.headTilt ?? null,
      event: this.pendingEvent,
    });
    this.pendingEvent = "";
  }

  log() {
    const rows = [];
    let previous = null;
    for (const sample of this.samples) {
      const events = sample.event.split(";").map((e) => e.trim()).filter(Boolean);
      const head = (event) => event.split(":")[0];
      const reason = events.filter((e) => head(e) === sample.state).map((e) => e.split(":").slice(1).join(":").trim()).pop() || "";
      if (sample.state !== previous) {
        const others = events.filter((e) => head(e) !== sample.state);
        rows.push({ clock: sample.clock, elapsed: sample.elapsed, status: REPORT_STATUS[sample.state], detail: reason || others.join("; ").toLowerCase() });
        previous = sample.state;
      } else {
        for (const event of events) if (head(event) !== sample.state) rows.push({ clock: sample.clock, elapsed: sample.elapsed, status: event.toUpperCase(), detail: "" });
      }
    }
    return rows;
  }

  timeInStates() {
    const seconds = Object.fromEntries(Object.keys(REPORT.levels).map((state) => [state, 0]));
    for (let i = 1; i < this.samples.length; i++) seconds[this.samples[i - 1].state] += this.samples[i].elapsed - this.samples[i - 1].elapsed;
    if (this.samples.length) seconds[this.samples[this.samples.length - 1].state] += this.interval;
    return seconds;
  }

  summary() {
    if (!this.samples.length) return "No data recorded";
    let warnings = 0, criticals = 0;
    for (let i = 1; i < this.samples.length; i++) {
      const current = this.samples[i - 1], following = this.samples[i];
      if (following.state === "Warning" && (current.state === "Normal" || current.state === "No face" || current.state === "Face covered")) warnings++;
      if (following.state === "Critical" && current.state !== "Critical") criticals++;
    }
    const started = new Date(this.samples[0].clock * 1000);
    return `Started ${started.toLocaleDateString("en-CA")} ${started.toLocaleTimeString([], { hour12: false })}  ·  duration ${reportClock(this.samples[this.samples.length - 1].elapsed)}`
      + `  ·  warnings ${warnings}  ·  critical alerts ${criticals}`;
  }

  csv() {
    const cell = (value) => (/[",]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value);
    const rows = this.log().map((r) => [new Date(r.clock * 1000).toLocaleTimeString([], { hour12: false }), reportClock(r.elapsed), r.status, r.detail].map(cell).join(","));
    return ["time,elapsed,status,detail", ...rows].join("\n");
  }

  // Mirrors SessionLog._write_png in src/watch/report.py.
  drawReport(canvas) {
    const scale = 150, W = 12 * scale, H = Math.round(5.2 * scale);
    canvas.width = W;
    canvas.height = H;
    const g = canvas.getContext("2d");
    g.fillStyle = REPORT.surface;
    g.fillRect(0, 0, W, H);
    const text = (str, x, y, { size = 9, weight = "normal", color = REPORT.ink2, align = "left", baseline = "alphabetic" } = {}) => {
      g.font = `${weight} ${Math.round((size * scale) / 72)}px ${REPORT.font}`;
      g.fillStyle = color;
      g.textAlign = align;
      g.textBaseline = baseline;
      g.fillText(str, x, y);
    };
    const label = (state) => REPORT_STATUS[state].charAt(0) + REPORT_STATUS[state].slice(1).toLowerCase();

    text("WATCH session report", 0.06 * W, 0.07 * H, { size: 16, weight: "bold", color: REPORT.ink });
    text(this.summary(), 0.06 * W, 0.125 * H, { size: 9.5 });
    const seconds = this.timeInStates();
    const totals = Object.keys(REPORT.levels).filter((s) => seconds[s] >= 0.5).map((s) => `${label(s)} ${reportClock(seconds[s])}`).join("  ·  ");
    text("Time in each state:  " + (totals || "-"), 0.06 * W, 0.17 * H, { size: 9.5 });
    text("Alertness over time", 0.06 * W, 0.26 * H, { size: 11, weight: "bold", color: REPORT.ink });

    const left = 0.11 * W, right = 0.82 * W, top = 0.32 * H, bottom = 0.85 * H;
    const end = this.samples.length ? this.samples[this.samples.length - 1].elapsed + this.interval : 1;
    const tMax = Math.max(end, 1);
    const x = (t) => left + (t / tMax) * (right - left);
    const y = (v) => bottom - ((v + 0.4) / 2.8) * (bottom - top);
    const runs = [];
    for (let i = 0, start = 0; i < this.samples.length; i++) {
      const next = this.samples[i + 1];
      if (next && next.state === this.samples[start].state) continue;
      runs.push({ state: this.samples[start].state, begin: this.samples[start].elapsed, finish: next ? next.elapsed : end });
      start = i + 1;
    }

    for (const run of runs) {
      if (run.state in REPORT_LEVEL_VALUE) continue;
      g.globalAlpha = 0.25;
      g.fillStyle = REPORT.levels[run.state];
      g.fillRect(x(run.begin), top, x(run.finish) - x(run.begin), bottom - top);
      g.globalAlpha = 1;
    }
    g.lineWidth = 1;
    for (const [state, value] of Object.entries(REPORT_LEVEL_VALUE)) {
      g.strokeStyle = REPORT.grid;
      g.beginPath();
      g.moveTo(left, Math.round(y(value)) + 0.5);
      g.lineTo(right, Math.round(y(value)) + 0.5);
      g.stroke();
      text(label(state), left - 12, y(value), { size: 8.5, color: REPORT.muted, align: "right", baseline: "middle" });
    }
    g.strokeStyle = REPORT.axis;
    g.beginPath();
    g.moveTo(left, bottom + 0.5);
    g.lineTo(right, bottom + 0.5);
    g.stroke();

    let previous = null;
    g.lineWidth = 2;
    g.strokeStyle = REPORT.axis;
    for (const run of runs) {
      if (!(run.state in REPORT_LEVEL_VALUE)) {
        previous = null;
        continue;
      }
      const value = REPORT_LEVEL_VALUE[run.state];
      if (previous && previous.finish === run.begin) {
        g.beginPath();
        g.moveTo(x(run.begin), y(previous.value));
        g.lineTo(x(run.begin), y(value));
        g.stroke();
      }
      previous = { finish: run.finish, value };
    }
    g.lineWidth = 4.5;
    for (const run of runs) {
      if (!(run.state in REPORT_LEVEL_VALUE)) continue;
      g.strokeStyle = REPORT.levels[run.state];
      g.beginPath();
      g.moveTo(x(run.begin), y(REPORT_LEVEL_VALUE[run.state]));
      g.lineTo(x(run.finish), y(REPORT_LEVEL_VALUE[run.state]));
      g.stroke();
    }

    const step = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600].find((s) => tMax / s <= 8) || 7200;
    for (let v = 0; v <= tMax + 1e-9; v += step) text(reportClock(v), x(v), bottom + 0.045 * H, { size: 8, color: REPORT.muted, align: "center" });
    text("time since start (m:ss)", (left + right) / 2, bottom + 0.09 * H, { size: 8.5, color: REPORT.muted, align: "center" });

    const used = new Set(runs.map((r) => r.state));
    const legend = [...Object.keys(REPORT_LEVEL_VALUE).filter((s) => used.has(s)).map((s) => [s, "line"]),
      ...Object.keys(REPORT.levels).filter((s) => !(s in REPORT_LEVEL_VALUE) && used.has(s)).map((s) => [s, "area"])];
    legend.forEach(([state, kind], i) => {
      const ly = top + 0.035 * H + i * 0.052 * H, lx = 0.835 * W;
      g.fillStyle = REPORT.levels[state];
      if (kind === "line") g.fillRect(lx, ly - 3, 48, 6);
      else {
        g.globalAlpha = 0.25;
        g.fillRect(lx, ly - 10, 48, 20);
        g.globalAlpha = 1;
      }
      text(label(state), lx + 66, ly, { size: 8.5, baseline: "middle" });
    });
  }
}
