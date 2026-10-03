// Browser port of src/watch/report.py: records the session and exports a PNG report with graphs and a CSV.

const REPORT = {
  surface: "#fcfcfb", grid: "#e1e0d9", axis: "#c3c2b7",
  ink: "#0b0b0b", ink2: "#52514e", muted: "#898781",
  series1: "#2a78d6", series2: "#eb6834",
  levels: { Normal: "#0ca30c", Warning: "#fab219", Critical: "#d03b3b", "No face": "#898781", Calibrating: "#c3c2b7" },
  font: 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
};
const REPORT_LEVEL_NAMES = { 0: "Normal", 1: "Warning", 2: "Critical" };

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

  summary() {
    if (!this.samples.length) return "No data recorded";
    const seconds = { Critical: 0 };
    let warnings = 0, criticals = 0;
    for (let i = 1; i < this.samples.length; i++) {
      const current = this.samples[i - 1], following = this.samples[i];
      seconds[current.state] = (seconds[current.state] || 0) + following.elapsed - current.elapsed;
      if (following.state === "Warning" && (current.state === "Normal" || current.state === "No face")) warnings++;
      if (following.state === "Critical" && current.state !== "Critical") criticals++;
    }
    const started = new Date(this.samples[0].clock * 1000);
    return `Started ${started.toLocaleDateString("en-CA")} ${started.toLocaleTimeString([], { hour12: false })}  ·  duration ${reportClock(this.samples[this.samples.length - 1].elapsed)}`
      + `  ·  warnings ${warnings}  ·  critical alerts ${criticals}  ·  time in critical ${reportClock(seconds.Critical)}`;
  }

  csv() {
    const columns = ["time", "elapsed", "state", "ear", "mar", "head_down", "head_tilt", "ear_closed_limit", "head_down_limit", "head_tilt_limit", "event"];
    const keys = ["clock", "elapsed", "state", "ear", "mar", "headDown", "headTilt", "earClosedLimit", "headDownLimit", "headTiltLimit", "event"];
    const cell = (value) => (value === null ? "" : typeof value === "number" ? String(Math.round(value * 10000) / 10000) : `"${String(value).replace(/"/g, '""')}"`);
    const rows = this.samples.map((s) => keys.map((key, i) => (i === 0 ? new Date(s.clock * 1000).toLocaleTimeString([], { hour12: false }) : cell(s[key]))).join(","));
    return [columns.join(","), ...rows].join("\n");
  }

  // Mirrors SessionLog._write_png in src/watch/report.py.
  drawReport(canvas) {
    const W = 1800, H = 1290;
    canvas.width = W;
    canvas.height = H;
    const g = canvas.getContext("2d");
    g.fillStyle = REPORT.surface;
    g.fillRect(0, 0, W, H);
    const left = 126, right = 1584, t = this.samples.map((s) => s.elapsed);
    const tMax = Math.max(t.length ? t[t.length - 1] : 0, 1);
    const x = (value) => left + (value / tMax) * (right - left);
    const text = (str, px, py, { size = 16, weight = "normal", color = REPORT.ink2, align = "left", baseline = "alphabetic" } = {}) => {
      g.font = `${weight} ${size}px ${REPORT.font}`;
      g.fillStyle = color;
      g.textAlign = align;
      g.textBaseline = baseline;
      g.fillText(str, px, py);
    };

    text("WATCH session report", left, 66, { size: 32, weight: "bold", color: REPORT.ink });
    text(this.summary(), left, 108, { size: 18 });

    // Panel layout follows the PNG from report.py: a short alert-level band, then eyes, mouth and head.
    const panels = [];
    let top = 180;
    for (const [title, height] of [["Alert level", 72], ["Eyes", 220], ["Mouth", 220], ["Head", 260]]) {
      panels.push({ title, top, bottom: top + height });
      top += height + 84;
    }

    const legend = (items, py) => {
      items.forEach(([label, color, kind], i) => {
        const ly = py + i * 30;
        g.fillStyle = color;
        g.strokeStyle = color;
        if (kind === "line") {
          g.lineWidth = 3;
          g.beginPath();
          g.moveTo(1606, ly);
          g.lineTo(1636, ly);
          g.stroke();
        } else {
          g.fillRect(1606, ly - 8, 22, 16);
        }
        text(label, 1646, ly, { size: 17, baseline: "middle" });
      });
    };

    // 1. Alert level band.
    const band = panels[0];
    text(band.title, left, band.top - 16, { size: 21, weight: "bold", color: REPORT.ink });
    // One block per run of the same state, on whole pixels, so the band has no seams.
    const used = new Set();
    for (let i = 0, start = 0; i < this.samples.length; i++) {
      const next = this.samples[i + 1];
      if (next && next.state === this.samples[start].state) continue;
      const begin = Math.round(x(this.samples[start].elapsed));
      const end = Math.round(x(next ? next.elapsed : this.samples[i].elapsed + this.interval));
      g.fillStyle = REPORT.levels[this.samples[start].state];
      g.fillRect(begin, band.top, Math.max(Math.min(end, right) - begin, 1), band.bottom - band.top);
      used.add(this.samples[start].state);
      start = i + 1;
    }
    legend(Object.keys(REPORT.levels).filter((state) => used.has(state)).map((state) => [state, REPORT.levels[state], "box"]), band.top + 8);

    const niceTicks = (lo, hi, count = 4) => {
      const span = hi - lo || 1, rough = span / count;
      const unit = 10 ** Math.floor(Math.log10(rough));
      const step = [1, 2, 2.5, 5, 10].map((m) => m * unit).find((s) => s >= rough);
      const ticks = [];
      const last = Math.ceil(hi / step - 1e-9) * step;
      for (let v = Math.floor(lo / step) * step; v <= last + step * 1e-9; v += step) ticks.push(Math.round(v * 1e6) / 1e6);
      return ticks;
    };

    const plot = (panel, unit, seriesList, limits, floorAtZero) => {
      text(panel.title, left, panel.top - 16, { size: 21, weight: "bold", color: REPORT.ink });
      const values = [...seriesList, ...limits].flatMap(([key]) => this.samples.map((s) => s[key])).filter((v) => v !== null);
      let lo = values.length ? Math.min(...values) : 0, hi = values.length ? Math.max(...values) : 1;
      if (floorAtZero) lo = Math.min(lo, 0);
      const pad = (hi - lo || 1) * 0.08;
      const ticks = niceTicks(lo - (floorAtZero ? 0 : pad), hi + pad);
      const yLo = ticks[0], yHi = ticks[ticks.length - 1];
      const y = (value) => panel.bottom - ((value - yLo) / (yHi - yLo || 1)) * (panel.bottom - panel.top);
      g.lineWidth = 1;
      for (const tick of ticks) {
        g.strokeStyle = tick === yLo ? REPORT.axis : REPORT.grid;
        g.beginPath();
        g.moveTo(left, Math.round(y(tick)) + 0.5);
        g.lineTo(right, Math.round(y(tick)) + 0.5);
        g.stroke();
        text(String(tick), left - 12, y(tick), { size: 15, color: REPORT.muted, align: "right", baseline: "middle" });
      }
      g.save();
      g.translate(left - 72, (panel.top + panel.bottom) / 2);
      g.rotate(-Math.PI / 2);
      text(unit, 0, 0, { size: 15, color: REPORT.muted, align: "center" });
      g.restore();

      // Calibrated limits as labelled step lines, since recalibration can move them.
      for (const [key, label] of limits) {
        const points = this.samples.filter((s) => s[key] !== null);
        if (!points.length) continue;
        g.strokeStyle = REPORT.muted;
        g.lineWidth = 1.5;
        g.setLineDash([8, 6]);
        g.beginPath();
        points.forEach((s, i) => {
          if (i === 0) g.moveTo(x(s.elapsed), y(s[key]));
          else {
            g.lineTo(x(s.elapsed), y(points[i - 1][key]));
            g.lineTo(x(s.elapsed), y(s[key]));
          }
        });
        g.lineTo(right, y(points[points.length - 1][key]));
        g.stroke();
        g.setLineDash([]);
        text(label(points[points.length - 1][key]), right + 10, y(points[points.length - 1][key]), { size: 15, baseline: "middle" });
      }

      for (const [key, color] of seriesList) {
        g.strokeStyle = color;
        g.lineWidth = 3;
        g.lineJoin = "round";
        g.lineCap = "round";
        g.beginPath();
        let drawing = false;
        for (const s of this.samples) {
          if (s[key] === null) {
            drawing = false;
            continue;
          }
          if (drawing) g.lineTo(x(s.elapsed), y(s[key]));
          else g.moveTo(x(s.elapsed), y(s[key]));
          drawing = true;
        }
        g.stroke();
      }
    };

    for (const s of this.samples) s.yawnLevel = 0.45;
    plot(panels[1], "EAR", [["ear", REPORT.series1]], [["earClosedLimit", (v) => `closed limit ${v.toFixed(2)}`]], true);
    plot(panels[2], "MAR", [["mar", REPORT.series1]], [["yawnLevel", (v) => `yawn level ${v.toFixed(2)}`]], true);
    plot(panels[3], "degrees", [["headDown", REPORT.series1], ["headTilt", REPORT.series2]], [
      ["headDownLimit", (v) => `down limit ${v.toFixed(1)}°`],
      ["headTiltLimit", (v) => `tilt limit ${v.toFixed(1)}°`],
    ], false);
    // Head legend sits inline beside its title, clear of the limit labels on the right.
    g.font = `bold 21px ${REPORT.font}`;
    let lx = left + g.measureText("Head").width + 28;
    for (const [label, color] of [["Down", REPORT.series1], ["Tilt", REPORT.series2]]) {
      g.strokeStyle = color;
      g.lineWidth = 3;
      g.beginPath();
      g.moveTo(lx, panels[3].top - 23);
      g.lineTo(lx + 28, panels[3].top - 23);
      g.stroke();
      text(label, lx + 36, panels[3].top - 23, { size: 17, baseline: "middle" });
      g.font = `normal 17px ${REPORT.font}`;
      lx += 36 + g.measureText(label).width + 24;
    }

    // Shared time axis under the last panel.
    const last = panels[3];
    const step = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600].find((s) => tMax / s <= 8) || 7200;
    for (let v = 0; v <= tMax + 1e-9; v += step) text(reportClock(v), x(v), last.bottom + 26, { size: 15, color: REPORT.muted, align: "center" });
    text("time since start (m:ss)", (left + right) / 2, last.bottom + 60, { size: 15, color: REPORT.muted, align: "center" });
  }
}
