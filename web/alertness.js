// Browser port of src/watch/alertness.py, keep behaviour identical.

class RollingMedian {
  constructor(size) {
    this.size = size;
    this.values = [];
  }

  add(value) {
    if (value === null) return null;
    this.values.push(value);
    if (this.values.length > this.size) this.values.shift();
    const sorted = [...this.values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }
}

class AlertnessMonitor {
  constructor({ closedDurationThreshold, perclosWindow, perclosThreshold, closednessThreshold, dipGracePeriod = 0.0 }) {
    this.closedDurationThreshold = closedDurationThreshold;
    this.perclosWindow = perclosWindow;
    this.perclosThreshold = perclosThreshold;
    this.closednessThreshold = closednessThreshold;
    this.dipGracePeriod = dipGracePeriod;
    this.closedSince = null;
    this.belowSince = null;
    this.history = []; // [{t, score}]
  }

  update(closedness, now) {
    if (closedness === null) {
      return { state: "no_face", closedDuration: 0.0, perclos: this._perclos(now) };
    }

    const isClosedNow = closedness >= this.closednessThreshold;
    if (isClosedNow) {
      this.belowSince = null;
      if (this.closedSince === null) this.closedSince = now;
    } else {
      if (this.belowSince === null) this.belowSince = now;
      if (now - this.belowSince > this.dipGracePeriod) this.closedSince = null;
    }

    this.history.push({ t: now, score: closedness });
    this._trimHistory(now);

    const closedDuration = this.closedSince !== null ? now - this.closedSince : 0.0;
    const perclos = this._perclos(now);
    const windowFilled = now - this.history[0].t >= this.perclosWindow;

    const isNotAlert = closedDuration >= this.closedDurationThreshold || (windowFilled && perclos >= this.perclosThreshold);
    return { state: isNotAlert ? "not_alert" : "alert", closedDuration, perclos };
  }

  _trimHistory(now) {
    while (this.history.length && now - this.history[0].t > this.perclosWindow) this.history.shift();
  }

  _perclos(now) {
    this._trimHistory(now);
    if (!this.history.length) return 0.0;
    return this.history.reduce((sum, h) => sum + h.score, 0) / this.history.length;
  }
}
