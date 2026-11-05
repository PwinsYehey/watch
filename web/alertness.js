// Browser port of src/watch/alertness.py, keep behaviour identical.

// Mirrors head_droop in src/watch/alertness.py.
function headDroop(pitch, roll, cfg) {
  const past = (value, upright, drooped) => (drooped === upright ? 0 : Math.max(0, (value - upright) / (drooped - upright)));
  const down = past(pitch, cfg.uprightThreshold, cfg.droopedThreshold);
  const side = past(roll, cfg.rollUprightThreshold, cfg.rollDroopedThreshold);
  return Math.min(1, Math.hypot(down, side));
}

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

const CUE_LABELS = { eyes: "eyes closed", mouth: "yawning", head: "head drooped", body: "slumped", checkIn: "no check-in answer" };

// Mirrors EscalationMonitor in src/watch/alertness.py.
class CheckInMonitor {
  constructor(blindSec, answerSec) {
    this.blind = blindSec;
    this.answer = answerSec;
    this.blindSince = null;
    this.askedAt = null;
    this.lastReaction = null;
    this.justAnswered = false;
  }

  get asking() {
    return this.askedAt !== null;
  }

  update(eyesReadable, personPresent, answered, now) {
    this.justAnswered = false;
    if (!personPresent) {
      this.blindSince = this.askedAt = null;
      return { state: "no_face", closedDuration: 0, perclos: 0 };
    }
    if (eyesReadable) {
      this.blindSince = this.askedAt = null;
      return { state: "alert", closedDuration: 0, perclos: 0 };
    }
    if (this.blindSince === null) this.blindSince = now;
    if (this.askedAt !== null && answered) {
      this.lastReaction = now - this.askedAt;
      this.justAnswered = true;
      this.askedAt = null;
      this.blindSince = now;
      return { state: "alert", closedDuration: 0, perclos: 0 };
    }
    if (this.askedAt === null && now - this.blindSince >= this.blind) this.askedAt = now;
    const overdue = this.askedAt !== null ? now - this.askedAt - this.answer : -1;
    return overdue >= 0 ? { state: "not_alert", closedDuration: overdue, perclos: 0 } : { state: "alert", closedDuration: 0, perclos: 0 };
  }
}

class EscalationMonitor {
  constructor({ windowSec, calmSec, persistSec, repeatCount }) {
    this.window = windowSec;
    this.calm = calmSec;
    this.persist = persistSec;
    this.repeat = repeatCount;
    this.level = 0;
    this.reason = "";
    this.events = [];
    this.active = {};
    this.windowEnd = 0.0;
    this.calmSince = null;
    this.last = null;
  }

  update(statuses, facePresent, now) {
    const elapsed = this.last !== null ? now - this.last : 0.0;
    this.last = now;
    if (!facePresent) {
      this.windowEnd += elapsed;
      this.calmSince = null;
      return this._status(now);
    }

    const concerning = Object.keys(statuses).filter((cue) => statuses[cue].state === "not_alert");
    const newEvents = concerning.filter((cue) => !this.active[cue]);
    for (const [cue, status] of Object.entries(statuses)) {
      if (status.state !== "no_face") this.active[cue] = status.state === "not_alert";
    }
    this.events.push(...newEvents);

    if (this.level === 2) {
      if (concerning.length) this.calmSince = null;
      else if (this.calmSince === null) this.calmSince = now;
      else if (now - this.calmSince >= this.calm) {
        this.level = 1;
        this.reason = "recovering";
        this.windowEnd = now + this.window;
      }
      return this._status(now);
    }

    if (newEvents.length) {
      this.windowEnd = now + this.window;
      if (this.level === 0) {
        this.level = 1;
        this.reason = newEvents.map((cue) => CUE_LABELS[cue]).join(" + ");
      }
      const reason = this._repeatedOrCombined();
      if (reason) return this._critical(reason, now);
    }
    if (this.level === 1) {
      for (const cue of concerning) {
        if (this.persist[cue] !== null && statuses[cue].closedDuration >= this.persist[cue]) return this._critical(`${CUE_LABELS[cue]} too long`, now);
      }
      if (!concerning.length && now >= this.windowEnd) {
        this.level = 0;
        this.reason = "";
        this.events = [];
      }
    }
    return this._status(now);
  }

  _repeatedOrCombined() {
    const cues = Object.keys(CUE_LABELS).filter((cue) => this.events.includes(cue));
    if (cues.length > 1) return cues.map((cue) => CUE_LABELS[cue]).join(" + ");
    const cue = cues[0];
    return this.events.filter((e) => e === cue).length >= this.repeat[cue] ? `${CUE_LABELS[cue]} again` : "";
  }

  _critical(reason, now) {
    this.level = 2;
    this.reason = reason;
    this.calmSince = null;
    return this._status(now);
  }

  _status(now) {
    const windowLeft = this.level === 1 ? Math.max(0.0, this.windowEnd - now) : 0.0;
    return { level: this.level, reason: this.reason, windowLeft };
  }
}
