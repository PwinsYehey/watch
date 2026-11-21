const $ = (id) => document.getElementById(id);
const output = $("output");
const placeholder = $("placeholder");
const originalPanel = $("originalPanel");
const originalOutput = $("originalOutput");
const scannedLabel = $("scannedLabel");
const captureHint = $("captureHint");
const flashOverlay = $("flashOverlay");
const startBtn = $("startBtn");
const startBtnLabel = $("startBtnLabel");
const cameraIcon = $("cameraIcon");
const liveDot = $("liveDot");
const uploadBtn = $("uploadBtn");
const imageUpload = $("imageUpload");
const muteBtn = $("muteBtn");
const checkInBtn = $("checkInBtn");
const exportBtn = $("exportBtn");
const statusEl = $("status");
const dashboard = $("dashboard");
const shutdownBtn = $("shutdownBtn");

let audioCtx = null;
let cameraActive = null;
let showingPhoto = false;
let muted = false;
let pendingExport = null;
let lastBeep = 0;
let lastState = null;
let capturing = false;
let contactLost = false;

const send = (key) => fetch("/key/" + key, { method: "POST" });

function ensureAudio() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  audioCtx.resume();
}

function playTone(freq, startOffset, duration) {
  if (!audioCtx) return;
  const t = audioCtx.currentTime + startOffset;
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type = "square";
  osc.frequency.value = freq;
  gain.gain.setValueAtTime(0.3, t);
  gain.gain.exponentialRampToValueAtTime(0.001, t + duration);
  osc.connect(gain).connect(audioCtx.destination);
  osc.start(t);
  osc.stop(t + duration);
}

function playShutterSound() {
  playTone(1400, 0, 0.03);
  playTone(700, 0.04, 0.04);
}

function playStartSound() {
  playTone(440, 0, 0.08);
  playTone(880, 0.09, 0.08);
}

function playStopSound() {
  playTone(880, 0, 0.08);
  playTone(440, 0.09, 0.08);
}

function flashCapture() {
  flashOverlay.style.transition = "none";
  flashOverlay.style.opacity = "1";
  requestAnimationFrame(() => {
    flashOverlay.style.transition = "opacity 400ms ease-out";
    flashOverlay.style.opacity = "0";
  });
}

function setDashboardVisible(visible) {
  if (dashboard.hidden === !visible) return;
  const apply = () => {
    dashboard.hidden = !visible;
  };
  if (document.startViewTransition && !matchMedia("(prefers-reduced-motion: reduce)").matches) document.startViewTransition(apply);
  else apply();
}

for (const tab of document.querySelectorAll(".dash-tabs button")) {
  tab.addEventListener("click", () => {
    dashboard.dataset.tab = tab.dataset.tab;
    for (const other of document.querySelectorAll(".dash-tabs button")) other.setAttribute("aria-selected", String(other === tab));
  });
}

const escapeHtml = (text) => String(text).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

function rowsHtml(rows) {
  return rows
    .map((row) => `<span class="dot" style="background:${row.dot}"></span><span>${escapeHtml(row.label)}</span><span style="color:${row.color}">${escapeHtml(row.value)}</span><span class="dim">${escapeHtml(row.extra)}</span>`)
    .join("");
}

function renderDashboard(view) {
  const level = $("dashLevel");
  level.style.background = view.level.color;
  level.style.color = view.level.dark ? "#000" : "#fff";
  $("dashLevelName").textContent = view.level.name;
  $("dashLevelReason").textContent = view.level.reason;
  $("dashLevelTimer").textContent = view.level.timer;
  $("dashSigns").innerHTML = rowsHtml(view.signs);
  $("dashChecks").innerHTML = rowsHtml(view.checks);
  $("dashCalibration").innerHTML = rowsHtml(view.calibration);
  $("dashLive").innerHTML = rowsHtml(view.live);
  $("dashEvents").innerHTML = view.events.length
    ? view.events.map((e) => `<li style="color:${e.color}">${escapeHtml(e.text)}</li>`).join("")
    : '<li class="dim">none yet</li>';
}

function hidePhoto() {
  showingPhoto = false;
  originalPanel.hidden = true;
  scannedLabel.hidden = true;
}

function setCameraActive(active) {
  if (cameraActive === active) return;
  cameraActive = active;
  if (active) {
    hidePhoto();
    output.src = "/stream";
    placeholder.hidden = true;
    startBtnLabel.textContent = "Stop camera";
  } else {
    if (!showingPhoto) {
      output.removeAttribute("src");
      placeholder.hidden = false;
    }
    startBtnLabel.textContent = "Use camera";
  }
  output.classList.toggle("capturable", active);
  captureHint.hidden = !active;
  cameraIcon.style.display = active ? "none" : "";
  liveDot.hidden = !active;
  muteBtn.hidden = !active;
  exportBtn.hidden = !active;
  if (!active) checkInBtn.hidden = true;
  imageUpload.disabled = active;
  uploadBtn.classList.toggle("disabled", active);
  setDashboardVisible(active);
}

async function poll() {
  try {
    const state = await (await fetch("/state", { cache: "no-store" })).json();
    lastState = state;
    shutdownBtn.hidden = !state.shutdown;
    if (capturing && state.paused) capturing = false;
    if (!capturing) setCameraActive(!state.paused);
    if (state.view && !state.paused) {
      renderDashboard(state.view);
      muted = state.view.muted;
      muteBtn.textContent = muted ? "Unmute Alarm" : "Mute Alarm";
    }
    checkInBtn.hidden = !(cameraActive && state.check_in);
    if (state.alarm && !muted && performance.now() - lastBeep > 500) {
      lastBeep = performance.now();
      playTone(880, 0, 0.25);
    }
    if (pendingExport !== null && state.export && state.export.id !== pendingExport) {
      pendingExport = null;
      for (const name of state.export.files) download("/sessions/" + name, name);
      statusEl.textContent = "";
    }
  } catch (error) {
    contactLost = true;
    statusEl.textContent = "Lost contact with the Pi, retrying...";
    return;
  }
  if (contactLost) {
    contactLost = false;
    reconnectVideo();
  }
  if (statusEl.textContent.startsWith("Lost contact")) statusEl.textContent = "";
}

function reconnectVideo() {
  if (cameraActive && !showingPhoto) output.src = "/stream?" + Date.now();
}

output.addEventListener("error", () => {
  if (cameraActive && !showingPhoto) setTimeout(reconnectVideo, 1000);
});

function download(url, filename) {
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
}

async function analyze(body) {
  statusEl.textContent = "Checking the photo on the Pi...";
  const response = await fetch("/analyze", { method: "POST", body });
  if (!response.ok) {
    statusEl.textContent = `Error: ${(await response.text()) || "could not check that photo"}`;
    return false;
  }
  const result = await response.json();
  showingPhoto = true;
  originalOutput.src = result.original;
  output.src = result.scanned;
  originalPanel.hidden = false;
  scannedLabel.hidden = false;
  placeholder.hidden = true;
  statusEl.textContent = "";
  return true;
}

function runImageUpload(file) {
  if (cameraActive) return;
  analyze(file).catch((err) => {
    statusEl.textContent = `Error: ${err.message}`;
  });
}

startBtn.addEventListener("click", () => {
  ensureAudio();
  if (cameraActive) playStopSound();
  else playStartSound();
  send("p");
});

output.addEventListener("click", async () => {
  if (!cameraActive) return;
  ensureAudio();
  flashCapture();
  playShutterSound();
  capturing = true;
  if (await analyze(null).catch(() => false)) {
    await send("p");
    setTimeout(() => (capturing = false), 3000);
  } else {
    capturing = false;
  }
});

imageUpload.addEventListener("change", () => {
  const file = imageUpload.files[0];
  if (file) runImageUpload(file);
  imageUpload.value = "";
});

document.addEventListener("paste", (event) => {
  const items = event.clipboardData ? event.clipboardData.items : [];
  for (const item of items) {
    if (item.type.startsWith("image/")) {
      const file = item.getAsFile();
      if (file) runImageUpload(file);
      break;
    }
  }
});

muteBtn.addEventListener("click", () => {
  ensureAudio();
  send("m");
});

checkInBtn.addEventListener("click", () => send("a"));

document.addEventListener("keydown", (event) => {
  if (event.code === "Space" && !checkInBtn.hidden) {
    event.preventDefault();
    send("a");
  }
});

exportBtn.addEventListener("click", () => {
  pendingExport = lastState && lastState.export ? lastState.export.id : 0;
  statusEl.textContent = "Preparing the report...";
  send("e");
});

shutdownBtn.addEventListener("click", async () => {
  if (!confirm("Shut down the device? It has to be unplugged and plugged back in to start again.")) return;
  const response = await fetch("/shutdown", { method: "POST" });
  statusEl.textContent = response.ok ? "Shutting down. Wait for the green light to stop blinking, then unplug the power." : "Could not shut down: " + (await response.text());
});

poll();
setInterval(poll, 250);
