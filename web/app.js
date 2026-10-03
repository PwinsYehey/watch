// Browser port of src/watch/live.py using MediaPipe Tasks Vision and TensorFlow.js.
import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const DISPLAY_WIDTH = 640;
const DISPLAY_HEIGHT = 480;

const video = document.getElementById("video");
const displayCanvas = document.getElementById("output");
const displayCtx = displayCanvas.getContext("2d");
const originalCanvas = document.getElementById("originalOutput");
const originalCtx = originalCanvas.getContext("2d");
const originalPanel = document.getElementById("originalPanel");
const scannedLabel = document.getElementById("scannedLabel");
const captureHint = document.getElementById("captureHint");
const flashOverlay = document.getElementById("flashOverlay");
const placeholder = document.getElementById("placeholder");
const dashboard = document.getElementById("dashboard");

const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");

const mirrorCanvas = document.createElement("canvas");
const mirrorCtx = mirrorCanvas.getContext("2d");

function updateMirror() {
  mirrorCtx.save();
  mirrorCtx.scale(-1, 1);
  mirrorCtx.drawImage(video, -mirrorCanvas.width, 0, mirrorCanvas.width, mirrorCanvas.height);
  mirrorCtx.restore();
}

// Glare-cleaned copy of the frame used only for tracking and model input.
const glareCleanCanvas = document.createElement("canvas");
const glareCleanCtx = glareCleanCanvas.getContext("2d", { willReadFrequently: true });
const glareBlurCanvas = document.createElement("canvas");
const glareBlurCtx = glareBlurCanvas.getContext("2d");

// The current frame's pixels and glare mask, reused by the glare and low-detail checks instead of reading the canvas again.
let framePixels = null;
let frameWidth = 0;
let glareMask = new Uint8Array(0);

function fitCanvas(target, w, h) {
  if (target.width !== w || target.height !== h) {
    target.width = w;
    target.height = h;
  }
}

// Replaces glare pixels with a blurred copy, standing in for the OpenCV inpaint used in live.py, then lifts low light.
// When box is given, only glare inside it is replaced, matching remove_glare in live.py.
function removeGlare(sourceCanvas, targetCtx, inpaint = true, box = null) {
  const w = sourceCanvas.width, h = sourceCanvas.height;
  fitCanvas(targetCtx.canvas, w, h);
  targetCtx.drawImage(sourceCanvas, 0, 0);
  const image = targetCtx.getImageData(0, 0, w, h);
  const data = image.data;

  if (glareMask.length !== w * h) glareMask = new Uint8Array(w * h);
  let glareCount = 0;
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    glareMask[j] = isGlarePixel(data[i], data[i + 1], data[i + 2]) ? 1 : 0;
    glareCount += glareMask[j];
  }

  const { x0, y0, x1, y1 } = box || { x0: 0, y0: 0, x1: w, y1: h };
  if (box) {
    glareCount = 0;
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) glareCount += glareMask[y * w + x];
  }

  let changed = false;
  if (inpaint && glareCount) {
    fitCanvas(glareBlurCanvas, w, h);
    glareBlurCtx.filter = "blur(6px)";
    glareBlurCtx.drawImage(sourceCanvas, 0, 0);
    glareBlurCtx.filter = "none";
    const blurData = glareBlurCtx.getImageData(0, 0, w, h).data;
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const j = y * w + x, i = j * 4;
        if (!glareMask[j]) continue;
        data[i] = blurData[i];
        data[i + 1] = blurData[i + 1];
        data[i + 2] = blurData[i + 2];
      }
    }
    changed = true;
  }

  const brightness = liftLowLight(data);
  if (changed || (brightness > 0 && brightness < CONFIG.live.lowLightMean)) targetCtx.putImageData(image, 0, 0);
  framePixels = data;
  frameWidth = w;
  return brightness;
}

// Mirrors brighten in src/watch/face.py.
function liftLowLight(data) {
  let sum = 0, count = 0;
  for (let i = 0; i < data.length; i += 64) {
    sum += data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
    count++;
  }
  const mean = sum / count / 255, target = CONFIG.live.lowLightTarget;
  if (mean >= CONFIG.live.lowLightMean || mean <= 0) return mean;
  const gamma = Math.log(target) / Math.log(mean);
  const lut = new Uint8ClampedArray(256);
  for (let v = 0; v < 256; v++) lut[v] = Math.pow(v / 255, gamma) * 255;
  for (let i = 0; i < data.length; i += 4) {
    data[i] = lut[data[i]];
    data[i + 1] = lut[data[i + 1]];
    data[i + 2] = lut[data[i + 2]];
  }
  return mean;
}

// Mirrors face_box in live.py.
function faceBox(landmarks, w, h, margin = 0.3) {
  const xs = landmarks.map((p) => p.x * w), ys = landmarks.map((p) => p.y * h);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const padX = (maxX - minX) * margin, padY = (maxY - minY) * margin;
  const box = {
    x0: Math.max(Math.floor(minX - padX), 0), y0: Math.max(Math.floor(minY - padY), 0),
    x1: Math.min(Math.ceil(maxX + padX), w), y1: Math.min(Math.ceil(maxY + padY), h),
  };
  return box.x1 > box.x0 && box.y1 > box.y0 ? box : null;
}

function isGlarePixel(r, g, b) {
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  return max > 235 && max - min < 60;
}

function boxRange(box) {
  const x0 = Math.max(0, Math.floor(box.xMin)), y0 = Math.max(0, Math.floor(box.yMin));
  const x1 = Math.min(frameWidth, Math.ceil(box.xMax)), y1 = Math.min(framePixels.length / 4 / frameWidth, Math.ceil(box.yMax));
  return x1 > x0 && y1 > y0 ? { x0, y0, x1, y1 } : null;
}

// Fraction of a box on the original frame that is glare.
function glareFraction(box) {
  const r = boxRange(box);
  if (!r) return 0.0;
  let glareCount = 0;
  for (let y = r.y0; y < r.y1; y++) {
    for (let x = r.x0; x < r.x1; x++) glareCount += glareMask[y * frameWidth + x];
  }
  return glareCount / ((r.x1 - r.x0) * (r.y1 - r.y0));
}

// True if a box is too flat to be real facial detail, such as an opaque visor.
function isLowDetail(box) {
  const r = boxRange(box);
  if (!r) return true;
  let sum = 0, sumSq = 0;
  for (let y = r.y0; y < r.y1; y++) {
    for (let x = r.x0, i = (y * frameWidth + r.x0) * 4; x < r.x1; x++, i += 4) {
      const gray = framePixels[i] * 0.299 + framePixels[i + 1] * 0.587 + framePixels[i + 2] * 0.114;
      sum += gray;
      sumSq += gray * gray;
    }
  }
  const n = (r.x1 - r.x0) * (r.y1 - r.y0), mean = sum / n;
  return Math.sqrt(Math.max(0, sumSq / n - mean * mean)) < CONFIG.live.lowDetailStd;
}

const statusEl = document.getElementById("status");
const startBtn = document.getElementById("startBtn");
const startBtnLabel = document.getElementById("startBtnLabel");
const cameraIcon = document.getElementById("cameraIcon");
const liveDot = document.getElementById("liveDot");
const uploadBtn = document.getElementById("uploadBtn");
const muteBtn = document.getElementById("muteBtn");
const exportBtn = document.getElementById("exportBtn");
let session = new SessionLog();
let sessionLevel = 0;
const imageUpload = document.getElementById("imageUpload");

let cameraActive = false;
let cameraStream = null;

const OPEN_COLOR = "#00c800";
const CLOSED_COLOR = "#dc0000";
const MOUTH_COLOR = "#ffc800";
const HEAD_COLOR = "#ff00c8";
const BANNER_COLORS = { alert: "#00c800", not_alert: "#dc0000", no_face: "#ffa500" };
const BANNER_TEXT = { alert: "ALERT", not_alert: "NOT ALERT", no_face: "NO FACE DETECTED" };
// Mirrors BANNERS in src/watch/live.py.
const LEVEL_BANNERS = {
  0: { text: "NORMAL", color: "#00c800" },
  1: { text: "WARNING", color: "#ffd700" },
  2: { text: "CRITICAL", color: "#dc0000" },
  no_face: { text: "NO FACE DETECTED", color: "#787878" },
};

function linearScore(value, lowAtZero, highAtOne) {
  const span = highAtOne - lowAtZero;
  if (span === 0) return 0.0;
  return Math.max(0.0, Math.min(1.0, (value - lowAtZero) / span));
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// Pulsing alarm tone that plays while the state is NOT ALERT.
let audioCtx = null;
let alarmOscillator = null;
let alarmGain = null;
let alarmPulseTimer = null;
let muted = false;

function startAlarm() {
  if (muted || alarmOscillator) return;
  audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
  alarmOscillator = audioCtx.createOscillator();
  alarmGain = audioCtx.createGain();
  alarmOscillator.type = "square";
  alarmOscillator.frequency.value = 880;
  alarmGain.gain.value = 1.0;
  alarmOscillator.connect(alarmGain).connect(audioCtx.destination);
  alarmOscillator.start();
  let on = true;
  alarmPulseTimer = setInterval(() => {
    on = !on;
    alarmGain.gain.value = on ? 1.0 : 0.0;
  }, 250);
}

function stopAlarm() {
  if (alarmPulseTimer) clearInterval(alarmPulseTimer);
  alarmPulseTimer = null;
  if (alarmOscillator) {
    alarmOscillator.stop();
    alarmOscillator.disconnect();
    alarmOscillator = null;
  }
  if (alarmGain) {
    alarmGain.disconnect();
    alarmGain = null;
  }
}

const cropCanvas = document.createElement("canvas");
cropCanvas.width = CONFIG.imageSize[0];
cropCanvas.height = CONFIG.imageSize[1];
const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });

function cropInput(sourceCanvas, crop, desaturate) {
  cropCtx.fillStyle = "#000";
  cropCtx.fillRect(0, 0, cropCanvas.width, cropCanvas.height);
  cropCtx.save();
  cropCtx.scale(cropCanvas.width / crop.width, cropCanvas.height / crop.height);
  cropCtx.translate(-crop.xMin, -crop.yMin);
  cropCtx.rotate(-crop.angle);
  cropCtx.translate(-crop.cx, -crop.cy);
  cropCtx.drawImage(sourceCanvas, 0, 0);
  cropCtx.restore();
  let input = tf.browser.fromPixels(cropCanvas).toFloat();
  if (desaturate) {
    const [r, g, b] = tf.split(input, 3, 2);
    const gray = r.mul(0.299).add(g.mul(0.587)).add(b.mul(0.114));
    input = tf.concat([gray, gray, gray], 2);
  }
  return input.div(255.0);
}

// Mirrors Classifier.predict in src/watch/model.py, running all of one model's crops as a single batch.
async function predictCrops(model, sourceCanvas, crops, classNames, positiveClass, desaturate = false) {
  const present = crops.filter(Boolean);
  if (!present.length) return crops.map(() => null);
  const output = tf.tidy(() => model.predict(tf.stack(present.map((crop) => cropInput(sourceCanvas, crop, desaturate)))));
  const values = await output.data();
  output.dispose();
  const size = classNames.length;
  let row = 0;
  return crops.map((crop) => {
    if (!crop) return null;
    const prediction = Array.from(values.slice(row * size, ++row * size));
    const predictedIndex = prediction.indexOf(Math.max(...prediction));
    return {
      label: classNames[predictedIndex],
      confidence: prediction[predictedIndex],
      positiveProbability: prediction[classNames.indexOf(positiveClass)],
    };
  });
}

async function predictRegions(regions, eyeModel, yawnModel) {
  const [[left, right], [mouth]] = await Promise.all([
    predictCrops(eyeModel, glareCleanCanvas, [regions.left?.crop, regions.right?.crop], CONFIG.eyeClassNames, "eyes_closed", true),
    predictCrops(yawnModel, glareCleanCanvas, [regions.mouth?.crop], CONFIG.yawnClassNames, "yawn"),
  ]);
  return { left, right, mouth };
}

let earCfg = CONFIG.ear;
let headPoseCfg = CONFIG.headPose;

let modelsPromise = null;
function ensureModels() {
  if (!modelsPromise) {
    modelsPromise = (async () => {
      tf.wasm.setWasmPaths("lib/");
      let backendOk = await tf.setBackend("webgl");
      if (!backendOk) backendOk = await tf.setBackend("wasm");
      await tf.ready();
      console.log("TF.js backend:", tf.getBackend());

      const eyeModel = await tf.loadLayersModel("models/eye/model.json");
      const yawnModel = await tf.loadLayersModel("models/yawn/model.json");

      tf.tidy(() => {
        const dummy = tf.zeros([1, ...CONFIG.imageSize, 3]);
        eyeModel.predict(dummy);
        yawnModel.predict(dummy);
      });

      const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
      const baseOptions = {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU",
      };
      const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions,
        runningMode: "VIDEO",
        numFaces: 1,
        outputFacialTransformationMatrixes: true,
        minFaceDetectionConfidence: CONFIG.live.faceDetectionConfidence,
        minTrackingConfidence: CONFIG.live.faceTrackingConfidence,
      });

      return { eyeModel, yawnModel, faceLandmarker, vision, baseOptions };
    })();
  }
  return modelsPromise;
}

let uploadFaceLandmarkerPromise = null;
function ensureUploadFaceLandmarker(vision, baseOptions) {
  if (!uploadFaceLandmarkerPromise) {
    uploadFaceLandmarkerPromise = FaceLandmarker.createFromOptions(vision, {
      baseOptions,
      runningMode: "IMAGE",
      numFaces: 1,
      outputFacialTransformationMatrixes: true,
      minFaceDetectionConfidence: 0.1,
      minFacePresenceConfidence: 0.1,
    });
  }
  return uploadFaceLandmarkerPromise;
}

let blazefacePromise = null;
function ensureBlazeface() {
  if (!blazefacePromise) {
    blazefacePromise = blazeface.load({ maxFaces: 1, scoreThreshold: 0.3 });
  }
  return blazefacePromise;
}

// Mirrors brackets in src/watch/live.py.
function drawBrackets(landmarks, connections, margin, color, lineWidth = 2) {
  const ids = new Set(connections.flatMap((c) => [c.start, c.end]));
  const xs = [...ids].map((i) => landmarks[i].x * canvas.width);
  const ys = [...ids].map((i) => landmarks[i].y * canvas.height);
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const padX = (x1 - x0) * margin, padY = (y1 - y0) * margin;
  x0 -= padX; x1 += padX; y0 -= padY; y1 += padY;
  const length = Math.max(6, Math.min(x1 - x0, y1 - y0) * 0.25);
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  for (const [x, y, dx, dy] of [[x0, y0, 1, 1], [x1, y0, -1, 1], [x0, y1, 1, -1], [x1, y1, -1, -1]]) {
    ctx.moveTo(x + dx * length, y);
    ctx.lineTo(x, y);
    ctx.lineTo(x, y + dy * length);
  }
  ctx.stroke();
}

// Forehead to chin, so sideways tilt is visible.
function drawHeadAxis(landmarks, color) {
  const [top, chin] = [10, 152].map((i) => ({ x: landmarks[i].x * canvas.width, y: landmarks[i].y * canvas.height }));
  ctx.strokeStyle = ctx.fillStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(top.x, top.y);
  ctx.lineTo(chin.x, chin.y);
  ctx.stroke();
  for (const p of [top, chin]) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, 2 * Math.PI);
    ctx.fill();
  }
}

function drawEye(side, box, prediction, ear, earCfg, landmarks, smoother) {
  const { confidence: cnnConfidence, positiveProbability: cnnClosedness } = prediction;
  if (cnnConfidence < CONFIG.live.minCnnConfidence) return null;

  // Glare corrupts EAR geometry, so ignore EAR on heavily glared eyes.
  const earClosedness = ear <= earCfg.closedThreshold && glareFraction(box) < CONFIG.live.earGlareTolerance ? 1.0 : 0.0;
  const rawClosedness = Math.max(earClosedness, cnnClosedness);
  const closedness = smoother ? smoother.add(rawClosedness) : rawClosedness;
  const label = closedness >= 0.5 ? "eyes_closed" : "eyes_opened";

  const connectors = side === "left" ? FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE : FaceLandmarker.FACE_LANDMARKS_LEFT_EYE;
  drawBrackets(landmarks, connectors, 0.25, label === "eyes_opened" ? OPEN_COLOR : CLOSED_COLOR);
  return { closedness, label, ear, cnnClosedness, width: screenWidth(landmarks, side === "left" ? LEFT_EYE_AXIS : RIGHT_EYE_AXIS, canvas.width, canvas.height) };
}

function drawMouth(prediction, mar, landmarks) {
  const { confidence: cnnConfidence, positiveProbability: cnnOpenness } = prediction;
  if (cnnConfidence < CONFIG.live.minCnnConfidence) return null;

  const marOpenness = linearScore(mar, CONFIG.mar.closedThreshold, CONFIG.mar.openThreshold);
  const openness = Math.max(cnnOpenness, marOpenness);
  const label = openness >= 0.5 ? "yawn" : "no yawn";

  drawBrackets(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, 0.2, label === "yawn" ? CLOSED_COLOR : MOUTH_COLOR);
  return { openness, label, mar };
}

function headPoseStatus(pitch, roll, landmarks) {
  if (pitch === null || roll === null) return null;
  const droop = headDroop(pitch, roll, headPoseCfg);
  const label = droop >= CONFIG.nod.closednessThreshold ? "nodding" : "upright";
  const color = label === "upright" ? HEAD_COLOR : CLOSED_COLOR;
  drawBrackets(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, 0.05, color, 3);
  drawHeadAxis(landmarks, color);
  return { droop, label, pitch, roll };
}

function readableRegions(lm, w, h) {
  const region = (ids, axis, margin, square) => {
    const box = boxFromLandmarks(lm, ids, w, h, margin, square);
    const crop = box && !isLowDetail(box) ? alignedCropFromLandmarks(lm, ids, axis, w, h, margin, square) : null;
    return crop ? { box, crop } : null;
  };
  return {
    left: region(LEFT_EYE_IDS, LEFT_EYE_AXIS, 0.15, true),
    right: region(RIGHT_EYE_IDS, RIGHT_EYE_AXIS, 0.15, true),
    mouth: region(MOUTH_IDS, MOUTH_AXIS, 0.3),
  };
}

function eyeClosednessOf(readings) {
  const eyes = readings.filter(Boolean);
  const total = eyes.reduce((s, r) => s + r.width, 0);
  return total ? eyes.reduce((s, r) => s + r.closedness * r.width, 0) / total : null;
}

function levelBanner(escalation, faceFound) {
  return LEVEL_BANNERS[!faceFound && escalation.level === 0 ? "no_face" : escalation.level];
}

const dashLevel = document.getElementById("dashLevel");
const DOT = { green: "#00c800", yellow: "#ffd700", red: "#dc0000", gray: "#787878" };
let dashEvents = [];
let dashLastLevel = 0;
let dashLastDraw = -Infinity;

function setDashLevel(name, color, reason, timer, darkText) {
  dashLevel.style.background = color;
  dashLevel.style.color = darkText ? "#000" : "#fff";
  document.getElementById("dashLevelName").textContent = name;
  document.getElementById("dashLevelReason").textContent = reason;
  document.getElementById("dashLevelTimer").textContent = timer;
}

function dashRow(color, label, value, extra = "", valueColor = color === DOT.gray ? "#888" : color) {
  return `<span class="dot" style="background:${color}"></span><span>${label}</span><span style="color:${valueColor}">${value}</span><span class="dim">${extra}</span>`;
}

// The graphics chip the browser renders with; laptops often default browsers to the slower built-in one.
function browserGpu() {
  const gl = document.createElement("canvas").getContext("webgl");
  const info = gl && gl.getExtension("WEBGL_debug_renderer_info");
  const renderer = info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : "";
  gl?.getExtension("WEBGL_lose_context")?.loseContext();
  const plain = renderer.replace(/\((R|TM)\)/g, "");
  const name = plain.match(/ANGLE \([^,]+, (.+?)(?: \(0x| Direct3D|,)/)?.[1] || plain || "unknown";
  const color = /swiftshader|software|basic render|llvmpipe|unknown/i.test(name) ? DOT.red : /nvidia|geforce|rtx|radeon rx/i.test(name) ? DOT.green : DOT.yellow;
  const short = name.replace(/^NVIDIA /, "").replace(/ Laptop GPU$/, "");
  return { name: short.length > 18 ? `${short.slice(0, 17)}…` : short, color };
}
const gpu = browserGpu();

// Mirrors the calibration section of LiveMonitor._dashboard in src/watch/live.py.
function showCalibration(calibrated) {
  const rows = [
    ["Status", calibrated ? "personal" : "default", calibrated ? DOT.green : DOT.yellow],
    ["Open eye", `EAR ${earCfg.openThreshold.toFixed(2)}`],
    ["Closed eye", `EAR ≤ ${earCfg.closedThreshold.toFixed(2)}`],
    ["Head down", `≥ ${headPoseCfg.droopedThreshold.toFixed(1)}°`],
    ["Head tilt", `≥ ${headPoseCfg.rollDroopedThreshold.toFixed(1)}°`],
  ];
  document.getElementById("dashCalibration").innerHTML = rows
    .map(([label, value, color]) => (color ? dashRow(color, label, value) : dashRow("transparent", label, value, "", "#eee")))
    .join("");
}

// The phone layout shows one of Detection, Calibration or Events at a time.
for (const tab of document.querySelectorAll(".dash-tabs button")) {
  tab.addEventListener("click", () => {
    document.getElementById("dashboard").dataset.tab = tab.dataset.tab;
    for (const other of document.querySelectorAll(".dash-tabs button")) other.setAttribute("aria-selected", String(other === tab));
  });
}

// Shows the dashboard only while monitoring, animating the layout change where the browser supports it.
function setDashboardVisible(visible) {
  if (dashboard.hidden === !visible) return;
  const apply = () => {
    dashboard.hidden = !visible;
  };
  if (document.startViewTransition && !matchMedia("(prefers-reduced-motion: reduce)").matches) document.startViewTransition(apply);
  else apply();
}

function resetDashboard() {
  dashEvents = [];
  dashLastLevel = 0;
  document.getElementById("dashSigns").innerHTML = "";
  document.getElementById("dashChecks").innerHTML = "";
  document.getElementById("dashCalibration").innerHTML = "";
  document.getElementById("dashLive").innerHTML = "";
  document.getElementById("dashEvents").innerHTML = '<li class="dim">none yet</li>';
}

// Mirrors LiveMonitor._dashboard in src/watch/live.py.
function updateDashboard({ escalation, faceFound, left, right, mouth, head, statuses, fps, brightness }) {
  if (escalation.level !== dashLastLevel) {
    dashLastLevel = escalation.level;
    dashEvents.unshift({ time: new Date().toLocaleTimeString([], { hour12: false }), level: escalation.level, reason: escalation.reason });
    dashEvents = dashEvents.slice(0, 50);
  }
  if (performance.now() - dashLastDraw < 150) return;
  dashLastDraw = performance.now();

  const banner = levelBanner(escalation, faceFound);
  const timer = { 1: `watching ${escalation.windowLeft.toFixed(0)}s`, 2: "alarm on" }[escalation.level] || "";
  setDashLevel(banner.text, banner.color, escalation.reason, timer, escalation.level === 1);

  const eyes = [left, right].filter(Boolean);
  const eyesClosed = eyes.length && eyes.reduce((s, r) => s + r.closedness, 0) / eyes.length >= 0.5;
  const signs = [
    ["Eyes", statuses.eyes, eyes.length ? (eyesClosed ? "closed" : "open") : null, "closed"],
    ["Yawn", statuses.mouth, mouth ? (mouth.label === "yawn" ? "yawning" : "normal") : null, "yawning"],
    ["Head", statuses.head, head ? (head.label === "nodding" ? "drooped" : "upright") : null, "drooped"],
  ];
  document.getElementById("dashSigns").innerHTML = signs
    .map(([label, status, state, concerning]) => {
      const color = !state ? DOT.gray : status.state === "not_alert" ? DOT.red : state === concerning ? DOT.yellow : DOT.green;
      return dashRow(color, label, state || "not visible", `${status.closedDuration.toFixed(1)}s`);
    })
    .join("");

  const dark = brightness < CONFIG.live.lowLightMean;
  const checks = [
    ["Face", faceFound ? "found" : "not found", faceFound ? DOT.green : DOT.red],
    ["Eyes", faceFound ? `${eyes.length} of 2 readable` : "-", faceFound ? [DOT.red, DOT.yellow, DOT.green][eyes.length] : DOT.gray],
    ["Mouth", faceFound ? (mouth ? "readable" : "not readable") : "-", faceFound ? (mouth ? DOT.green : DOT.yellow) : DOT.gray],
    ["Lighting", dark ? "low, boosted" : "good", dark ? DOT.yellow : DOT.green],
    ["FPS", fps.toFixed(0), fps >= 15 ? DOT.green : DOT.yellow],
    ["GPU", gpu.name, gpu.color],
  ];
  document.getElementById("dashChecks").innerHTML = checks.map(([label, value, color]) => dashRow(color, label, value)).join("");

  // What is being measured right now, to compare with the calibrated limits; red once past them.
  const ears = [left, right].filter(Boolean).map((r) => r.ear);
  const ear = ears.length ? ears.reduce((s, v) => s + v, 0) / ears.length : null;
  const live = [
    ["Eyes (EAR)", ear === null ? "-" : ear.toFixed(2), ear !== null && ear <= earCfg.closedThreshold],
    ["Mouth (MAR)", mouth ? mouth.mar.toFixed(2) : "-", mouth?.label === "yawn"],
    ["Head down", head ? `${head.pitch.toFixed(1)}°` : "-", head && head.pitch >= headPoseCfg.droopedThreshold],
    ["Head tilt", head ? `${head.roll.toFixed(1)}°` : "-", head && head.roll >= headPoseCfg.rollDroopedThreshold],
  ];
  document.getElementById("dashLive").innerHTML = live
    .map(([label, value, past]) => dashRow("transparent", label, value, "", past ? DOT.red : value === "-" ? "#888" : "#eee"))
    .join("");

  document.getElementById("dashEvents").innerHTML = dashEvents.length
    ? dashEvents.map((e) => `<li style="color:${LEVEL_BANNERS[e.level].color}">${e.time} ${LEVEL_BANNERS[e.level].text} ${e.reason}</li>`).join("")
    : '<li class="dim">none yet</li>';
}

function drawStatusPanel(left, right, mouth, head, targetCtx = ctx) {
  const lines = [
    left && { text: `left eye: ${left.label} ${Math.round(left.closedness * 100)}% | EAR ${left.ear.toFixed(2)}`, color: left.label === "eyes_opened" ? OPEN_COLOR : CLOSED_COLOR },
    right && { text: `right eye: ${right.label} ${Math.round(right.closedness * 100)}% | EAR ${right.ear.toFixed(2)}`, color: right.label === "eyes_opened" ? OPEN_COLOR : CLOSED_COLOR },
    mouth && { text: `mouth: ${mouth.label} ${Math.round(mouth.openness * 100)}% | MAR ${mouth.mar.toFixed(2)}`, color: mouth.label === "yawn" ? CLOSED_COLOR : MOUTH_COLOR },
    head && { text: `head: ${head.label} | pitch ${head.pitch.toFixed(1)} | roll ${head.roll.toFixed(1)} deg`, color: head.label === "upright" ? HEAD_COLOR : CLOSED_COLOR },
  ].filter(Boolean);
  if (!lines.length) return;

  const panelY = 88;
  targetCtx.fillStyle = "rgba(0, 0, 0, 0.55)";
  targetCtx.fillRect(10, panelY, 260, lines.length * 18 + 10);
  targetCtx.font = "11px sans-serif";
  lines.forEach((line, i) => {
    targetCtx.fillStyle = line.color;
    targetCtx.fillText(line.text, 18, panelY + 18 * (i + 1));
  });
}

async function calibrate(faceLandmarker, durationSec, closedOffset, droopedOffset, rollDroopedOffset) {
  const earSamples = [];
  const pitchSamples = [];
  const rollSamples = [];
  const start = performance.now();
  while (performance.now() - start < durationSec * 1000) {
    updateMirror();
    removeGlare(mirrorCanvas, glareCleanCtx);
    const result = faceLandmarker.detectForVideo(glareCleanCanvas, performance.now());
    session.add(Date.now() / 1000, "Calibrating");
    ctx.drawImage(mirrorCanvas, 0, 0, canvas.width, canvas.height);
    if (result.faceLandmarks.length) {
      const lm = result.faceLandmarks[0];
      earSamples.push(eyeAspectRatio(lm, canvas.width, canvas.height, LEFT_EYE_EAR_POINTS));
      earSamples.push(eyeAspectRatio(lm, canvas.width, canvas.height, RIGHT_EYE_EAR_POINTS));
      if (result.facialTransformationMatrixes.length) {
        const headPose = headPoseFromTransformMatrix(result.facialTransformationMatrixes[0].data);
        pitchSamples.push(headPose.pitch);
        rollSamples.push(headPose.roll);
      }
    }
    const remaining = durationSec - (performance.now() - start) / 1000;
    ctx.fillStyle = "#00c8ff";
    ctx.font = "bold 14px sans-serif";
    ctx.fillText("Calibrating -- look at the camera normally", 20, 40);
    ctx.fillText(`${remaining.toFixed(1)}s`, 20, 80);
    displayCtx.drawImage(canvas, 0, 0);
    await new Promise((r) => requestAnimationFrame(r));
  }

  if (earSamples.length < 10) return null;
  const earBaseline = median(earSamples);
  const result = { ear: { openThreshold: earBaseline, closedThreshold: earBaseline - closedOffset } };

  if (pitchSamples.length >= 10) {
    const pitchBaseline = median(pitchSamples);
    const rollBaseline = median(rollSamples);
    result.headPose = {
      uprightThreshold: pitchBaseline,
      droopedThreshold: pitchBaseline + droopedOffset,
      rollUprightThreshold: rollBaseline,
      rollDroopedThreshold: rollBaseline + rollDroopedOffset,
    };
  }
  return result;
}

async function main() {
  originalPanel.hidden = true;
  scannedLabel.hidden = true;
  statusEl.textContent = "Loading models...";
  const { eyeModel, yawnModel, faceLandmarker, vision, baseOptions } = await ensureModels();
  const zoomLandmarker = await ensureUploadFaceLandmarker(vision, baseOptions);

  statusEl.textContent = "Requesting camera...";
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: DISPLAY_WIDTH, height: DISPLAY_HEIGHT } });
  cameraStream = stream;
  cameraActive = true;
  startBtn.disabled = false;
  startBtnLabel.textContent = "Stop camera";
  cameraIcon.style.display = "none";
  liveDot.hidden = false;
  resetDashboard();
  session = new SessionLog();
  sessionLevel = 0;
  exportBtn.hidden = false;
  setDashLevel("STARTING", "#2a2f3a", "Getting the camera ready", "", false);
  setDashboardVisible(true);
  placeholder.hidden = true;
  playStartSound();
  displayCanvas.classList.add("capturable");
  captureHint.hidden = false;
  imageUpload.disabled = true;
  uploadBtn.classList.add("disabled");
  video.srcObject = stream;
  await new Promise((r) => (video.onloadedmetadata = r));
  video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  displayCanvas.width = video.videoWidth;
  displayCanvas.height = video.videoHeight;
  mirrorCanvas.width = video.videoWidth;
  mirrorCanvas.height = video.videoHeight;
  glareCleanCanvas.width = video.videoWidth;
  glareCleanCanvas.height = video.videoHeight;

  statusEl.textContent = "";
  earCfg = CONFIG.ear;
  headPoseCfg = CONFIG.headPose;
  let paused = false;
  let noFaceSince = null;
  let lastZoomTry = -Infinity;
  let glareBox = null;

  async function runCalibration() {
    paused = true;
    stopAlarm();
    setDashLevel("CALIBRATING", "#ffa500", "look at the camera normally", `${CONFIG.calibration.durationSec}s`, false);
    const calibrated = await calibrate(
      faceLandmarker, CONFIG.calibration.durationSec, CONFIG.calibration.closedOffset,
      CONFIG.calibration.droopedOffset, CONFIG.calibration.rollDroopedOffset,
    );
    if (calibrated) {
      earCfg = calibrated.ear;
      if (calibrated.headPose) headPoseCfg = calibrated.headPose;
    }
    // The camera may have been stopped, or a photo captured, while calibrating.
    if (!cameraActive) return;
    showCalibration(Boolean(calibrated));
    paused = false;
  }

  await runCalibration();
  if (!cameraActive) return;
  muteBtn.hidden = false;

  let eyeSmoother, leftEyeSmoother, rightEyeSmoother, mouthSmoother, headSmoother;
  let eyeMonitor, yawnMonitor, nodMonitor, escalationMonitor;

  // Mirrors LiveMonitor._reset in src/watch/live.py: a fresh start after every calibration.
  function resetMonitors() {
    eyeSmoother = new RollingMedian(CONFIG.live.smoothingWindow);
    leftEyeSmoother = new RollingMedian(CONFIG.live.smoothingWindow);
    rightEyeSmoother = new RollingMedian(CONFIG.live.smoothingWindow);
    mouthSmoother = new RollingMedian(CONFIG.live.smoothingWindow);
    headSmoother = new RollingMedian(CONFIG.live.smoothingWindow);

    eyeMonitor = new AlertnessMonitor({
      closedDurationThreshold: CONFIG.alertness.closedDurationThresholdSec,
      perclosWindow: CONFIG.alertness.perclosWindowSec,
      perclosThreshold: CONFIG.alertness.perclosThreshold,
      closednessThreshold: CONFIG.alertness.closednessThreshold,
    });
    yawnMonitor = new AlertnessMonitor({
      closedDurationThreshold: CONFIG.yawn.closedDurationThresholdSec,
      perclosWindow: CONFIG.yawn.perclosWindowSec,
      perclosThreshold: CONFIG.yawn.perclosThreshold,
      closednessThreshold: CONFIG.yawn.closednessThreshold,
      dipGracePeriod: CONFIG.yawn.dipGracePeriodSec,
    });
    escalationMonitor = new EscalationMonitor(CONFIG.escalation);
    nodMonitor = new AlertnessMonitor({
      closedDurationThreshold: CONFIG.nod.closedDurationThresholdSec,
      perclosWindow: CONFIG.nod.perclosWindowSec,
      perclosThreshold: CONFIG.nod.perclosThreshold,
      closednessThreshold: CONFIG.nod.closednessThreshold,
      dipGracePeriod: CONFIG.nod.dipGracePeriodSec,
    });
  }
  resetMonitors();

  let prevTime = performance.now();

  async function frame() {
    if (!cameraActive) return;
    try {
      if (paused) {
        requestAnimationFrame(frame);
        return;
      }
      updateMirror();
      // Only glare on the face matters, so skip bright spots elsewhere such as windows.
      const brightness = removeGlare(mirrorCanvas, glareCleanCtx, true, glareBox);
      let result = faceLandmarker.detectForVideo(glareCleanCanvas, performance.now());
      // Retry faces the tracker misses, such as one far from the camera.
      if (!result.faceLandmarks.length && performance.now() - lastZoomTry > 500) {
        lastZoomTry = performance.now();
        result = (await detectZoomed(zoomLandmarker, glareCleanCanvas)) || result;
      }
      const faceFound = result.faceLandmarks.length > 0;
      glareBox = faceFound ? faceBox(result.faceLandmarks[0], canvas.width, canvas.height) : null;

      if (!faceFound) {
        if (noFaceSince === null) noFaceSince = performance.now();
      } else if (noFaceSince !== null) {
        // The user left the camera, so calibrate again for whoever is in front of it now.
        const gap = performance.now() - noFaceSince;
        noFaceSince = null;
        if (gap >= CONFIG.live.recalibrateGapSec * 1000) {
          await runCalibration();
          resetMonitors();
          dashLastLevel = 0;
          dashEvents.unshift({ time: new Date().toLocaleTimeString([], { hour12: false }), level: 0, reason: "recalibrated" });
          session.event("Recalibrated");
          sessionLevel = 0;
          requestAnimationFrame(frame);
          return;
        }
      }

      ctx.drawImage(mirrorCanvas, 0, 0, canvas.width, canvas.height);

      let eyeClosedness = null, mouthOpenness = null, headDroop = null;
      let leftResult = null, rightResult = null, mouthResult = null, headResult = null;
      if (result.faceLandmarks.length) {
        const lm = result.faceLandmarks[0];
        const w = canvas.width, h = canvas.height;

        const regions = readableRegions(lm, w, h);
        const predictions = await predictRegions(regions, eyeModel, yawnModel);
        leftResult = regions.left && drawEye("left", regions.left.box, predictions.left, eyeAspectRatio(lm, w, h, LEFT_EYE_EAR_POINTS), earCfg, lm, leftEyeSmoother);
        rightResult = regions.right && drawEye("right", regions.right.box, predictions.right, eyeAspectRatio(lm, w, h, RIGHT_EYE_EAR_POINTS), earCfg, lm, rightEyeSmoother);
        mouthResult = regions.mouth && drawMouth(predictions.mouth, mouthAspectRatio(lm, w, h), lm);
        eyeClosedness = eyeClosednessOf([leftResult, rightResult]);
        mouthOpenness = mouthResult ? mouthResult.openness : null;
        if (result.facialTransformationMatrixes.length) {
          const headPose = headPoseFromTransformMatrix(result.facialTransformationMatrixes[0].data);
          headResult = headPoseStatus(headPose.pitch, headPose.roll, lm);
          headDroop = headResult?.droop ?? null;
        }
      }
      // A frame still in flight when the camera stops must not overwrite the idle dashboard or start the alarm.
      if (!cameraActive) return;
      const now = performance.now() / 1000;
      const eyeStatus = eyeMonitor.update(eyeSmoother.add(eyeClosedness), now);
      const yawnStatus = yawnMonitor.update(mouthSmoother.add(mouthOpenness), now);
      const nodStatus = nodMonitor.update(headSmoother.add(headDroop), now);

      const escalation = escalationMonitor.update({ eyes: eyeStatus, mouth: yawnStatus, head: nodStatus }, faceFound, now);
      if (escalation.level !== sessionLevel) {
        sessionLevel = escalation.level;
        session.event(escalation.reason ? `${REPORT_LEVEL_NAMES[escalation.level]}: ${escalation.reason}` : REPORT_LEVEL_NAMES[escalation.level]);
      }
      const ears = [leftResult, rightResult].filter(Boolean).map((r) => r.ear);
      session.add(
        Date.now() / 1000,
        !faceFound && escalation.level === 0 ? "No face" : REPORT_LEVEL_NAMES[escalation.level],
        { ear: ears.length ? ears.reduce((a, b) => a + b, 0) / ears.length : null, mar: mouthResult?.mar ?? null, headDown: headResult?.pitch ?? null, headTilt: headResult?.roll ?? null },
        { earClosed: earCfg.closedThreshold, headDown: headPoseCfg.droopedThreshold, headTilt: headPoseCfg.rollDroopedThreshold },
      );
      if (escalation.level === 2) startAlarm();
      else stopAlarm();

      const fps = now > prevTime ? 1.0 / (now - prevTime) : 0.0;
      prevTime = now;
      updateDashboard({
        escalation, faceFound, left: leftResult, right: rightResult, mouth: mouthResult, head: headResult,
        statuses: { eyes: eyeStatus, mouth: yawnStatus, head: nodStatus }, fps, brightness,
      });

      if (!cameraActive) return;
      displayCtx.drawImage(canvas, 0, 0);
      requestAnimationFrame(frame);
    } catch (err) {
      console.error("frame() error, recovering:", err);
      requestAnimationFrame(frame);
    }
  }

  requestAnimationFrame(frame);
}

function drawSnapshotBanner(state) {
  displayCtx.fillStyle = BANNER_COLORS[state];
  displayCtx.fillRect(0, 0, displayCanvas.width, 50);
  displayCtx.fillStyle = "#ffffff";
  displayCtx.font = "bold 18px sans-serif";
  displayCtx.fillText(BANNER_TEXT[state], 20, 34);
}

const zoomCanvas = document.createElement("canvas");
const zoomCtx = zoomCanvas.getContext("2d");

// Mirrors FaceTracker._landmarks in src/watch/face.py.
async function detectZoomed(landmarker, source) {
  const w = source.width, h = source.height;
  const blazefaceModel = await ensureBlazeface();
  const faces = await blazefaceModel.estimateFaces(source, false);
  if (!faces.length) return null;

  const [x1, y1] = faces[0].topLeft;
  const [x2, y2] = faces[0].bottomRight;
  const boxW = x2 - x1, boxH = y2 - y1;
  const pad = 0.6;
  const cropX = Math.max(0, x1 - boxW * pad);
  const cropY = Math.max(0, y1 - boxH * pad);
  const cropW = Math.min(w, x2 + boxW * pad) - cropX;
  const cropH = Math.min(h, y2 + boxH * pad) - cropY;
  if (cropW <= 0 || cropH <= 0) return null;

  zoomCanvas.width = 256;
  zoomCanvas.height = Math.round((256 * cropH) / cropW);
  zoomCtx.drawImage(source, cropX, cropY, cropW, cropH, 0, 0, zoomCanvas.width, zoomCanvas.height);

  const zoomed = landmarker.detect(zoomCanvas);
  if (!zoomed.faceLandmarks.length) return null;
  const toFrame = (p) => ({ x: (cropX + p.x * cropW) / w, y: (cropY + p.y * cropH) / h, z: (p.z * cropW) / w });
  return { ...zoomed, faceLandmarks: zoomed.faceLandmarks.map((face) => face.map(toFrame)) };
}

function blitUnmirrored() {
  displayCtx.save();
  displayCtx.scale(-1, 1);
  displayCtx.drawImage(canvas, -canvas.width, 0, canvas.width, canvas.height);
  displayCtx.restore();
}

async function handleImageUpload(file, alreadyMirrored = false) {
  const finalBlit = () => {
    if (alreadyMirrored) displayCtx.drawImage(canvas, 0, 0);
    else blitUnmirrored();
  };

  statusEl.textContent = "Loading models...";
  const { eyeModel, yawnModel, vision, baseOptions } = await ensureModels();
  const uploadFaceLandmarker = await ensureUploadFaceLandmarker(vision, baseOptions);

  const img = new Image();
  const url = URL.createObjectURL(file);
  try {
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = () => reject(new Error("Could not load that image file."));
      img.src = url;
    });
  } finally {
    URL.revokeObjectURL(url);
  }

  const w = DISPLAY_WIDTH, h = DISPLAY_HEIGHT;
  canvas.width = displayCanvas.width = mirrorCanvas.width = w;
  canvas.height = displayCanvas.height = mirrorCanvas.height = h;

  const fitScale = Math.min(w / img.naturalWidth, h / img.naturalHeight);
  const drawW = img.naturalWidth * fitScale;
  const drawH = img.naturalHeight * fitScale;
  const offsetX = (w - drawW) / 2;
  const offsetY = (h - drawH) / 2;

  originalCanvas.width = w;
  originalCanvas.height = h;
  originalCtx.fillStyle = "#000";
  originalCtx.fillRect(0, 0, w, h);
  originalCtx.drawImage(img, offsetX, offsetY, drawW, drawH);
  originalPanel.hidden = false;
  scannedLabel.hidden = false;
  placeholder.hidden = true;
  // A photo is not monitored over time, so the dashboard has nothing to show.
  setDashboardVisible(false);

  mirrorCtx.fillStyle = "#000";
  mirrorCtx.fillRect(0, 0, w, h);
  if (alreadyMirrored) {
    mirrorCtx.drawImage(img, offsetX, offsetY, drawW, drawH);
  } else {
    mirrorCtx.save();
    mirrorCtx.scale(-1, 1);
    mirrorCtx.drawImage(img, -(offsetX + drawW), offsetY, drawW, drawH);
    mirrorCtx.restore();
  }
  ctx.drawImage(mirrorCanvas, 0, 0);
  // Uploaded photos skip glare removal, matching predict.py.
  removeGlare(mirrorCanvas, glareCleanCtx, false);

  statusEl.textContent = "";
  let result = uploadFaceLandmarker.detect(glareCleanCanvas);
  if (!result.faceLandmarks.length) {
    statusEl.textContent = "No face found directly -- retrying with a coarser detector...";
    result = (await detectZoomed(uploadFaceLandmarker, glareCleanCanvas)) || result;
    statusEl.textContent = "";
  }

  if (!result.faceLandmarks.length) {
    finalBlit();
    drawSnapshotBanner("no_face");
    return;
  }

  const lm = result.faceLandmarks[0];
  const regions = readableRegions(lm, w, h);
  const predictions = await predictRegions(regions, eyeModel, yawnModel);
  const left = regions.left ? drawEye("left", regions.left.box, predictions.left, eyeAspectRatio(lm, w, h, LEFT_EYE_EAR_POINTS), earCfg, lm) : null;
  const right = regions.right ? drawEye("right", regions.right.box, predictions.right, eyeAspectRatio(lm, w, h, RIGHT_EYE_EAR_POINTS), earCfg, lm) : null;
  const mouth = regions.mouth ? drawMouth(predictions.mouth, mouthAspectRatio(lm, w, h), lm) : null;
  let head = null;
  if (result.facialTransformationMatrixes.length) {
    const headPose = headPoseFromTransformMatrix(result.facialTransformationMatrixes[0].data);
    head = headPoseStatus(headPose.pitch, headPose.roll, lm);
  }
  const eyeClosedness = eyeClosednessOf([left, right]);
  const notAlert =
    (eyeClosedness !== null && eyeClosedness >= CONFIG.alertness.closednessThreshold) ||
    (mouth !== null && mouth.openness >= CONFIG.yawn.closednessThreshold) ||
    (head !== null && head.droop >= CONFIG.nod.closednessThreshold);

  finalBlit();
  drawSnapshotBanner(notAlert ? "not_alert" : "alert");
  drawStatusPanel(left, right, mouth, head, displayCtx);
}

function runImageUpload(file, alreadyMirrored = false) {
  if (cameraActive) return;
  handleImageUpload(file, alreadyMirrored).catch((err) => {
    statusEl.textContent = `Error: ${err.message}`;
    console.error(err);
  });
}

imageUpload.addEventListener("change", () => {
  const file = imageUpload.files[0];
  if (file) runImageUpload(file);
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

displayCanvas.addEventListener("click", () => {
  if (!cameraActive) return;
  flashCapture();
  playShutterSound();
  const captureCanvas = document.createElement("canvas");
  captureCanvas.width = mirrorCanvas.width;
  captureCanvas.height = mirrorCanvas.height;
  captureCanvas.getContext("2d").drawImage(mirrorCanvas, 0, 0);
  captureCanvas.toBlob((blob) => {
    if (!blob) return;
    const file = new File([blob], "capture.png", { type: "image/png" });
    stopCamera();
    runImageUpload(file, true);
  }, "image/png");
});

function stopCamera() {
  cameraActive = false;
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  video.srcObject = null;
  displayCtx.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
  displayCanvas.classList.remove("capturable");
  captureHint.hidden = true;
  stopAlarm();
  playStopSound();
  muteBtn.hidden = true;
  cameraIcon.style.display = "";
  liveDot.hidden = true;
  resetDashboard();
  setDashboardVisible(false);
  placeholder.hidden = false;
  startBtnLabel.textContent = "Use camera";
  imageUpload.disabled = false;
  uploadBtn.classList.remove("disabled");
  statusEl.textContent = "";
}

startBtn.addEventListener("click", () => {
  if (cameraActive) {
    stopCamera();
    return;
  }
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  startBtn.disabled = true;
  main().catch((err) => {
    statusEl.textContent = `Error: ${err.message}`;
    console.error(err);
    startBtn.disabled = false;
  });
});

function download(blob, filename) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

// Mirrors LiveMonitor.export in src/watch/live.py: a PNG report with graphs and a CSV of the readings.
exportBtn.addEventListener("click", () => {
  if (!session.samples.length) {
    statusEl.textContent = "Nothing recorded yet. Start the camera first.";
    return;
  }
  const pad = (n) => String(n).padStart(2, "0");
  const d = new Date();
  const stem = `watch-session-${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
  const reportCanvas = document.createElement("canvas");
  session.drawReport(reportCanvas);
  reportCanvas.toBlob((blob) => blob && download(blob, `${stem}.png`), "image/png");
  download(new Blob([session.csv()], { type: "text/csv" }), `${stem}.csv`);
});

muteBtn.addEventListener("click", () => {
  muted = !muted;
  muteBtn.textContent = muted ? "Unmute Alarm" : "Mute Alarm";
  if (muted) stopAlarm();
});
