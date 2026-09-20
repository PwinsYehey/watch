// Ports src/frames/inference/live_monitor.py to the browser using MediaPipe Tasks Vision + TensorFlow.js.
import { FaceLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// a real gap this long (not just a blink or a momentary tracking miss) means the driver likely changed
const NO_FACE_RECALIBRATE_GAP_MS = 3000;

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

// tracking/CNN-input-only copy of the frame with glare smoothed out -- the displayed video (mirrorCanvas/canvas) stays untouched
const glareCleanCanvas = document.createElement("canvas");
const glareCleanCtx = glareCleanCanvas.getContext("2d", { willReadFrequently: true });
const glareBlurCanvas = document.createElement("canvas");
const glareBlurCtx = glareBlurCanvas.getContext("2d");

// mirrors _remove_glare in live_monitor.py -- smooths out bright, washed-out specular highlights (e.g. visor glare)
// before landmark tracking; no OpenCV inpaint available in-browser, so a blurred copy stands in for the reconstructed pixels
function removeGlare(sourceCanvas, targetCtx) {
  const w = sourceCanvas.width, h = sourceCanvas.height;
  targetCtx.canvas.width = w;
  targetCtx.canvas.height = h;
  targetCtx.drawImage(sourceCanvas, 0, 0);

  glareBlurCanvas.width = w;
  glareBlurCanvas.height = h;
  glareBlurCtx.filter = "blur(6px)";
  glareBlurCtx.drawImage(sourceCanvas, 0, 0);
  glareBlurCtx.filter = "none";

  const sharp = targetCtx.getImageData(0, 0, w, h);
  const blurred = glareBlurCtx.getImageData(0, 0, w, h);
  const data = sharp.data;
  const blurData = blurred.data;

  for (let i = 0; i < data.length; i += 4) {
    if (isGlarePixel(data[i], data[i + 1], data[i + 2])) {
      data[i] = blurData[i];
      data[i + 1] = blurData[i + 1];
      data[i + 2] = blurData[i + 2];
    }
  }

  targetCtx.putImageData(sharp, 0, 0);
}

// glare is bright and washed-out (low saturation), unlike genuine skin/eye color
function isGlarePixel(r, g, b) {
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  return max > 235 && max - min < 60;
}

const EAR_GLARE_TOLERANCE = 0.15;

// mirrors _glare_fraction in live_monitor.py -- fraction of a box on the ORIGINAL uncleaned frame that was glare
function glareFraction(box) {
  const x = Math.max(0, Math.floor(box.xMin));
  const y = Math.max(0, Math.floor(box.yMin));
  const w = Math.max(0, Math.ceil(box.xMax) - x);
  const h = Math.max(0, Math.ceil(box.yMax) - y);
  if (w <= 0 || h <= 0) return 0.0;
  const { data } = mirrorCtx.getImageData(x, y, w, h);
  let glareCount = 0;
  for (let i = 0; i < data.length; i += 4) {
    if (isGlarePixel(data[i], data[i + 1], data[i + 2])) glareCount++;
  }
  return glareCount / (data.length / 4);
}

const LOW_DETAIL_STD_THRESHOLD = 12.0;

// mirrors _is_low_detail in facemesh.py -- true if a box is too flat/featureless to be real facial
// detail (e.g. a solid patch of opaque tinted visor), checked on the crop the model would actually see
function isLowDetail(box) {
  const x = Math.max(0, Math.floor(box.xMin));
  const y = Math.max(0, Math.floor(box.yMin));
  const w = Math.max(0, Math.ceil(box.xMax) - x);
  const h = Math.max(0, Math.ceil(box.yMax) - y);
  if (w <= 0 || h <= 0) return true;
  const { data } = glareCleanCtx.getImageData(x, y, w, h);
  const n = data.length / 4;
  const grays = new Float64Array(n);
  let sum = 0;
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    // matches cv2.cvtColor's BGR2GRAY luma weights
    const gray = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
    grays[j] = gray;
    sum += gray;
  }
  const mean = sum / n;
  let variance = 0;
  for (let j = 0; j < n; j++) variance += (grays[j] - mean) ** 2;
  return Math.sqrt(variance / n) < LOW_DETAIL_STD_THRESHOLD;
}

const statusEl = document.getElementById("status");
const startBtn = document.getElementById("startBtn");
const startBtnLabel = document.getElementById("startBtnLabel");
const cameraIcon = document.getElementById("cameraIcon");
const liveDot = document.getElementById("liveDot");
const uploadBtn = document.getElementById("uploadBtn");
const muteBtn = document.getElementById("muteBtn");
const imageUpload = document.getElementById("imageUpload");

let cameraActive = false;
let cameraStream = null;

const OPEN_COLOR = "#00c800";
const CLOSED_COLOR = "#dc0000";
const MOUTH_COLOR = "#ffc800";
const HEAD_COLOR = "#ff00c8";
const BANNER_COLORS = { alert: "#00c800", not_alert: "#dc0000", no_face: "#ffa500" };
const BANNER_TEXT = { alert: "ALERT", not_alert: "NOT ALERT", no_face: "NO FACE DETECTED" };

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

// a loud pulsing siren tone, started on NOT ALERT and stopped as soon as it clears
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

// tiny offscreen canvas reused for every crop -> model-input resize, avoids per-frame allocation
const cropCanvas = document.createElement("canvas");
cropCanvas.width = CONFIG.imageSize[0];
cropCanvas.height = CONFIG.imageSize[1];
const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });

// mirrors _predict_class_state in predict.py
async function predictClassState(model, sourceCanvas, box, classNames, positiveClass, desaturate = false) {
  cropCtx.drawImage(sourceCanvas, box.xMin, box.yMin, box.xMax - box.xMin, box.yMax - box.yMin, 0, 0, cropCanvas.width, cropCanvas.height);
  // .data() is async and non-blocking; dataSync() stalls the main thread waiting on the GPU every call
  const predictionTensor = tf.tidy(() => {
    let input = tf.browser.fromPixels(cropCanvas).toFloat();
    if (desaturate) {
      // matches cv2.cvtColor(..., COLOR_BGR2GRAY)'s luma weights, replicated back to 3 channels
      const [r, g, b] = tf.split(input, 3, 2);
      const gray = r.mul(0.299).add(g.mul(0.587)).add(b.mul(0.114));
      input = tf.concat([gray, gray, gray], 2);
    }
    return model.predict(input.div(255.0).expandDims(0));
  });
  const prediction = await predictionTensor.data();
  predictionTensor.dispose();
  const predictedIndex = prediction.indexOf(Math.max(...prediction));
  return {
    label: classNames[predictedIndex],
    confidence: prediction[predictedIndex],
    positiveProbability: prediction[classNames.indexOf(positiveClass)],
  };
}

let drawingUtils = null;
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
      });

      drawingUtils = new DrawingUtils(ctx);
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

const MIN_CNN_CONFIDENCE = 0.65;

async function drawEye(side, box, ear, model, earCfg, landmarks, smoother) {
  const { confidence: cnnConfidence, positiveProbability: cnnClosedness } = await predictClassState(model, glareCleanCanvas, box, CONFIG.eyeClassNames, "eyes_closed", true);
  if (cnnConfidence < MIN_CNN_CONFIDENCE) return null;

  // a heavily glared eye region corrupts EAR's precise landmark geometry more than the CNN's holistic read,
  // so a strong glare patch on one side of the eye no longer forces a false "eyes_closed" through the EAR side of the max() below
  const earClosedness = ear <= earCfg.closedThreshold && glareFraction(box) < EAR_GLARE_TOLERANCE ? 1.0 : 0.0;
  // no gate -- now that the crop is square (matching the CNN's training data framing), trust it fully
  const rawClosedness = Math.max(earClosedness, cnnClosedness);
  // smoothed per-eye so a single glare-jittered frame under a reflective visor can't flip the on-screen label
  const closedness = smoother ? smoother.add(rawClosedness) : rawClosedness;
  const label = closedness >= 0.5 ? "eyes_closed" : "eyes_opened";

  const connectors = side === "left" ? FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE : FaceLandmarker.FACE_LANDMARKS_LEFT_EYE;
  drawingUtils.drawConnectors(landmarks, connectors, { color: label === "eyes_opened" ? OPEN_COLOR : CLOSED_COLOR, lineWidth: 2 });
  return { closedness, label, ear, cnnClosedness };
}

async function drawMouth(box, mar, model, landmarks) {
  const { confidence: cnnConfidence, positiveProbability: cnnOpenness } = await predictClassState(model, glareCleanCanvas, box, CONFIG.yawnClassNames, "yawn");
  // the yawn CNN has never seen a hand covering the mouth, so a low-confidence prediction is a sign
  // this isn't really a mouth at all -- e.g. MediaPipe still guessed a plausible-looking box over a hand
  if (cnnConfidence < MIN_CNN_CONFIDENCE) return null;

  const marOpenness = linearScore(mar, CONFIG.mar.closedThreshold, CONFIG.mar.openThreshold);
  // either signal suggesting a yawn is enough -- a safety monitor should not need both to agree
  const openness = Math.max(cnnOpenness, marOpenness);
  const label = openness >= 0.5 ? "yawn" : "no yawn";

  drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: label === "yawn" ? CLOSED_COLOR : MOUTH_COLOR, lineWidth: 2 });
  return { openness, label, mar };
}

function headPoseStatus(pitch, roll, landmarks) {
  if (pitch === null || roll === null) return null;
  const pitchDroop = linearScore(pitch, headPoseCfg.uprightThreshold, headPoseCfg.droopedThreshold);
  const rollDroop = linearScore(roll, headPoseCfg.rollUprightThreshold, headPoseCfg.rollDroopedThreshold);
  // either a forward nod or a sideways tilt toward the shoulder counts as drooping
  const droop = Math.max(pitchDroop, rollDroop);
  const label = droop >= 0.5 ? "nodding" : "upright";
  drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: label === "upright" ? HEAD_COLOR : CLOSED_COLOR, lineWidth: 2 });
  return { droop, label, pitch, roll };
}

function drawBanner(state, eyeStatus, yawnStatus, nodStatus) {
  ctx.fillStyle = BANNER_COLORS[state];
  ctx.fillRect(0, 0, canvas.width, 50);
  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 18px sans-serif";
  ctx.fillText(BANNER_TEXT[state], 20, 34);
  ctx.font = "10px sans-serif";
  ctx.fillText(
    `eyes closed ${eyeStatus.closedDuration.toFixed(1)}s | mouth open ${yawnStatus.closedDuration.toFixed(1)}s | head drooped ${nodStatus.closedDuration.toFixed(1)}s`,
    20,
    70
  );
}

// a single fixed panel for all per-cue readouts, so labels never overlap regardless of face size/position
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
  const { eyeModel, yawnModel, faceLandmarker } = await ensureModels();

  statusEl.textContent = "Requesting camera...";
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: DISPLAY_WIDTH, height: DISPLAY_HEIGHT } });
  cameraStream = stream;
  cameraActive = true;
  startBtn.disabled = false;
  startBtnLabel.textContent = "Stop camera";
  cameraIcon.style.display = "none";
  liveDot.hidden = false;
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

  async function runCalibration() {
    // pause the main render loop so it doesn't fight the calibration overlay for the same canvas
    paused = true;
    stopAlarm();
    const calibrated = await calibrate(
      faceLandmarker, CONFIG.calibration.durationSec, CONFIG.calibration.closedOffset,
      CONFIG.calibration.droopedOffset, CONFIG.calibration.rollDroopedOffset,
    );
    if (calibrated) {
      earCfg = calibrated.ear;
      statusEl.textContent = `Calibrated EAR thresholds: open>=${earCfg.openThreshold.toFixed(2)}, closed<=${earCfg.closedThreshold.toFixed(2)}`;
      if (calibrated.headPose) {
        headPoseCfg = calibrated.headPose;
        statusEl.textContent += ` | head-pose: upright<=${headPoseCfg.uprightThreshold.toFixed(1)}, drooped>=${headPoseCfg.droopedThreshold.toFixed(1)}`;
      }
    } else {
      statusEl.textContent = "Calibration failed (no face detected) -- using default EAR/head-pose thresholds.";
    }
    paused = false;
  }

  await runCalibration();
  muteBtn.style.display = "inline-block";

  const eyeSmoother = new RollingMedian(5);
  const leftEyeSmoother = new RollingMedian(5);
  const rightEyeSmoother = new RollingMedian(5);
  const mouthSmoother = new RollingMedian(5);
  const headSmoother = new RollingMedian(5);

  const eyeMonitor = new AlertnessMonitor({
    closedDurationThreshold: CONFIG.alertness.closedDurationThresholdSec,
    perclosWindow: CONFIG.alertness.perclosWindowSec,
    perclosThreshold: CONFIG.alertness.perclosThreshold,
    closednessThreshold: CONFIG.alertness.closednessThreshold,
  });
  const yawnMonitor = new AlertnessMonitor({
    closedDurationThreshold: CONFIG.yawn.closedDurationThresholdSec,
    perclosWindow: CONFIG.yawn.perclosWindowSec,
    perclosThreshold: CONFIG.yawn.perclosThreshold,
    closednessThreshold: CONFIG.yawn.closednessThreshold,
    dipGracePeriod: CONFIG.yawn.dipGracePeriodSec,
  });
  const nodMonitor = new AlertnessMonitor({
    closedDurationThreshold: CONFIG.nod.closedDurationThresholdSec,
    perclosWindow: CONFIG.nod.perclosWindowSec,
    perclosThreshold: CONFIG.nod.perclosThreshold,
    closednessThreshold: CONFIG.nod.closednessThreshold,
    dipGracePeriod: CONFIG.nod.dipGracePeriodSec,
  });

  drawingUtils = new DrawingUtils(ctx);
  let prevTime = performance.now();

  async function frame() {
    if (!cameraActive) return;
    try {
      if (paused) {
        requestAnimationFrame(frame);
        return;
      }
      updateMirror();
      removeGlare(mirrorCanvas, glareCleanCtx);
      const result = faceLandmarker.detectForVideo(glareCleanCanvas, performance.now());
      const faceFound = result.faceLandmarks.length > 0;

      if (!faceFound) {
        if (noFaceSince === null) noFaceSince = performance.now();
      } else if (noFaceSince !== null) {
        // a face reappearing after a real gap likely means a new driver, so recalibrate for them automatically
        const gap = performance.now() - noFaceSince;
        noFaceSince = null;
        if (gap >= NO_FACE_RECALIBRATE_GAP_MS) {
          await runCalibration();
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

        const leftEar = eyeAspectRatio(lm, w, h, LEFT_EYE_EAR_POINTS);
        const rightEar = eyeAspectRatio(lm, w, h, RIGHT_EYE_EAR_POINTS);
        const leftBoxRaw = boxFromLandmarks(lm, LEFT_EYE_IDS, w, h, 0.15, true);
        const rightBoxRaw = boxFromLandmarks(lm, RIGHT_EYE_IDS, w, h, 0.15, true);
        const mouthBoxRaw = boxFromLandmarks(lm, MOUTH_IDS, w, h, 0.3);
        // drop a region individually (not just when the crop geometry fails, but also when the crop is too
        // flat/featureless to be real detail -- e.g. an opaque tinted visor) rather than discarding the whole frame
        const leftBox = leftBoxRaw && !isLowDetail(leftBoxRaw) ? leftBoxRaw : null;
        const rightBox = rightBoxRaw && !isLowDetail(rightBoxRaw) ? rightBoxRaw : null;
        const mouthBox = mouthBoxRaw && !isLowDetail(mouthBoxRaw) ? mouthBoxRaw : null;

        if (leftBox || rightBox) {
          // run all CNN calls concurrently instead of awaiting each GPU round-trip in turn
          const calls = [];
          if (leftBox) calls.push(drawEye("left", leftBox, leftEar, eyeModel, earCfg, lm, leftEyeSmoother));
          if (rightBox) calls.push(drawEye("right", rightBox, rightEar, eyeModel, earCfg, lm, rightEyeSmoother));
          if (mouthBox) calls.push(drawMouth(mouthBox, mouthAspectRatio(lm, w, h), yawnModel, lm));
          const results = await Promise.all(calls);
          let idx = 0;
          if (leftBox) leftResult = results[idx++];
          if (rightBox) rightResult = results[idx++];
          if (mouthBox) mouthResult = results[idx++];

          const eyeReadings = [leftResult, rightResult].filter(Boolean);
          // average both eyes so one drooping eyelid still raises the score; if neither eye is readable
          // (e.g. an opaque tinted visor), report no reading rather than fabricating one
          eyeClosedness = eyeReadings.length ? eyeReadings.reduce((s, r) => s + r.closedness, 0) / eyeReadings.length : null;
          mouthOpenness = mouthResult ? mouthResult.openness : null;
          if (result.facialTransformationMatrixes.length) {
            const headPose = headPoseFromTransformMatrix(result.facialTransformationMatrixes[0].data);
            headResult = headPoseStatus(headPose.pitch, headPose.roll, lm);
            headDroop = headResult?.droop ?? null;
          }
        }
      }
      drawStatusPanel(leftResult, rightResult, mouthResult, headResult);

      const now = performance.now() / 1000;
      const eyeStatus = eyeMonitor.update(eyeSmoother.add(eyeClosedness), now);
      const yawnStatus = yawnMonitor.update(mouthSmoother.add(mouthOpenness), now);
      const nodStatus = nodMonitor.update(headSmoother.add(headDroop), now);

      let overallState;
      if (eyeStatus.state === "no_face") overallState = "no_face";
      else if ([eyeStatus.state, yawnStatus.state, nodStatus.state].includes("not_alert")) overallState = "not_alert";
      else overallState = "alert";

      if (overallState === "not_alert") startAlarm();
      else stopAlarm();

      drawBanner(overallState, eyeStatus, yawnStatus, nodStatus);

      const fps = now > prevTime ? 1.0 / (now - prevTime) : 0.0;
      prevTime = now;
      ctx.fillStyle = "#00ffff";
      ctx.font = "bold 12px sans-serif";
      ctx.fillText(`FPS: ${fps.toFixed(1)}`, 20, canvas.height - 20);

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

async function tryCropAndRetry(uploadFaceLandmarker, w, h) {
  const blazefaceModel = await ensureBlazeface();
  const faces = await blazefaceModel.estimateFaces(mirrorCanvas, false);
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

  const cropped = document.createElement("canvas");
  cropped.width = w;
  cropped.height = h;
  cropped.getContext("2d").drawImage(mirrorCanvas, cropX, cropY, cropW, cropH, 0, 0, w, h);

  const retryResult = uploadFaceLandmarker.detect(cropped);
  if (!retryResult.faceLandmarks.length) return null;

  mirrorCtx.clearRect(0, 0, w, h);
  mirrorCtx.drawImage(cropped, 0, 0);
  return retryResult;
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
  // no glare removal for a single uploaded photo (matches predict.py's scope) -- just keep this canvas valid for drawEye/drawMouth's crop source
  glareCleanCanvas.width = mirrorCanvas.width;
  glareCleanCanvas.height = mirrorCanvas.height;
  glareCleanCtx.drawImage(mirrorCanvas, 0, 0);

  statusEl.textContent = "";
  let result = uploadFaceLandmarker.detect(mirrorCanvas);
  if (!result.faceLandmarks.length) {
    statusEl.textContent = "No face found directly -- retrying with a coarser detector...";
    const retried = await tryCropAndRetry(uploadFaceLandmarker, w, h);
    statusEl.textContent = "";
    if (retried) {
      result = retried;
      ctx.drawImage(mirrorCanvas, 0, 0);
      glareCleanCtx.clearRect(0, 0, glareCleanCanvas.width, glareCleanCanvas.height);
      glareCleanCtx.drawImage(mirrorCanvas, 0, 0);
    }
  }

  if (!result.faceLandmarks.length) {
    finalBlit();
    drawSnapshotBanner("no_face");
    return;
  }

  const lm = result.faceLandmarks[0];
  const leftEar = eyeAspectRatio(lm, w, h, LEFT_EYE_EAR_POINTS);
  const rightEar = eyeAspectRatio(lm, w, h, RIGHT_EYE_EAR_POINTS);
  const leftBoxRaw = boxFromLandmarks(lm, LEFT_EYE_IDS, w, h, 0.15, true);
  const rightBoxRaw = boxFromLandmarks(lm, RIGHT_EYE_IDS, w, h, 0.15, true);
  const mouthBoxRaw = boxFromLandmarks(lm, MOUTH_IDS, w, h, 0.3);
  // drop a region individually (not just when the crop geometry fails, but also when the crop is too
  // flat/featureless to be real detail -- e.g. an opaque tinted visor) rather than discarding the whole frame
  const leftBox = leftBoxRaw && !isLowDetail(leftBoxRaw) ? leftBoxRaw : null;
  const rightBox = rightBoxRaw && !isLowDetail(rightBoxRaw) ? rightBoxRaw : null;
  const mouthBox = mouthBoxRaw && !isLowDetail(mouthBoxRaw) ? mouthBoxRaw : null;

  if (!leftBox && !rightBox) {
    finalBlit();
    drawSnapshotBanner("no_face");
    return;
  }

  const left = leftBox ? await drawEye("left", leftBox, leftEar, eyeModel, earCfg, lm) : null;
  const right = rightBox ? await drawEye("right", rightBox, rightEar, eyeModel, earCfg, lm) : null;
  const mouth = mouthBox ? await drawMouth(mouthBox, mouthAspectRatio(lm, w, h), yawnModel, lm) : null;
  let head = null;
  if (result.facialTransformationMatrixes.length) {
    const headPose = headPoseFromTransformMatrix(result.facialTransformationMatrixes[0].data);
    head = headPoseStatus(headPose.pitch, headPose.roll, lm);
  }
  const eyeReadings = [left, right].filter(Boolean);
  const eyeClosedness = eyeReadings.length ? eyeReadings.reduce((s, r) => s + r.closedness, 0) / eyeReadings.length : null;
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
  muteBtn.style.display = "none";
  cameraIcon.style.display = "";
  liveDot.hidden = true;
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

muteBtn.addEventListener("click", () => {
  muted = !muted;
  muteBtn.textContent = muted ? "Unmute Alarm" : "Mute Alarm";
  if (muted) stopAlarm();
});
