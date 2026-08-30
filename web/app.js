// Ports src/frames/inference/live_monitor.py to the browser using MediaPipe Tasks Vision + TensorFlow.js.
import { FaceLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// a real gap this long (not just a blink or a momentary tracking miss) means the driver likely changed
const NO_FACE_RECALIBRATE_GAP_MS = 3000;

const video = document.getElementById("video");
const canvas = document.getElementById("output");
const ctx = canvas.getContext("2d");

const mirrorCanvas = document.createElement("canvas");
const mirrorCtx = mirrorCanvas.getContext("2d");

function updateMirror() {
  mirrorCtx.save();
  mirrorCtx.scale(-1, 1);
  mirrorCtx.drawImage(video, -mirrorCanvas.width, 0, mirrorCanvas.width, mirrorCanvas.height);
  mirrorCtx.restore();
}
const statusEl = document.getElementById("status");
const startBtn = document.getElementById("startBtn");
const muteBtn = document.getElementById("muteBtn");

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
async function predictClassState(model, sourceCanvas, box, classNames, positiveClass) {
  cropCtx.drawImage(sourceCanvas, box.xMin, box.yMin, box.xMax - box.xMin, box.yMax - box.yMin, 0, 0, cropCanvas.width, cropCanvas.height);
  // .data() is async and non-blocking; dataSync() stalls the main thread waiting on the GPU every call
  const predictionTensor = tf.tidy(() => {
    const input = tf.browser.fromPixels(cropCanvas).toFloat().div(255.0).expandDims(0);
    return model.predict(input);
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

async function drawEye(side, box, ear, model, earCfg, landmarks) {
  const { positiveProbability: cnnClosedness } = await predictClassState(model, canvas, box, CONFIG.eyeClassNames, "closed");
  const earClosedness = linearScore(ear, earCfg.openThreshold, earCfg.closedThreshold);
  // the CNN can raise closedness when EAR is ambiguous, but can't override a clearly-open EAR reading
  const closedness = earClosedness > 0 ? Math.max(cnnClosedness, earClosedness) : 0.0;
  const label = closedness >= 0.5 ? "closed" : "open";

  const connectors = side === "left" ? FaceLandmarker.FACE_LANDMARKS_LEFT_EYE : FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE;
  drawingUtils.drawConnectors(landmarks, connectors, { color: label === "open" ? OPEN_COLOR : CLOSED_COLOR, lineWidth: 2 });
  return { closedness, label, ear };
}

async function drawMouth(box, mar, model, landmarks) {
  const { positiveProbability: cnnOpenness } = await predictClassState(model, canvas, box, CONFIG.yawnClassNames, "yawn");
  const marOpenness = linearScore(mar, CONFIG.mar.closedThreshold, CONFIG.mar.openThreshold);
  // either signal suggesting a yawn is enough -- a safety monitor should not need both to agree
  const openness = Math.max(cnnOpenness, marOpenness);
  const label = openness >= 0.5 ? "yawn" : "no yawn";

  drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: label === "yawn" ? CLOSED_COLOR : MOUTH_COLOR, lineWidth: 2 });
  return { openness, label, mar };
}

function headPoseStatus(pitch, landmarks) {
  if (pitch === null) return null;
  const droop = linearScore(pitch, CONFIG.headPose.uprightThreshold, CONFIG.headPose.droopedThreshold);
  const label = droop >= 0.5 ? "nodding" : "upright";
  drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: label === "upright" ? HEAD_COLOR : CLOSED_COLOR, lineWidth: 2 });
  return { droop, label, pitch };
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
function drawStatusPanel(left, right, mouth, head) {
  const lines = [
    left && { text: `left eye: ${left.label} ${Math.round(left.closedness * 100)}% | EAR ${left.ear.toFixed(2)}`, color: left.label === "open" ? OPEN_COLOR : CLOSED_COLOR },
    right && { text: `right eye: ${right.label} ${Math.round(right.closedness * 100)}% | EAR ${right.ear.toFixed(2)}`, color: right.label === "open" ? OPEN_COLOR : CLOSED_COLOR },
    mouth && { text: `mouth: ${mouth.label} ${Math.round(mouth.openness * 100)}% | MAR ${mouth.mar.toFixed(2)}`, color: mouth.label === "yawn" ? CLOSED_COLOR : MOUTH_COLOR },
    head && { text: `head: ${head.label} | pitch ${head.pitch.toFixed(1)} deg`, color: head.label === "upright" ? HEAD_COLOR : CLOSED_COLOR },
  ].filter(Boolean);
  if (!lines.length) return;

  const panelY = 88;
  ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
  ctx.fillRect(10, panelY, 260, lines.length * 18 + 10);
  ctx.font = "11px sans-serif";
  lines.forEach((line, i) => {
    ctx.fillStyle = line.color;
    ctx.fillText(line.text, 18, panelY + 18 * (i + 1));
  });
}

async function calibrateEar(faceLandmarker, durationSec, closedFraction) {
  const samples = [];
  const start = performance.now();
  while (performance.now() - start < durationSec * 1000) {
    updateMirror();
    const result = faceLandmarker.detectForVideo(mirrorCanvas, performance.now());
    ctx.drawImage(mirrorCanvas, 0, 0, canvas.width, canvas.height);
    if (result.faceLandmarks.length) {
      const lm = result.faceLandmarks[0];
      samples.push(eyeAspectRatio(lm, canvas.width, canvas.height, LEFT_EYE_EAR_POINTS));
      samples.push(eyeAspectRatio(lm, canvas.width, canvas.height, RIGHT_EYE_EAR_POINTS));
    }
    const remaining = durationSec - (performance.now() - start) / 1000;
    ctx.fillStyle = "#00c8ff";
    ctx.font = "bold 14px sans-serif";
    ctx.fillText("Calibrating -- look at the camera normally", 20, 40);
    ctx.fillText(`${remaining.toFixed(1)}s`, 20, 80);
    await new Promise((r) => requestAnimationFrame(r));
  }

  if (samples.length < 10) return null;
  const baseline = median(samples);
  return { openThreshold: baseline, closedThreshold: baseline * closedFraction };
}

async function main() {
  // plain "cpu" is a slow reference backend, so prefer webgl, then the XNNPACK/SIMD-accelerated "wasm" backend
  tf.wasm.setWasmPaths("lib/");
  let backendOk = await tf.setBackend("webgl");
  if (!backendOk) backendOk = await tf.setBackend("wasm");
  await tf.ready();
  console.log("TF.js backend:", tf.getBackend());

  statusEl.textContent = "Loading models...";
  const eyeModel = await tf.loadLayersModel("models/eye/model.json");
  const yawnModel = await tf.loadLayersModel("models/yawn/model.json");

  // warm up both models so shader compilation happens now, not on the first real frame
  tf.tidy(() => {
    const dummy = tf.zeros([1, ...CONFIG.imageSize, 3]);
    eyeModel.predict(dummy);
    yawnModel.predict(dummy);
  });

  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFacialTransformationMatrixes: true,
  });

  statusEl.textContent = "Requesting camera...";
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
  video.srcObject = stream;
  await new Promise((r) => (video.onloadedmetadata = r));
  video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  mirrorCanvas.width = video.videoWidth;
  mirrorCanvas.height = video.videoHeight;

  statusEl.textContent = "";
  let earCfg = CONFIG.ear;
  let paused = false;
  let noFaceSince = null;

  async function runCalibration() {
    // pause the main render loop so it doesn't fight the calibration overlay for the same canvas
    paused = true;
    stopAlarm();
    const calibrated = await calibrateEar(faceLandmarker, CONFIG.calibration.durationSec, CONFIG.calibration.closedFraction);
    if (calibrated) {
      earCfg = calibrated;
      statusEl.textContent = `Calibrated EAR thresholds: open>=${earCfg.openThreshold.toFixed(2)}, closed<=${earCfg.closedThreshold.toFixed(2)}`;
    } else {
      statusEl.textContent = "Calibration failed (no face detected) -- using default EAR thresholds.";
    }
    paused = false;
  }

  await runCalibration();
  muteBtn.style.display = "inline-block";

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
    if (paused) {
      requestAnimationFrame(frame);
      return;
    }
    updateMirror();
    const result = faceLandmarker.detectForVideo(mirrorCanvas, performance.now());
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
      const leftBox = boxFromLandmarks(lm, LEFT_EYE_IDS, w, h, 0.15);
      const rightBox = boxFromLandmarks(lm, RIGHT_EYE_IDS, w, h, 0.15);
      const mouthBox = boxFromLandmarks(lm, MOUTH_IDS, w, h, 0.3);

      if (leftBox && rightBox && mouthBox) {
        leftResult = await drawEye("left", leftBox, leftEar, eyeModel, earCfg, lm);
        rightResult = await drawEye("right", rightBox, rightEar, eyeModel, earCfg, lm);
        // average both eyes so one drooping eyelid still raises the score
        eyeClosedness = (leftResult.closedness + rightResult.closedness) / 2;
        mouthResult = await drawMouth(mouthBox, mouthAspectRatio(lm, w, h), yawnModel, lm);
        mouthOpenness = mouthResult.openness;
        if (result.facialTransformationMatrixes.length) {
          headResult = headPoseStatus(pitchFromTransformMatrix(result.facialTransformationMatrixes[0].data), lm);
          headDroop = headResult?.droop ?? null;
        }
      }
    }
    drawStatusPanel(leftResult, rightResult, mouthResult, headResult);

    const now = performance.now() / 1000;
    const eyeStatus = eyeMonitor.update(eyeClosedness, now);
    const yawnStatus = yawnMonitor.update(mouthOpenness, now);
    const nodStatus = nodMonitor.update(headDroop, now);

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

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

startBtn.addEventListener("click", () => {
  // unlock the alarm's AudioContext inside this click, browsers block audio without a user gesture
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  startBtn.disabled = true;
  main().catch((err) => {
    statusEl.textContent = `Error: ${err.message}`;
    console.error(err);
  });
});

muteBtn.addEventListener("click", () => {
  muted = !muted;
  muteBtn.textContent = muted ? "Unmute Alarm" : "Mute Alarm";
  if (muted) stopAlarm();
});
