// Browser port of src/watch/face.py using the same MediaPipe landmark indices.
const LEFT_EYE_EAR_POINTS = { horizontal: [33, 133], verticalA: [160, 144], verticalB: [158, 153] };
const RIGHT_EYE_EAR_POINTS = { horizontal: [362, 263], verticalA: [385, 380], verticalB: [387, 373] };
const LEFT_EYE_IDS = [33, 133, 159, 145, 160, 144, 158, 153];
const RIGHT_EYE_IDS = [362, 263, 386, 374, 385, 380, 387, 373];
const MOUTH_IDS = [61, 291, 13, 14, 78, 308, 0, 17];
const MOUTH_MAR_POINTS = { horizontal: [61, 291], vertical: [13, 14] };
const LEFT_EYE_AXIS = [33, 133];
const RIGHT_EYE_AXIS = [362, 263];
const MOUTH_AXIS = [61, 291];

// 3D keeps ratios steady when the face turns.
function pointPx(landmarks, index, w, h) {
  return { x: landmarks[index].x * w, y: landmarks[index].y * h, z: (landmarks[index].z || 0) * w };
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

function screenWidth(landmarks, axis, w, h) {
  return Math.hypot((landmarks[axis[0]].x - landmarks[axis[1]].x) * w, (landmarks[axis[0]].y - landmarks[axis[1]].y) * h);
}

function eyeAspectRatio(landmarks, w, h, points) {
  const horizontal = dist(pointPx(landmarks, points.horizontal[0], w, h), pointPx(landmarks, points.horizontal[1], w, h));
  const verticalA = dist(pointPx(landmarks, points.verticalA[0], w, h), pointPx(landmarks, points.verticalA[1], w, h));
  const verticalB = dist(pointPx(landmarks, points.verticalB[0], w, h), pointPx(landmarks, points.verticalB[1], w, h));
  if (horizontal === 0) return 0.0;
  return (verticalA + verticalB) / (2.0 * horizontal);
}

function mouthAspectRatio(landmarks, w, h) {
  const horizontal = dist(pointPx(landmarks, MOUTH_MAR_POINTS.horizontal[0], w, h), pointPx(landmarks, MOUTH_MAR_POINTS.horizontal[1], w, h));
  const vertical = dist(pointPx(landmarks, MOUTH_MAR_POINTS.vertical[0], w, h), pointPx(landmarks, MOUTH_MAR_POINTS.vertical[1], w, h));
  if (horizontal === 0) return 0.0;
  return vertical / horizontal;
}

// Mirrors crop_box in src/watch/face.py and returns a pixel-space box.
function boxFromLandmarks(landmarks, ids, w, h, margin, square = false) {
  const xs = ids.map((i) => landmarks[i].x * w);
  const ys = ids.map((i) => landmarks[i].y * h);
  const xMin0 = Math.min(...xs), xMax0 = Math.max(...xs);
  const yMin0 = Math.min(...ys), yMax0 = Math.max(...ys);
  const boxW = xMax0 - xMin0, boxH = yMax0 - yMin0;

  let xMin = xMin0 - boxW * margin;
  let xMax = xMax0 + boxW * margin;
  let yMin = yMin0 - boxH * margin;
  let yMax = yMax0 + boxH * margin;

  if (square) {
    const cx = (xMin + xMax) / 2, cy = (yMin + yMax) / 2;
    const side = Math.max(xMax - xMin, yMax - yMin);
    xMin = cx - side / 2; xMax = cx + side / 2;
    yMin = cy - side / 2; yMax = cy + side / 2;
  }

  xMin = Math.max(xMin, 0);
  xMax = Math.min(xMax, w);
  yMin = Math.max(yMin, 0);
  yMax = Math.min(yMax, h);
  if (!Number.isFinite(xMin) || !Number.isFinite(xMax) || !Number.isFinite(yMin) || !Number.isFinite(yMax)) return null;
  if (xMax <= xMin || yMax <= yMin) return null;
  return { xMin, yMin, xMax, yMax };
}

// Mirrors aligned_crop in src/watch/face.py.
function alignedCropFromLandmarks(landmarks, ids, axis, w, h, margin, square = false) {
  const [a, b] = axis.map((i) => pointPx(landmarks, i, w, h));
  const tilt = Math.atan2(b.y - a.y, b.x - a.x);
  const angle = Math.abs((tilt * 180) / Math.PI) < CONFIG.live.levelMinAngleDeg ? 0 : tilt;
  const points = ids.map((i) => pointPx(landmarks, i, w, h));
  const cx = points.reduce((s, p) => s + p.x, 0) / points.length;
  const cy = points.reduce((s, p) => s + p.y, 0) / points.length;
  const cos = Math.cos(-angle), sin = Math.sin(-angle);
  const xs = points.map((p) => cos * (p.x - cx) - sin * (p.y - cy));
  const ys = points.map((p) => sin * (p.x - cx) + cos * (p.y - cy));

  let xMin = Math.min(...xs), xMax = Math.max(...xs), yMin = Math.min(...ys), yMax = Math.max(...ys);
  const padX = (xMax - xMin) * margin, padY = (yMax - yMin) * margin;
  xMin -= padX; xMax += padX; yMin -= padY; yMax += padY;
  if (square) {
    const mx = (xMin + xMax) / 2, my = (yMin + yMax) / 2;
    const half = Math.max(xMax - xMin, yMax - yMin) / 2;
    xMin = mx - half; xMax = mx + half; yMin = my - half; yMax = my + half;
  }
  if (!(xMax > xMin && yMax > yMin)) return null;
  return { cx, cy, angle, xMin, yMin, width: xMax - xMin, height: yMax - yMin };
}

// Approximates the solvePnP head pose from MediaPipe's transform matrix, in degrees.
function headPoseFromTransformMatrix(data, landmarks, w, h) {
  const r = [data[0], data[1], data[2], data[4], data[5], data[6], data[8], data[9], data[10]];
  const sy = Math.sqrt(r[0] * r[0] + r[3] * r[3]);
  const pitchRad = sy < 1e-6 ? Math.atan2(-r[5], r[4]) : Math.atan2(r[7], r[8]);
  const rawPitchDeg = (pitchRad * 180) / Math.PI;
  const normalized = ((rawPitchDeg + 180) % 360) - 180;
  // Flip the sign so nodding forward increases pitch, as in live.py.
  const pitch = -normalized;

  return { pitch, roll: eyeLineRoll(landmarks, w, h) };
}

function eyeLineRoll(landmarks, w, h) {
  const a = landmarks[LEFT_EYE_AXIS[0]], b = landmarks[RIGHT_EYE_AXIS[1]];
  return Math.abs((Math.atan2((b.y - a.y) * h, (b.x - a.x) * w) * 180) / Math.PI);
}

// Mirrors shoulders and body_slump in src/watch/body.py.
const NOSE = 0, LEFT_SHOULDER = 11, RIGHT_SHOULDER = 12, LEFT_ELBOW = 13, RIGHT_ELBOW = 14, LEFT_WRIST = 15, RIGHT_WRIST = 16;

function shouldersFromPose(landmarks, w, h, minVisibility) {
  const left = landmarks[LEFT_SHOULDER], right = landmarks[RIGHT_SHOULDER];
  if (Math.min(left.visibility ?? 0, right.visibility ?? 0) < minVisibility) return null;
  const lx = left.x * w, ly = left.y * h, rx = right.x * w, ry = right.y * h;
  const width = Math.hypot(lx - rx, ly - ry);
  if (width < 1) return null;
  const nose = landmarks[NOSE];
  const head = (nose.visibility ?? 0) >= minVisibility ? (nose.y * h - (ly + ry) / 2) / width : null;
  return { centerX: (lx + rx) / 2, centerY: (ly + ry) / 2, width, tilt: (Math.atan2(ry - ly, Math.abs(rx - lx)) * 180) / Math.PI, left: { x: lx, y: ly }, right: { x: rx, y: ry }, head, armRaised: armRaised(landmarks, minVisibility) };
}

function armRaised(landmarks, minVisibility) {
  return [[LEFT_SHOULDER, [LEFT_ELBOW, LEFT_WRIST]], [RIGHT_SHOULDER, [RIGHT_ELBOW, RIGHT_WRIST]]].some(([shoulder, joints]) =>
    joints.some((j) => (landmarks[j].visibility ?? 0) >= minVisibility && landmarks[j].y < landmarks[shoulder].y));
}

function bodySlump(pose, baseline, cfg) {
  const drop = Math.max(0, (pose.centerY - baseline.centerY) / baseline.width);
  const lean = Math.abs(pose.tilt - baseline.tilt);
  const headDrop = pose.head != null && baseline.head != null ? Math.max(0, pose.head - baseline.head) : null;
  const countedLean = pose.armRaised ? 0 : lean;
  return { score: Math.min(1, Math.max(drop / cfg.dropLimit, countedLean / cfg.leanLimitDeg, (headDrop ?? 0) / cfg.headDropLimit)), drop, lean, headDrop };
}
