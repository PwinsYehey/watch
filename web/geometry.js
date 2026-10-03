// Browser port of src/watch/face.py using the same MediaPipe landmark indices.
const LEFT_EYE_EAR_POINTS = { horizontal: [33, 133], verticalA: [160, 144], verticalB: [158, 153] };
const RIGHT_EYE_EAR_POINTS = { horizontal: [362, 263], verticalA: [385, 380], verticalB: [387, 373] };
const LEFT_EYE_IDS = [33, 133, 159, 145, 160, 144, 158, 153];
const RIGHT_EYE_IDS = [362, 263, 386, 374, 385, 380, 387, 373];
const MOUTH_IDS = [61, 291, 13, 14, 78, 308, 0, 17];
const MOUTH_MAR_POINTS = { horizontal: [61, 291], vertical: [13, 14] };

function pointPx(landmarks, index, w, h) {
  return { x: landmarks[index].x * w, y: landmarks[index].y * h };
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
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

// Approximates the solvePnP head pose from MediaPipe's transform matrix, in degrees.
function headPoseFromTransformMatrix(data) {
  const r = [data[0], data[1], data[2], data[4], data[5], data[6], data[8], data[9], data[10]];
  const sy = Math.sqrt(r[0] * r[0] + r[3] * r[3]);
  const pitchRad = sy < 1e-6 ? Math.atan2(-r[5], r[4]) : Math.atan2(r[7], r[8]);
  const rawPitchDeg = (pitchRad * 180) / Math.PI;
  const normalized = ((rawPitchDeg + 180) % 360) - 180;
  // Flip the sign so nodding forward increases pitch, as in live.py.
  const pitch = -normalized;

  const rollRad = Math.atan2(r[3], r[0]);
  const roll = Math.abs((rollRad * 180) / Math.PI);

  return { pitch, roll };
}
