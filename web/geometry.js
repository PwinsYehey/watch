// Ports src/frames/preprocessing/facemesh.py -- same MediaPipe FaceMesh landmark topology, so the indices match exactly.
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

// lower EAR means more closed, mirrors _eye_aspect_ratio in facemesh.py
function eyeAspectRatio(landmarks, w, h, points) {
  const horizontal = dist(pointPx(landmarks, points.horizontal[0], w, h), pointPx(landmarks, points.horizontal[1], w, h));
  const verticalA = dist(pointPx(landmarks, points.verticalA[0], w, h), pointPx(landmarks, points.verticalA[1], w, h));
  const verticalB = dist(pointPx(landmarks, points.verticalB[0], w, h), pointPx(landmarks, points.verticalB[1], w, h));
  if (horizontal === 0) return 0.0;
  return (verticalA + verticalB) / (2.0 * horizontal);
}

// higher MAR means more open, mirrors _mouth_aspect_ratio in facemesh.py
function mouthAspectRatio(landmarks, w, h) {
  const horizontal = dist(pointPx(landmarks, MOUTH_MAR_POINTS.horizontal[0], w, h), pointPx(landmarks, MOUTH_MAR_POINTS.horizontal[1], w, h));
  const vertical = dist(pointPx(landmarks, MOUTH_MAR_POINTS.vertical[0], w, h), pointPx(landmarks, MOUTH_MAR_POINTS.vertical[1], w, h));
  if (horizontal === 0) return 0.0;
  return vertical / horizontal;
}

// mirrors _crop_from_landmarks in facemesh.py -- returns a bounding box in pixel space
function boxFromLandmarks(landmarks, ids, w, h, margin) {
  const xs = ids.map((i) => landmarks[i].x * w);
  const ys = ids.map((i) => landmarks[i].y * h);
  const xMin0 = Math.min(...xs), xMax0 = Math.max(...xs);
  const yMin0 = Math.min(...ys), yMax0 = Math.max(...ys);
  const boxW = xMax0 - xMin0, boxH = yMax0 - yMin0;

  const xMin = Math.max(xMin0 - boxW * margin, 0);
  const xMax = Math.min(xMax0 + boxW * margin, w);
  const yMin = Math.max(yMin0 - boxH * margin, 0);
  const yMax = Math.min(yMax0 + boxH * margin, h);
  if (!Number.isFinite(xMin) || !Number.isFinite(xMax) || !Number.isFinite(yMin) || !Number.isFinite(yMax)) return null;
  if (xMax <= xMin || yMax <= yMin) return null;
  return { xMin, yMin, xMax, yMax };
}

// stands in for the Python cv2.solvePnP head-pose estimate; won't numerically match the desktop app exactly.
function pitchFromTransformMatrix(data) {
  const r = [data[0], data[1], data[2], data[4], data[5], data[6], data[8], data[9], data[10]];
  const sy = Math.sqrt(r[0] * r[0] + r[3] * r[3]);
  const pitchRad = sy < 1e-6 ? Math.atan2(-r[5], r[4]) : Math.atan2(r[7], r[8]);
  const rawPitchDeg = (pitchRad * 180) / Math.PI;
  // MediaPipe's own matrix already reads near 0 for a neutral head, unlike solvePnP's near-180 convention -- just normalize to (-180, 180]
  const normalized = ((rawPitchDeg + 180) % 360) - 180;
  // this axis reads opposite the config's convention (nodding forward should increase the value, toward drooped_threshold)
  return -normalized;
}
