// Generated from configs/config.yaml by ./scripts/export.sh, do not edit.
const CONFIG = {
  "imageSize": [
    64,
    64
  ],
  "eyeClassNames": [
    "eyes_closed",
    "eyes_opened"
  ],
  "yawnClassNames": [
    "no_yawn",
    "yawn"
  ],
  "alertness": {
    "closedDurationThresholdSec": 1.5,
    "perclosWindowSec": 5.0,
    "perclosThreshold": 0.75,
    "closednessThreshold": 0.85
  },
  "yawn": {
    "closedDurationThresholdSec": 2.0,
    "perclosWindowSec": 8.0,
    "perclosThreshold": 0.8,
    "closednessThreshold": 0.5,
    "dipGracePeriodSec": 0.5
  },
  "nod": {
    "closedDurationThresholdSec": 1.5,
    "perclosWindowSec": 6.0,
    "perclosThreshold": 0.9,
    "closednessThreshold": 1.0,
    "dipGracePeriodSec": 0.0
  },
  "ear": {
    "openThreshold": 0.35,
    "closedThreshold": 0.1
  },
  "mar": {
    "closedThreshold": 0.3,
    "openThreshold": 0.6
  },
  "headPose": {
    "uprightThreshold": 8,
    "droopedThreshold": 33,
    "rollUprightThreshold": 5,
    "rollDroopedThreshold": 30
  },
  "calibration": {
    "durationSec": 4.0,
    "closedOffset": 0.1,
    "droopedOffset": 25.0,
    "rollDroopedOffset": 25.0
  },
  "escalation": {
    "windowSec": 30.0,
    "calmSec": 5.0,
    "persistSec": {
      "eyes": 3.0,
      "mouth": null,
      "head": 4.0
    },
    "repeatCount": {
      "eyes": 2,
      "mouth": 3,
      "head": 2
    }
  },
  "live": {
    "minCnnConfidence": 0.65,
    "earGlareTolerance": 0.15,
    "lowDetailStd": 12.0,
    "lowLightMean": 0.3,
    "lowLightTarget": 0.45,
    "levelMinAngleDeg": 15,
    "sharpMin": 100.0,
    "cameraRetrySec": 1.0,
    "cameraLostAlarmSec": 3.0,
    "blurryBelow": 40.0,
    "faceDetectionConfidence": 0.3,
    "faceTrackingConfidence": 0.5,
    "recalibrateGapSec": 3.0,
    "smoothingWindow": 5
  }
};
