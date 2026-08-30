// Mirrors configs/config.yaml -- keep these in sync if the Python config changes.
const CONFIG = {
  imageSize: [64, 64],
  eyeClassNames: ["closed", "open"],
  yawnClassNames: ["no_yawn", "yawn"],

  alertness: { closedDurationThresholdSec: 1.5, perclosWindowSec: 5.0, perclosThreshold: 0.5, closednessThreshold: 0.4 },
  ear: { openThreshold: 0.35, closedThreshold: 0.10 },
  calibration: { durationSec: 4.0, closedFraction: 0.3 },
  mar: { closedThreshold: 0.3, openThreshold: 0.6 },
  yawn: { closedDurationThresholdSec: 3.0, perclosWindowSec: 8.0, perclosThreshold: 0.8, closednessThreshold: 0.5, dipGracePeriodSec: 0.5 },
  headPose: { uprightThreshold: 8, droopedThreshold: 20 },
  nod: { closedDurationThresholdSec: 2.0, perclosWindowSec: 6.0, perclosThreshold: 0.6, closednessThreshold: 0.5, dipGracePeriodSec: 0.0 },
};
