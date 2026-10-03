from pathlib import Path

import cv2

from watch.face import FaceTracker
from watch.model import Classifier


def predict(image_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image {image_path}")

    with FaceTracker(static_image_mode=True) as tracker:
        regions = tracker.extract(image)
    if regions is None:
        print("No face detected.")
        return

    eye_model, yawn_model = Classifier("eye"), Classifier("yawn")
    for side, eye in regions.eyes.items():
        label, confidence, _ = eye_model.predict(eye.crop)
        print(f"{side} eye: {label} ({confidence:.1%}, EAR {eye.ratio:.2f})")
    if regions.mouth:
        label, confidence, _ = yawn_model.predict(regions.mouth.crop)
        print(f"mouth: {label} ({confidence:.1%}, MAR {regions.mouth.ratio:.2f})")
    else:
        print("mouth: not readable")
    if regions.pitch is None:
        print("head: pose unavailable")
    else:
        print(f"head: pitch {regions.pitch:.1f} deg, roll {regions.roll:.1f} deg")
