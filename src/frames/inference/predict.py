import argparse

import cv2
import numpy as np

from frames.config import load_config
from frames.models.build_model import load_trained_model
from frames.preprocessing.facemesh import extract_face_regions


def _predict_class_state(model, crop_bgr: np.ndarray, image_size: tuple, class_names: list[str], positive_class: str):
    """Returns (label, confidence, positive_class_probability), the raw softmax score for `positive_class`."""
    resized = cv2.resize(crop_bgr, image_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0
    batch = np.expand_dims(rgb, axis=0)

    # direct call instead of model.predict(), which costs ~5x more per single-frame call
    prediction = model(batch, training=False)[0].numpy()
    predicted_index = int(np.argmax(prediction))
    positive_probability = float(prediction[class_names.index(positive_class)])
    return class_names[predicted_index], float(prediction[predicted_index]), positive_probability


def predict_eye_state(model, eye_crop_bgr: np.ndarray, image_size: tuple, class_names: list[str]):
    return _predict_class_state(model, eye_crop_bgr, image_size, class_names, positive_class="closed")


def predict_yawn_state(model, mouth_crop_bgr: np.ndarray, image_size: tuple, class_names: list[str]):
    return _predict_class_state(model, mouth_crop_bgr, image_size, class_names, positive_class="yawn")


def main():
    parser = argparse.ArgumentParser(description="Report eye state, mouth MAR, and head pitch for a face photo.")
    parser.add_argument("image_path", help="Path to a full face / helmet-cam image")
    args = parser.parse_args()

    config = load_config()
    image_size = tuple(config["preprocessing"]["image_size"])
    class_names = config["model"]["class_names"]
    yawn_class_names = config["yawn_model"]["class_names"]

    model = load_trained_model(config)
    yawn_model = load_trained_model(config, checkpoint_name="best_yawn_model.keras")

    image = cv2.imread(args.image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image_path}")

    regions = extract_face_regions(image)
    if regions is None:
        print("No face detected in this image (fully occluded, or no face present).")
        return

    for side, eye in regions.eyes.items():
        label, confidence, _closed_probability = predict_eye_state(model, eye.crop, image_size, class_names)
        print(f"{side} eye: {label} ({confidence * 100:.1f}% confidence, EAR={eye.ear:.2f})")

    label, confidence, _yawn_probability = predict_yawn_state(yawn_model, regions.mouth.crop, image_size, yawn_class_names)
    print(f"mouth: {label} ({confidence * 100:.1f}% confidence, MAR={regions.mouth.mar:.2f})")
    print(f"head pitch: {regions.pitch:.1f} deg" if regions.pitch is not None else "head pitch: unavailable")


if __name__ == "__main__":
    main()
