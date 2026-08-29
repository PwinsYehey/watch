"""Single-image demo: predicts per-eye open/closed state from a full face/helmet photo."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from frames.config import load_config
from frames.preprocessing.facemesh import extract_eye_crops


def predict_eye_state(model, eye_crop_bgr: np.ndarray, image_size: tuple, class_names: list[str]):
    resized = cv2.resize(eye_crop_bgr, image_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0
    batch = np.expand_dims(rgb, axis=0)

    prediction = model.predict(batch, verbose=0)[0]
    predicted_index = int(np.argmax(prediction))
    return class_names[predicted_index], float(prediction[predicted_index])


def main():
    parser = argparse.ArgumentParser(description="Predict eye state for both eyes in a face photo.")
    parser.add_argument("image_path", help="Path to a full face / helmet-cam image")
    args = parser.parse_args()

    config = load_config()
    image_size = tuple(config["preprocessing"]["image_size"])
    class_names = config["model"]["class_names"]

    model_path = Path(config["paths"]["models_dir"]) / "best_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found -- run `python scripts/train.py` first.")
    model = tf.keras.models.load_model(model_path)

    image = cv2.imread(args.image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image_path}")

    eyes = extract_eye_crops(image)
    if eyes is None:
        print("No face/eyes detected in this image (fully occluded, or no face present).")
        return

    for side, crop in eyes.items():
        label, confidence = predict_eye_state(model, crop, image_size, class_names)
        print(f"{side} eye: {label} ({confidence * 100:.1f}% confidence)")


if __name__ == "__main__":
    main()
