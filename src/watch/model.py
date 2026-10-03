import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from watch.config import load_config, model_path


def _custom_cnn(input_shape):
    stack = [layers.Input(shape=input_shape)]
    for filters in (32, 64, 128):
        stack += [layers.Conv2D(filters, 3, padding="same", activation="relu"), layers.BatchNormalization(), layers.MaxPooling2D()]
    return models.Sequential(stack, name="custom_cnn_backbone")


def _tiny_cnn(input_shape):
    stack = [layers.Input(shape=input_shape)]
    for filters in (16, 32):
        stack += [layers.Conv2D(filters, 3, padding="same", activation="relu"), layers.MaxPooling2D()]
    return models.Sequential(stack, name="tiny_cnn_backbone")


def _mobilenetv2(input_shape):
    base = tf.keras.applications.MobileNetV2(include_top=False, input_shape=input_shape, weights="imagenet")
    base.trainable = False
    return base


BACKBONES = {"custom_cnn": _custom_cnn, "tiny_cnn": _tiny_cnn, "mobilenetv2": _mobilenetv2}


def build_model(architecture: str, dropout_rate: float, dense_units: int, learning_rate: float, num_classes: int) -> tf.keras.Model:
    input_shape = (*load_config()["image_size"], 3)
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            BACKBONES[architecture](input_shape),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(dropout_rate),
            layers.Dense(dense_units, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def load_model(name: str, path=None) -> tf.keras.Model:
    path = path or model_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found, run ./scripts/train.sh {name} first.")
    return tf.keras.models.load_model(path)


class Classifier:
    """Single-crop inference for one trained model, with the same preprocessing as training."""

    def __init__(self, name: str):
        config = load_config()
        spec = config["models"][name]
        self.model = load_model(name)
        self.class_names = spec["class_names"]
        self.positive_index = self.class_names.index(spec["positive_class"])
        self.grayscale = spec["grayscale"]
        self.image_size = tuple(config["image_size"])
        # Compiled once, which avoids most of the per-call overhead of running the Keras model eagerly.
        self._infer = tf.function(lambda x: self.model(x, training=False), input_signature=[tf.TensorSpec([None, *self.image_size, 3], tf.float32)])

    def _prepare(self, crop_bgr: np.ndarray) -> np.ndarray:
        image = cv2.resize(crop_bgr, self.image_size)
        if self.grayscale:
            image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.float32) / 255.0

    def predict(self, crop_bgr: np.ndarray) -> tuple[str, float, float]:
        """Returns (label, confidence, positive-class probability)."""
        return self.predict_many([crop_bgr])[0]

    def predict_many(self, crops_bgr: list[np.ndarray]) -> list[tuple[str, float, float]]:
        """Like predict, for several crops in one batch."""
        probabilities = self._infer(np.stack([self._prepare(crop) for crop in crops_bgr])).numpy()
        results = []
        for row in probabilities:
            index = int(np.argmax(row))
            results.append((self.class_names[index], float(row[index]), float(row[self.positive_index])))
        return results
