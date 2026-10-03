import numpy as np
import pandas as pd
import tensorflow as tf

from watch.config import load_config
from watch.data.split import load_split

AUGMENT = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(10 / 360, fill_mode="nearest"),
        tf.keras.layers.RandomZoom(0.1, fill_mode="nearest"),
        tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),
    ]
)


def _decode(path, grayscale: bool, image_size):
    image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    image = tf.image.resize(image, image_size)
    if grayscale:
        image = tf.image.rgb_to_grayscale(image)
    return tf.cast(tf.round(image), tf.uint8)


def _to_input(image):
    image = tf.cast(image, tf.float32) / 255.0
    return tf.image.grayscale_to_rgb(image) if image.shape[-1] == 1 else image


def _augment(images):
    images = AUGMENT(images, training=True)
    brightness = tf.random.uniform([tf.shape(images)[0], 1, 1, 1], 0.8, 1.2)
    return tf.clip_by_value(images * brightness, 0.0, 1.0)


def make_dataset(rows: pd.DataFrame, name: str, training: bool = False, batch_size: int | None = None) -> tf.data.Dataset:
    config = load_config()
    spec = config["models"][name]
    labels = rows["label"].map(spec["class_names"].index).to_numpy()
    one_hot = tf.one_hot(labels, len(spec["class_names"]))
    image_size = tuple(config["image_size"])

    dataset = tf.data.Dataset.from_tensor_slices((rows["filepath"].to_numpy(), one_hot))
    dataset = dataset.map(lambda path, label: (_decode(path, spec["grayscale"], image_size), label), num_parallel_calls=tf.data.AUTOTUNE).cache()
    if training:
        dataset = dataset.shuffle(len(rows), seed=config["seed"], reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size or spec["batch_size"]).map(lambda x, y: (_to_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.map(lambda x, y: (_augment(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)


def load_datasets(name: str) -> dict[str, tf.data.Dataset]:
    manifest = load_split(name)
    return {split: make_dataset(manifest[manifest["split"] == split], name, training=split == "train") for split in ("train", "val", "test")}


def labels_of(name: str, split: str) -> np.ndarray:
    manifest = load_split(name)
    class_names = load_config()["models"][name]["class_names"]
    return manifest.loc[manifest["split"] == split, "label"].map(class_names.index).to_numpy()


def class_weights(name: str) -> dict[int, float]:
    """Balanced weights so a minority class such as yawn is not ignored."""
    labels = labels_of(name, "train")
    counts = np.bincount(labels, minlength=len(load_config()["models"][name]["class_names"]))
    return {i: len(labels) / (len(counts) * count) for i, count in enumerate(counts)}
