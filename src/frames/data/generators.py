"""Builds Keras data generators from the split manifest CSV via flow_from_dataframe, since images aren't sorted into per-class folders."""

from pathlib import Path

import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from frames.config import PROJECT_ROOT


def load_split_manifest(splits_dir: str | Path) -> pd.DataFrame:
    path = Path(splits_dir) / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `python cli/make_splits.py` first.")
    manifest = pd.read_csv(path)
    # manifest.csv stores paths relative to the project root so it's portable across machines/clones
    manifest["filepath"] = manifest["filepath"].apply(lambda p: str(PROJECT_ROOT / p))
    return manifest


def make_generators(config: dict):
    manifest = load_split_manifest(config["paths"]["splits_dir"])
    image_size = tuple(config["preprocessing"]["image_size"])
    batch_size = config["training"]["batch_size"]
    class_names = config["model"]["class_names"]

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        zoom_range=0.10,
        width_shift_range=0.10,
        height_shift_range=0.10,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode="nearest",
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    def _flow(datagen, split_name, shuffle):
        df = manifest[manifest["split"] == split_name]
        return datagen.flow_from_dataframe(
            df,
            x_col="filepath",
            y_col="label",
            target_size=image_size,
            batch_size=batch_size,
            class_mode="categorical",
            classes=class_names,
            shuffle=shuffle,
        )

    train_generator = _flow(train_datagen, "train", shuffle=True)
    val_generator = _flow(eval_datagen, "val", shuffle=False)
    test_generator = _flow(eval_datagen, "test", shuffle=False)

    return train_generator, val_generator, test_generator
