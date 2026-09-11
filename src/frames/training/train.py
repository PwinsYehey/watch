from pathlib import Path

import pandas as pd
import tensorflow as tf

from frames.config import load_config
from frames.data.generators import make_generators
from frames.models.build_model import build_model


def _train(config: dict, model_cfg: dict, train_cfg: dict, manifest_filename: str, checkpoint_name: str, history_name: str):
    tf.keras.utils.set_random_seed(config["project"]["seed"])
    image_size = tuple(config["preprocessing"]["image_size"])

    train_generator, val_generator, _ = make_generators(
        config,
        manifest_filename=manifest_filename,
        batch_size=train_cfg["batch_size"],
        class_names=model_cfg["class_names"],
    )

    model, early_stop, reduce_lr = build_model(
        architecture=model_cfg["architecture"],
        input_shape=image_size + (3,),
        dropout_rate=model_cfg["dropout_rate"],
        dense_units=model_cfg["dense_units"],
        learning_rate=model_cfg["learning_rate"],
        num_classes=model_cfg["num_classes"],
        early_stopping_patience=train_cfg["early_stopping_patience"],
        reduce_lr_patience=train_cfg["reduce_lr_patience"],
    )

    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / checkpoint_name

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=train_cfg["epochs"],
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1,
    )

    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    history_path = results_dir / history_name
    pd.DataFrame(history.history).to_csv(history_path, index=False)

    print(f"\nBest model saved to: {checkpoint_path}")
    print(f"Training history saved to: {history_path}")


def main():
    config = load_config()
    _train(config, config["model"], config["training"], "manifest.csv", "eye_model.keras", "eye_training_history.csv")


def main_yawn():
    config = load_config()
    _train(config, config["yawn_model"], config["yawn_training"], "yawn_manifest.csv", "yawn_model.keras", "yawn_training_history.csv")


if __name__ == "__main__":
    main()
