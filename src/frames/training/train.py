from pathlib import Path

import pandas as pd
import tensorflow as tf

from frames.config import load_config
from frames.data.generators import make_generators
from frames.models.build_model import build_model


def main():
    config = load_config()
    model_cfg = config["model"]
    train_cfg = config["training"]
    image_size = tuple(config["preprocessing"]["image_size"])

    train_generator, val_generator, _ = make_generators(config)

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
    checkpoint_path = models_dir / "best_model.keras"

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
    pd.DataFrame(history.history).to_csv(results_dir / "training_history.csv", index=False)

    print(f"\nBest model saved to: {checkpoint_path}")
    print(f"Training history saved to: {results_dir / 'training_history.csv'}")


if __name__ == "__main__":
    main()
