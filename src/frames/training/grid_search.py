import gc
from itertools import product
from pathlib import Path

import pandas as pd
import tensorflow as tf

from frames.config import load_config
from frames.data.generators import make_generators
from frames.models.build_model import build_model


def _already_done(results: list[dict], architecture, dropout, dense_units, lr) -> bool:
    return any(
        r["architecture"] == architecture
        and float(r["dropout"]) == float(dropout)
        and int(r["dense_units"]) == int(dense_units)
        and float(r["learning_rate"]) == float(lr)
        for r in results
    )


def main():
    config = load_config()
    gs_cfg = config["grid_search"]
    model_cfg = config["model"]
    image_size = tuple(config["preprocessing"]["image_size"])

    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    results_csv = results_dir / "gridsearch_results.csv"

    if results_csv.exists():
        results = pd.read_csv(results_csv).to_dict("records")
        print(f"Loaded {len(results)} previously completed run(s)")
    else:
        results = []

    train_generator, val_generator, _ = make_generators(config)

    combinations = list(
        product(
            gs_cfg["architectures"],
            gs_cfg["dropout_rates"],
            gs_cfg["dense_units"],
            gs_cfg["learning_rates"],
        )
    )

    for architecture, dropout, dense_units, lr in combinations:
        if _already_done(results, architecture, dropout, dense_units, lr):
            print(f"Skipping completed: {architecture} dropout={dropout} units={dense_units} lr={lr}")
            continue

        print(f"\n{'=' * 70}\nTraining: {architecture} dropout={dropout} units={dense_units} lr={lr}\n{'=' * 70}")

        tf.keras.backend.clear_session()
        gc.collect()

        model, early_stop, reduce_lr = build_model(
            architecture=architecture,
            input_shape=image_size + (3,),
            dropout_rate=dropout,
            dense_units=dense_units,
            learning_rate=lr,
            num_classes=model_cfg["num_classes"],
        )

        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=gs_cfg["epochs_per_run"],
            callbacks=[early_stop, reduce_lr],
            verbose=1,
        )

        results.append(
            {
                "architecture": architecture,
                "dropout": dropout,
                "dense_units": dense_units,
                "learning_rate": lr,
                "best_val_accuracy": max(history.history["val_accuracy"]),
                "best_val_loss": min(history.history["val_loss"]),
                "best_val_precision": max(history.history.get("val_precision", [0])),
                "best_val_recall": max(history.history.get("val_recall", [0])),
                "best_val_auc": max(history.history.get("val_auc", [0])),
            }
        )

        pd.DataFrame(results).to_csv(results_csv, index=False)
        print(f"Progress saved to: {results_csv}")

        del model
        tf.keras.backend.clear_session()
        gc.collect()

    print("\nGrid search complete.")
    print(pd.DataFrame(results).sort_values("best_val_accuracy", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
