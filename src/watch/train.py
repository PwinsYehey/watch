import gc
from itertools import product

import pandas as pd
import tensorflow as tf

from watch.config import load_config, model_path, results_dir
from watch.data.dataset import class_weights, load_datasets
from watch.model import build_model, load_model


def _fit(name: str, datasets: dict, architecture: str, dropout_rate: float, dense_units: int, learning_rate: float, epochs: int, checkpoint=None):
    spec = load_config()["models"][name]
    model = build_model(architecture, dropout_rate, dense_units, learning_rate, len(spec["class_names"]))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=spec["early_stopping_patience"], restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=spec["reduce_lr_patience"], min_lr=1e-7, verbose=1),
    ]
    if checkpoint:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss", save_best_only=True))
    history = model.fit(
        datasets["train"], validation_data=datasets["val"], epochs=epochs, callbacks=callbacks, class_weight=class_weights(name), verbose=2
    )
    return model, pd.DataFrame(history.history)


def _val_accuracy(model, dataset) -> float:
    return float(model.evaluate(dataset, return_dict=True, verbose=0)["accuracy"])


def train(name: str, force: bool = False) -> None:
    """Trains a candidate and replaces the current model only if it scores higher on the validation split."""
    config = load_config()
    spec = config["models"][name]
    tf.keras.utils.set_random_seed(config["seed"])
    datasets = load_datasets(name)
    out_dir = results_dir(name)
    candidate_path = out_dir / "candidate.keras"

    model, history = _fit(
        name, datasets, spec["architecture"], spec["dropout_rate"], spec["dense_units"], spec["learning_rate"], spec["epochs"], candidate_path
    )
    history.to_csv(out_dir / "training_history.csv", index=False)

    current_path = model_path(name)
    candidate_score = _val_accuracy(model, datasets["val"])
    current_score = _val_accuracy(load_model(name), datasets["val"]) if current_path.exists() else None
    print(f"[{name}] val accuracy: candidate {candidate_score:.4f}, current {current_score if current_score is None else f'{current_score:.4f}'}")

    if force or current_score is None or candidate_score > current_score:
        candidate_path.replace(current_path)
        print(f"[{name}] saved new model to {current_path}")
    else:
        print(f"[{name}] kept the current model, candidate left at {candidate_path}")


def sweep(name: str) -> None:
    """Resumable grid search over the sweep settings in config.yaml; never touches the saved model."""
    config = load_config()
    grid = config["sweep"]
    datasets = load_datasets(name)
    results_csv = results_dir(name) / "sweep.csv"
    results = pd.read_csv(results_csv).to_dict("records") if results_csv.exists() else []
    done = {(r["architecture"], float(r["dropout"]), int(r["dense_units"]), float(r["learning_rate"])) for r in results}

    for architecture, dropout, units, lr in product(grid["architectures"], grid["dropout_rates"], grid["dense_units"], grid["learning_rates"]):
        if (architecture, float(dropout), int(units), float(lr)) in done:
            continue
        print(f"\n[{name}] {architecture} dropout={dropout} units={units} lr={lr}")
        tf.keras.utils.set_random_seed(config["seed"])
        _, history = _fit(name, datasets, architecture, dropout, units, lr, grid["epochs"])
        results.append(
            {
                "architecture": architecture,
                "dropout": dropout,
                "dense_units": units,
                "learning_rate": lr,
                "best_val_accuracy": history["val_accuracy"].max(),
                "best_val_loss": history["val_loss"].min(),
                "best_val_auc": history["val_auc"].max(),
            }
        )
        pd.DataFrame(results).to_csv(results_csv, index=False)
        tf.keras.backend.clear_session()
        gc.collect()

    print(pd.DataFrame(results).sort_values("best_val_accuracy", ascending=False).to_string(index=False))
