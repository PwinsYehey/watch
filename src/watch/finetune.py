"""Fine-tunes the trained eye model on crops recorded with `watch record`, such as eyes behind tinted lenses under IR."""

import numpy as np
import pandas as pd
import tensorflow as tf

from watch.config import load_config, model_path, results_dir
from watch.data.dataset import make_dataset
from watch.data.manifest import build_own_eye
from watch.model import load_model

NAME = "eye"


def split_takes(rows: pd.DataFrame, test_fraction: float, seed: int, test_takes: list[str] | None = None) -> pd.DataFrame:
    """Marks whole takes as train or test, so test frames never come from a take the model trained on."""
    takes = sorted(rows["take_id"].unique())
    if len(takes) < 2:
        raise ValueError(f"Only {len(takes)} take recorded, record at least 2 so one can be held out for testing.")
    if test_takes:
        missing = set(test_takes) - set(takes)
        if missing or set(test_takes) == set(takes):
            raise ValueError(f"test_takes must name recorded takes and leave some for training, got {sorted(test_takes)}.")
        test = set(test_takes)
    else:
        rng = np.random.default_rng(seed)
        rng.shuffle(takes)
        test = set(takes[: max(1, round(len(takes) * test_fraction))])
    rows = rows.copy()
    rows["split"] = rows["take_id"].map(lambda take: "test" if take in test else "train")
    return rows


def score(model, rows: pd.DataFrame) -> dict[str, float]:
    """Accuracy, and recall on each class, which shows whether closed eyes are caught as well as open ones."""
    class_names = load_config()["models"][NAME]["class_names"]
    predicted = model.predict(make_dataset(rows, NAME), verbose=0).argmax(axis=1)
    actual = rows["label"].map(class_names.index).to_numpy()
    result = {"accuracy": float((predicted == actual).mean())}
    for index, name in enumerate(class_names):
        mask = actual == index
        result[f"{name}_recall"] = float((predicted[mask] == index).mean()) if mask.any() else float("nan")
    return result


def _weights(rows: pd.DataFrame) -> dict[int, float]:
    class_names = load_config()["models"][NAME]["class_names"]
    counts = rows["label"].map(class_names.index).value_counts().reindex(range(len(class_names)), fill_value=0)
    return {i: len(rows) / (len(class_names) * max(count, 1)) for i, count in counts.items()}


def _report(label: str, result: dict[str, float]):
    print(f"  {label}: " + ", ".join(f"{key} {value:.3f}" for key, value in result.items()))


def finetune(force: bool = False) -> None:
    """Fine-tunes on the training takes and replaces the eye model only if it scores higher on the held-out takes."""
    config = load_config()
    settings = config["finetune"]
    tf.keras.utils.set_random_seed(config["seed"])
    rows = build_own_eye(config["paths"]["raw_dir"] / config["record"]["dir"])
    if rows.empty:
        raise FileNotFoundError("No recorded crops yet, run `watch record` first.")
    rows = split_takes(rows, settings["test_fraction"], config["seed"], settings.get("test_takes"))
    train_rows, test_rows = rows[rows["split"] == "train"], rows[rows["split"] == "test"]
    val_rows = train_rows.sample(frac=0.15, random_state=config["seed"])
    fit_rows = train_rows.drop(val_rows.index)
    print(f"[{NAME}] {len(fit_rows)} train, {len(val_rows)} val, {len(test_rows)} test crops; test takes: {sorted(test_rows['take_id'].unique())}")

    current_model = load_model(NAME)
    before = score(current_model, test_rows)
    per_take_before = {take: score(current_model, part) for take, part in test_rows.groupby("take_id")}
    # Each run starts from the original model, so runs compare fairly instead of stacking fine-tunes.
    base = model_path(NAME).with_name(f"{NAME}_model.base.keras")
    model = load_model(NAME, base if base.exists() else None)
    # Only the dense head learns: a few takes are too narrow to retrain the convolutions without forgetting.
    for layer in model.layers:
        layer.trainable = not settings["train_head_only"] or isinstance(layer, tf.keras.layers.Dense)
    model.compile(optimizer=tf.keras.optimizers.Adam(settings["learning_rate"]), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        make_dataset(fit_rows, NAME, training=True),
        validation_data=make_dataset(val_rows, NAME),
        epochs=settings["epochs"],
        class_weight=_weights(fit_rows),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=settings["early_stopping_patience"], restore_best_weights=True)],
        verbose=2,
    )
    after = score(model, test_rows)
    print(f"[{NAME}] held-out takes, current model then fine-tuned")
    _report("before", before)
    _report("after ", after)
    for take, part in test_rows.groupby("take_id"):
        print(f"[{NAME}] {take}")
        _report("before", per_take_before[take])
        _report("after ", score(model, part))

    candidate = results_dir(NAME) / "finetuned.keras"
    model.save(candidate)
    if force or after["accuracy"] > before["accuracy"]:
        current = model_path(NAME)
        if not base.exists():
            current.replace(base)
            print(f"[{NAME}] kept the original model at {base}")
        model.save(current)
        print(f"[{NAME}] saved the fine-tuned model to {current}")
    else:
        print(f"[{NAME}] kept the current model, fine-tuned candidate left at {candidate}")
