import sys

import numpy as np
import tensorflow as tf

from watch.config import load_config
from watch.data.dataset import class_weights, load_datasets


def extended(old):
    inputs = tf.keras.Input(shape=old.input_shape[1:])
    x = inputs
    for layer in old.layers[:-1]:
        layer.trainable = False
        x = layer(x)
    w, b = old.layers[-1].get_weights()
    kept = tf.keras.layers.Dense(2, name="yawn_logits", trainable=False)
    kept_logits = kept(x)
    kept.set_weights([w, b])
    hidden = tf.keras.layers.Dense(32, activation="relu", name="blocked_hidden")(x)
    blocked_logit = tf.keras.layers.Dense(1, name="blocked_logit")(hidden)
    outputs = tf.keras.layers.Softmax(name="head3")(tf.keras.layers.Concatenate()([kept_logits, blocked_logit]))
    return tf.keras.Model(inputs, outputs)


if __name__ == "__main__":
    old_path, out_path = sys.argv[1], sys.argv[2]
    config = load_config()
    tf.keras.utils.set_random_seed(config["seed"])
    datasets = load_datasets("yawn")
    model = extended(tf.keras.models.load_model(old_path))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    weights = class_weights("yawn")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6, verbose=1),
    ]
    model.fit(datasets["train"], validation_data=datasets["val"], epochs=40, class_weight=weights, callbacks=callbacks, verbose=2)
    model.save(out_path)
    print("saved", out_path)
