import sys

import numpy as np
import tensorflow as tf


def fixed_dense(units, weights, bias, name):
    layer = tf.keras.layers.Dense(units, trainable=False, name=name)
    return layer, (np.array(weights, np.float32), np.array(bias, np.float32))


def flat(model, prefix, x):
    for layer in model.layers:
        if getattr(layer, "layers", None):
            x = flat(layer, prefix, x)
            continue
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        copy = layer.__class__.from_config({**layer.get_config(), "name": f"{prefix}_{layer.name}"})
        x = copy(x)
        copy.set_weights(layer.get_weights())
        copy.trainable = False
    return x


if __name__ == "__main__":
    yawn_path, blocked_path, out_path = sys.argv[1:4]
    yawn_model, blocked_model = tf.keras.models.load_model(yawn_path), tf.keras.models.load_model(blocked_path)
    inputs = tf.keras.Input(shape=yawn_model.input_shape[1:])
    yawn_probs = flat(yawn_model, "yawn", inputs)
    three = flat(blocked_model, "blocked", inputs)
    keep_layer, keep_weights = fixed_dense(2, [[0, 0], [0, 0], [-1, -1]], [1, 1], "not_blocked")
    pick_layer, pick_weights = fixed_dense(1, [[0], [0], [1]], [0], "blocked_prob")
    not_blocked, blocked = keep_layer(three), pick_layer(three)
    keep_layer.set_weights(keep_weights)
    pick_layer.set_weights(pick_weights)
    outputs = tf.keras.layers.Concatenate(name="head3")([tf.keras.layers.Multiply()([yawn_probs, not_blocked]), blocked])
    model = tf.keras.Model(inputs, outputs)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
    model.save(out_path)
    probe = np.random.default_rng(0).random((4, *yawn_model.input_shape[1:]), np.float32)
    out, old, new = model.predict(probe, verbose=0), yawn_model.predict(probe, verbose=0), blocked_model.predict(probe, verbose=0)
    assert np.allclose(out[:, 2], new[:, 2], atol=1e-5) and np.allclose(out[:, :2] / out[:, :2].sum(1, keepdims=True), old, atol=1e-5)
    print("saved", out_path, "sum check", np.round(out.sum(1), 4))
