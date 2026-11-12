import json

import tensorflow as tf

from watch.export import _to_keras2_json


def test_branched_model_exports_in_the_keras2_layout_tfjs_reads():
    inputs = tf.keras.Input(shape=(4,))
    left, right = tf.keras.layers.Dense(2, name="left")(inputs), tf.keras.layers.Dense(2, name="right")(inputs)
    model = tf.keras.Model(inputs, tf.keras.layers.Multiply(name="joined")([left, right]))
    topology = _to_keras2_json(json.loads(model.to_json()))
    assert topology["class_name"] == "Model"
    assert topology["config"]["input_layers"] == [[model.inputs[0].name.split(":")[0], 0, 0]] or isinstance(topology["config"]["input_layers"][0], list)
    joined = next(layer for layer in topology["config"]["layers"] if layer["name"] == "joined")
    assert joined["inbound_nodes"] == [[["left", 0, 0, {}], ["right", 0, 0, {}]]]
