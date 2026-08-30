import json
from pathlib import Path

import tensorflow as tf


def _to_legacy_keras_json(node):
    """TF.js's model loader expects Keras 2's JSON schema, but TF 2.18's model.to_json() emits Keras 3's -- translate the parts that differ."""
    if isinstance(node, dict):
        node = {k: _to_legacy_keras_json(v) for k, v in node.items() if k not in ("module", "registered_name")}
        if "dtype" in node and isinstance(node["dtype"], dict):
            node["dtype"] = node["dtype"]["config"]["name"]
        if "batch_shape" in node:
            node["batch_input_shape"] = node.pop("batch_shape")
        return node
    if isinstance(node, list):
        return [_to_legacy_keras_json(v) for v in node]
    return node


def _iter_named_weights(layer):
    """Recurses into nested Sequential/Functional sub-models so weights are attributed to their real innermost layer."""
    sublayers = getattr(layer, "layers", None)
    if sublayers:
        for sublayer in sublayers:
            yield from _iter_named_weights(sublayer)
    else:
        for weight in layer.weights:
            yield layer.name, weight


def convert_to_tfjs(model_path: Path, output_dir: Path) -> None:
    """Writes a TensorFlow.js LayersModel (model.json + weight shard) without the heavy tensorflowjs pip package."""
    model = tf.keras.models.load_model(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    weight_entries = []
    buffer = bytearray()
    for layer_name, weight in _iter_named_weights(model):
        # TFJS matches weights to layers by name, so this must be "<layer_name>/<role>" regardless of the raw variable name
        role = weight.name.split("/")[-1].split(":")[0]
        array = weight.numpy().astype("float32")
        buffer.extend(array.tobytes())
        weight_entries.append({"name": f"{layer_name}/{role}", "shape": list(array.shape), "dtype": "float32"})

    (output_dir / "group1-shard1of1.bin").write_bytes(bytes(buffer))

    manifest = {
        "format": "layers-model",
        "generatedBy": f"keras v{tf.keras.__version__}",
        "convertedBy": "frames/convert_web.py",
        "modelTopology": _to_legacy_keras_json(json.loads(model.to_json())),
        "weightsManifest": [{"paths": ["group1-shard1of1.bin"], "weights": weight_entries}],
    }
    (output_dir / "model.json").write_text(json.dumps(manifest))
    print(f"Converted {model_path} -> {output_dir}")


def main():
    from frames.config import PROJECT_ROOT, load_config

    config = load_config()
    models_dir = Path(config["paths"]["models_dir"])
    web_models_dir = PROJECT_ROOT / "web" / "models"

    convert_to_tfjs(models_dir / "best_eye_model.keras", web_models_dir / "eye")
    convert_to_tfjs(models_dir / "best_yawn_model.keras", web_models_dir / "yawn")


if __name__ == "__main__":
    main()
