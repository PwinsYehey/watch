import json
import re
import tempfile
from pathlib import Path

import tensorflow as tf

from watch.config import MODEL_NAMES, load_config, model_path
from watch.model import load_model

WEB_CONFIG_SECTIONS = ("alertness", "yawn", "nod", "ear", "mar", "head_pose", "calibration", "live")


def _camel(key: str) -> str:
    return re.sub(r"_(\w)", lambda m: m.group(1).upper(), key)


def _camelize(value):
    if isinstance(value, dict):
        return {_camel(k): _camelize(v) for k, v in value.items()}
    return value


def web_config_js() -> str:
    config = load_config()
    web_config = {
        "imageSize": config["image_size"],
        "eyeClassNames": config["models"]["eye"]["class_names"],
        "yawnClassNames": config["models"]["yawn"]["class_names"],
        **{_camel(section): _camelize(config[section]) for section in WEB_CONFIG_SECTIONS},
    }
    return f"// Generated from configs/config.yaml by ./scripts/export.sh, do not edit.\nconst CONFIG = {json.dumps(web_config, indent=2)};\n"


def _to_keras2_json(node):
    """TF.js expects the Keras 2 topology schema, so rewrite the parts Keras 3 changed."""
    if isinstance(node, dict):
        node = {k: _to_keras2_json(v) for k, v in node.items() if k not in ("module", "registered_name", "optional")}
        if isinstance(node.get("dtype"), dict):
            node["dtype"] = node["dtype"]["config"]["name"]
        if "batch_shape" in node:
            node["batch_input_shape"] = node.pop("batch_shape")
        return node
    if isinstance(node, list):
        return [_to_keras2_json(v) for v in node]
    return node


def _named_weights(layer):
    for sublayer in getattr(layer, "layers", None) or []:
        yield from _named_weights(sublayer)
    if not getattr(layer, "layers", None):
        for weight in layer.weights:
            yield f"{layer.name}/{weight.name.split('/')[-1].split(':')[0]}", weight.numpy().astype("float32")


def export_tfjs(model: tf.keras.Model, output_dir: Path) -> None:
    """Writes a TF.js LayersModel without the heavy tensorflowjs package."""
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = list(_named_weights(model))
    (output_dir / "group1-shard1of1.bin").write_bytes(b"".join(array.tobytes() for _, array in weights))
    manifest = {
        "format": "layers-model",
        "generatedBy": f"keras v{tf.keras.__version__}",
        "convertedBy": "watch.export",
        "modelTopology": _to_keras2_json(json.loads(model.to_json())),
        "weightsManifest": [
            {"paths": ["group1-shard1of1.bin"], "weights": [{"name": n, "shape": list(a.shape), "dtype": "float32"} for n, a in weights]}
        ],
    }
    (output_dir / "model.json").write_text(json.dumps(manifest))


def export_tflite(model: tf.keras.Model, output_path: Path) -> None:
    with tempfile.TemporaryDirectory() as saved_model_dir:
        model.export(saved_model_dir, verbose=False)
        output_path.write_bytes(tf.lite.TFLiteConverter.from_saved_model(saved_model_dir).convert())


def export_all() -> None:
    web_dir = load_config()["paths"]["web_dir"]
    for name in MODEL_NAMES:
        model = load_model(name)
        export_tfjs(model, web_dir / "models" / name)
        export_tflite(model, model_path(name, ".tflite"))
        print(f"[{name}] exported {web_dir / 'models' / name} and {model_path(name, '.tflite')}")
    (web_dir / "config.js").write_text(web_config_js())
    print(f"Generated {web_dir / 'config.js'}")
