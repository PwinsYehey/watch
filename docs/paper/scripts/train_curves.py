import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import HELMET_DIR, ROOT, serif_fonts
import copy
import json
import sys
import time
from pathlib import Path

import tensorflow as tf

import watch.data.dataset as dataset
import watch.model as model_module
import watch.train as train
from watch.config import load_config

OUT = Path(f"{ROOT}/results/paper_training")
base = load_config()
original_split = dataset.load_split


def run(tag, name, config, keep_labels=None):
    def patched_config():
        return config

    def patched_split(n):
        rows = original_split(n)
        return rows[rows.label.isin(keep_labels)] if keep_labels else rows

    for module in (dataset, model_module, train):
        module.load_config = patched_config
    dataset.load_split = patched_split
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config["seed"])
    spec = config["models"][name]
    start = time.time()
    datasets = dataset.load_datasets(name)
    model, history = train._fit(name, datasets, spec["architecture"], spec["dropout_rate"], spec["dense_units"], spec["learning_rate"], spec["epochs"])
    scores = {split: model.evaluate(datasets[split], return_dict=True, verbose=0) for split in ("val", "test")}
    history.to_csv(OUT / f"{tag}_history.csv", index=False)
    (OUT / f"{tag}_summary.json").write_text(json.dumps({"epochs_run": len(history), "minutes": round((time.time() - start) / 60, 1), "scores": scores, "params": model.count_params()}, indent=2))
    print(tag, "done", len(history), "epochs", scores, flush=True)


eye = copy.deepcopy(base)
run("eye", "eye", eye)

yawn2 = copy.deepcopy(base)
yawn2["models"]["yawn"]["class_names"] = ["no_yawn", "yawn"]
yawn2["models"]["yawn"].pop("unreadable_class", None)
run("yawn_branch", "yawn", yawn2, ["no_yawn", "yawn"])

blocked = copy.deepcopy(base)
run("blocked_branch", "yawn", blocked)
