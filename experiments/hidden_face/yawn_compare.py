import sys

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from watch.config import PROJECT_ROOT, load_config

OLD_CLASSES = ["no_yawn", "yawn"]


def prepare(path, size):
    image = cv2.cvtColor(cv2.resize(cv2.imread(str(PROJECT_ROOT / path)), size), cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def predict(model, paths, size, batch=512):
    return np.concatenate([model.predict(np.stack([prepare(p, size) for p in paths[i : i + batch]]), verbose=0) for i in range(0, len(paths), batch)])


if __name__ == "__main__":
    config = load_config()
    spec, size = config["models"]["yawn"], tuple(config["image_size"])
    classes, threshold = spec["class_names"], spec["unreadable_threshold"]
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "yawn_manifest.csv", dtype={"subject_id": str})
    test = manifest[manifest.split == "test"].reset_index(drop=True)
    old = predict(tf.keras.models.load_model(sys.argv[1]), list(test.filepath), size)
    new = predict(tf.keras.models.load_model(sys.argv[2]), list(test.filepath), size)
    test["old"] = [OLD_CLASSES[i] for i in old.argmax(1)]
    b = classes.index("blocked")
    rest = new.copy()
    rest[:, b] = 0
    rest = rest / rest.sum(1, keepdims=True)
    test["new"] = np.where(new[:, b] >= threshold, "blocked", np.array(classes)[rest.argmax(1)])
    rows = []
    for label, g in test.groupby("label"):
        rows.append({"true": label, "n": len(g), "new_correct_%": 100 * (g.new == label).mean(), "new_says_blocked_%": 100 * (g.new == "blocked").mean(), "old_says_yawn_%": 100 * (g.old == "yawn").mean()})
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    seen = test[test.label != "blocked"]
    print(f"\nyawn/no_yawn accuracy: old {100 * (seen.old == seen.label).mean():.2f}%, new {100 * (seen.new == seen.label).mean():.2f}% (visible mouths wrongly called blocked: {100 * (seen.new == 'blocked').mean():.2f}%)")
