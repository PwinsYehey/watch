import sys

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from watch.config import PROJECT_ROOT, load_config

OLD_CLASSES = ["eyes_closed", "eyes_opened"]


def prepare(path, size):
    image = cv2.imread(str(PROJECT_ROOT / path))
    image = cv2.resize(image, size)
    image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
    return image.astype(np.float32) / 255.0


def predict(model, paths, size, batch=512):
    out = []
    for i in range(0, len(paths), batch):
        out.append(model.predict(np.stack([prepare(p, size) for p in paths[i : i + batch]]), verbose=0))
    return np.concatenate(out)


if __name__ == "__main__":
    old_path, new_path = sys.argv[1], sys.argv[2]
    config = load_config()
    classes = config["models"]["eye"]["class_names"]
    size = tuple(config["image_size"])
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "eye_manifest.csv", dtype={"subject_id": str})
    test = manifest[manifest.split == "test"].reset_index(drop=True)
    old = predict(tf.keras.models.load_model(old_path), list(test.filepath), size)
    new = predict(tf.keras.models.load_model(new_path), list(test.filepath), size)
    test["old"] = [OLD_CLASSES[i] for i in old.argmax(1)]
    test["new"] = [classes[i] for i in new.argmax(1)]
    pd.set_option("display.width", 200)
    rows = []
    for (dataset, label), g in test.groupby(["dataset", "label"]):
        row = {"dataset": dataset, "true": label, "n": len(g), "new_correct_%": 100 * (g.new == label).mean(), "new_says_blocked_%": 100 * (g.new == "blocked").mean()}
        row["old_correct_%"] = 100 * (g.old == label).mean() if label != "blocked" else np.nan
        row["old_says_closed_%"] = 100 * (g.old == "eyes_closed").mean()
        rows.append(row)
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    seen = test[test.label != "blocked"]
    print(f"\nOpen/closed accuracy on test people: old {100 * (seen.old == seen.label).mean():.2f}%, new {100 * (seen.new == seen.label).mean():.2f}% (new wrongly says blocked on {100 * (seen.new == 'blocked').mean():.2f}%)")
    blocked = test[test.label == "blocked"]
    print(f"Blocked eyes on test people: new catches {100 * (blocked.new == 'blocked').mean():.1f}%; old called them closed {100 * (blocked.old == 'eyes_closed').mean():.1f}% of the time")
