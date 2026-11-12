import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import HELMET_DIR, ROOT, serif_fonts
import json
import numpy as np
import pandas as pd
from watch.config import load_config
from watch.data.dataset import labels_of, load_datasets
from watch.data.split import load_split
from watch.model import load_model

config = load_config()
out = {}
for name in ("eye", "yawn"):
    spec = config["models"][name]
    classes = spec["class_names"]
    pos, blk, thr = classes.index(spec["positive_class"]), classes.index(spec["unreadable_class"]), spec["unreadable_threshold"]
    probs = load_model(name).predict(load_datasets(name)["test"], verbose=0)
    y = labels_of(name, "test")
    rule = []
    for row in probs:
        if row[blk] >= thr:
            rule.append(blk)
        else:
            r = row.copy(); r[blk] = 0
            rule.append(int(np.argmax(r)))
    rule = np.array(rule)
    rows = load_split(name).query("split == 'test'").reset_index(drop=True)
    rows["true"] = [classes[i] for i in y]; rows["pred"] = [classes[i] for i in rule]
    cm = pd.crosstab(rows.true, rows.pred).reindex(index=classes, columns=classes, fill_value=0)
    visible = rows[rows.true != classes[blk]]
    res = {
        "accuracy_rule": float((rule == y).mean()),
        "accuracy_argmax": float((probs.argmax(1) == y).mean()),
        "visible_accuracy": float((visible.true == visible.pred).mean()),
        "visible_wrongly_blocked": float((visible.pred == classes[blk]).mean()),
        "blocked_caught": float((rows[rows.true == classes[blk]].pred == classes[blk]).mean()),
        "recall": {c: float((rows[rows.true == c].pred == c).mean()) for c in classes},
        "precision": {c: float((rows[rows.pred == c].true == c).mean()) for c in classes},
        "confusion": cm.to_dict(),
        "per_dataset": {f"{d}/{c}": [int(len(g)), float((g.true == g.pred).mean())] for (d, c), g in rows.groupby(["dataset", "true"])},
        "test_people": int(rows.group_id.nunique()),
    }
    out[name] = res
    np.save(f"{ROOT}/results/paper_training/{name}_test_probs.npy", probs)
    rows.to_csv(f"{ROOT}/results/paper_training/{name}_test_rule.csv", index=False)
    print(name, json.dumps({k: v for k, v in res.items() if k not in ("confusion", "per_dataset")}, indent=1))
    print(cm.to_string())
    print(json.dumps(res["per_dataset"], indent=1))
json.dump(out, open(f"{ROOT}/results/paper_training/eval_rule.json", "w"), indent=2)
