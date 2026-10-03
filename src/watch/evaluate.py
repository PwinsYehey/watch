import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, classification_report, roc_auc_score

from watch.config import load_config, model_path, results_dir
from watch.data.dataset import labels_of, load_datasets
from watch.data.split import load_split
from watch.model import load_model


def evaluate(name: str) -> dict:
    spec = load_config()["models"][name]
    class_names, positive = spec["class_names"], spec["class_names"].index(spec["positive_class"])
    out_dir = results_dir(name)

    probabilities = load_model(name).predict(load_datasets(name)["test"], verbose=0)
    y_pred, y_true = probabilities.argmax(axis=1), labels_of(name, "test")

    print(pd.DataFrame(classification_report(y_true, y_pred, target_names=class_names, output_dict=True)).transpose().round(4).to_string())
    metrics = {
        "accuracy": float((y_pred == y_true).mean()),
        "roc_auc": float(roc_auc_score(y_true == positive, probabilities[:, positive])),
        "misclassified": int((y_pred != y_true).sum()),
        "test_images": int(len(y_true)),
    }

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names, cmap="Blues")
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    RocCurveDisplay.from_predictions(y_true == positive, probabilities[:, positive], name=spec["positive_class"])
    plt.savefig(out_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    test_rows = load_split(name).query("split == 'test'")
    pd.DataFrame(
        {
            "filepath": test_rows["filepath"].to_numpy(),
            "actual": [class_names[i] for i in y_true],
            "predicted": [class_names[i] for i in y_pred],
            "confidence": probabilities.max(axis=1),
        }
    ).to_csv(out_dir / "test_predictions.csv", index=False)

    metrics_path = out_dir / "metrics.json"
    previous = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"\n[{name}] accuracy {metrics['accuracy']:.4f}, ROC-AUC {metrics['roc_auc']:.4f}, misclassified {metrics['misclassified']}/{metrics['test_images']}")
    if previous:
        print(f"[{name}] previous run: accuracy {previous['accuracy']:.4f}, ROC-AUC {previous['roc_auc']:.4f}")
    print(f"[{name}] results saved to {out_dir}")
    return metrics


def benchmark(name: str, runs: int = 100) -> dict:
    model = load_model(name)
    sample = np.random.rand(1, *load_config()["image_size"], 3).astype(np.float32)
    for _ in range(10):
        model(sample, training=False)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        model(sample, training=False)
        times.append((time.perf_counter() - start) * 1000)

    result = {
        "params": model.count_params(),
        "size_mb": round(model_path(name).stat().st_size / 1024**2, 2),
        "ms_mean": round(float(np.mean(times)), 2),
        "ms_std": round(float(np.std(times)), 2),
        "fps": round(1000 / float(np.mean(times)), 1),
    }
    print(f"[{name}] " + ", ".join(f"{k}={v}" for k, v in result.items()))
    pd.DataFrame([result]).to_csv(results_dir(name) / "benchmark.csv", index=False)
    return result
