from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from frames.config import load_config
from frames.data.generators import make_generators
from frames.models.build_model import load_trained_model


def main():
    config = load_config()
    class_names = config["model"]["class_names"]

    _, _, test_generator = make_generators(config)

    model = load_trained_model(config)

    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.asarray(test_generator.classes)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(results_dir / "classification_report.csv")
    print(report_df.to_string())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    positive_index = 1  # class_names[1], e.g. "open"
    y_score = predictions[:, positive_index]
    y_true_binary = (y_true == positive_index).astype(int)
    fpr, tpr, _ = roc_curve(y_true_binary, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({class_names[positive_index]} vs rest)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "roc_curve.png", dpi=300)
    plt.close()

    predictions_df = pd.DataFrame(
        {
            "filepath": test_generator.filepaths,
            "actual": [class_names[i] for i in y_true],
            "predicted": [class_names[i] for i in y_pred],
            "confidence": np.max(predictions, axis=1),
        }
    )
    predictions_df["correct"] = predictions_df["actual"] == predictions_df["predicted"]
    predictions_df.to_csv(results_dir / "test_predictions.csv", index=False)

    print(f"\nTest accuracy: {(y_true == y_pred).mean():.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Misclassified: {(~predictions_df['correct']).sum()} / {len(predictions_df)}")
    print(f"\nAll results saved to: {results_dir}")


if __name__ == "__main__":
    main()
