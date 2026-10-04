# Perform the final one-time evaluation on the untouched test set.
# Models and decision thresholds were selected using training/validation data only.

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)

PROJECT_ROOT = Path(r"D:\Steam_Games_Success_Prediction")

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

TEST_FILE = DATA_DIR / "test_features.csv"
TUNING_RESULTS_FILE = METRICS_DIR / "tuning_results.json"

RANDOM_SEED = 42
TARGET = "Success"

MODELS = [
    "logistic_regression",
    "random_forest",
    "xgboost",
]

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("Loading the untouched test dataset...")

test = pd.read_csv(TEST_FILE)

X_test = test.drop(columns=[TARGET])
y_test = test[TARGET].astype(int)

print(f"Test rows: {len(test)}")
print(f"Test features: {X_test.shape[1]}")

# Load the validation-selected parameters and thresholds.
with open(TUNING_RESULTS_FILE, "r", encoding="utf-8") as file:
    tuning_results = json.load(file)

final_results = {}
roc_curves = {}

for model_name in MODELS:
    model_file = MODELS_DIR / f"{model_name}_tuned.joblib"

    print(f"\nEvaluating: {model_name}")
    print(f"Loading: {model_file}")

    model = joblib.load(model_file)

    probabilities = model.predict_proba(X_test)[:, 1]

    selected_threshold = float(
        tuning_results[model_name]["selected_threshold"]
    )

    predictions = (
        probabilities >= selected_threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    metrics = {
        "Accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "Precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),
        "Recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),
        "F1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),
        "ROC-AUC": float(
            roc_auc_score(y_test, probabilities)
        ),
        "PR-AUC": float(
            average_precision_score(
                y_test,
                probabilities
            )
        ),
        "True Negative": int(tn),
        "False Positive": int(fp),
        "False Negative": int(fn),
        "True Positive": int(tp),
        "Selected Threshold": selected_threshold,
    }

    final_results[model_name] = metrics

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test,
        probabilities
    )

    roc_curves[model_name] = (
        false_positive_rate,
        true_positive_rate,
        metrics["ROC-AUC"]
    )

    print(f"Threshold: {selected_threshold:.2f}")
    print(f"Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall:    {metrics['Recall']:.4f}")
    print(f"F1:        {metrics['F1']:.4f}")
    print(f"ROC-AUC:   {metrics['ROC-AUC']:.4f}")
    print(f"PR-AUC:    {metrics['PR-AUC']:.4f}")
    print(
        "Confusion matrix: "
        f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
    )

    # Save an individual confusion matrix figure.
    cm = np.array([
        [tn, fp],
        [fn, tp]
    ])

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Success", "Success"]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax)
    ax.set_title(
        f"{model_name.replace('_', ' ').title()} - Test Confusion Matrix"
    )
    fig.tight_layout()

    confusion_file = (
        FIGURES_DIR / f"{model_name}_test_confusion_matrix.png"
    )
    fig.savefig(confusion_file, dpi=200)
    plt.close(fig)

# Save one combined ROC curve figure.
fig, ax = plt.subplots(figsize=(7, 6))

for model_name, (
    false_positive_rate,
    true_positive_rate,
    auc_value
) in roc_curves.items():
    ax.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"{model_name.replace('_', ' ').title()} (AUC={auc_value:.3f})"
    )

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Test ROC Curves")
ax.legend()
fig.tight_layout()

roc_file = FIGURES_DIR / "test_roc_curves.png"
fig.savefig(roc_file, dpi=200)
plt.close(fig)

# Save a compact final comparison table.
comparison_rows = []

for model_name, metrics in final_results.items():
    comparison_rows.append({
        "Model": model_name,
        "Accuracy": metrics["Accuracy"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1": metrics["F1"],
        "ROC-AUC": metrics["ROC-AUC"],
        "PR-AUC": metrics["PR-AUC"],
        "Threshold": metrics["Selected Threshold"],
    })

comparison = pd.DataFrame(comparison_rows)

comparison_file = METRICS_DIR / "final_test_comparison.csv"
comparison.to_csv(comparison_file, index=False)

# Save detailed metrics as JSON.
metrics_file = METRICS_DIR / "final_test_metrics.json"
metrics_file.write_text(
    json.dumps(final_results, indent=2),
    encoding="utf-8"
)

# Save a readable report.
report_lines = [
    "Final Test Evaluation",
    "=====================",
    "",
    "This is the single final evaluation on the held-out test set.",
    "The test set was not used for preprocessing, model fitting,",
    "hyperparameter selection, or threshold selection.",
    "",
    "Thresholds were selected from validation data only.",
    "",
]

for model_name, metrics in final_results.items():
    report_lines.extend([
        model_name,
        "-" * len(model_name),
        f"Threshold: {metrics['Selected Threshold']:.2f}",
        f"Accuracy: {metrics['Accuracy']:.4f}",
        f"Precision: {metrics['Precision']:.4f}",
        f"Recall: {metrics['Recall']:.4f}",
        f"F1: {metrics['F1']:.4f}",
        f"ROC-AUC: {metrics['ROC-AUC']:.4f}",
        f"PR-AUC: {metrics['PR-AUC']:.4f}",
        f"TN: {metrics['True Negative']}",
        f"FP: {metrics['False Positive']}",
        f"FN: {metrics['False Negative']}",
        f"TP: {metrics['True Positive']}",
        "",
    ])

report_lines.extend([
    f"Final comparison: {comparison_file}",
    f"Detailed metrics: {metrics_file}",
    f"ROC curves: {roc_file}",
    "",
    "Confusion matrices:",
])

for model_name in MODELS:
    report_lines.append(
        f"{model_name}: "
        f"{FIGURES_DIR / f'{model_name}_test_confusion_matrix.png'}"
    )

report_file = METRICS_DIR / "final_test_evaluation_summary.txt"
report_file.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print("\nFinal test evaluation completed successfully.")
print("\nFinal test comparison:")
print(comparison.to_string(index=False))

print(f"\nComparison saved to:\n{comparison_file}")
print(f"Detailed metrics saved to:\n{metrics_file}")
print(f"ROC curves saved to:\n{roc_file}")
print(f"Summary saved to:\n{report_file}")
print(f"Confusion matrices saved to:\n{FIGURES_DIR}")
