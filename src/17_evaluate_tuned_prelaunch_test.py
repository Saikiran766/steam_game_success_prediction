
"""Evaluate validation-selected thresholds on the untouched test set."""

from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data" / "processed" / "prelaunch"
MODEL_DIR = ROOT_DIR / "results" / "models" / "prelaunch"

OUTPUT_DIR = ROOT_DIR / "results" / "metrics" / "prelaunch"
FIGURES_DIR = ROOT_DIR / "results" / "figures" / "prelaunch"

TARGET_COLUMN = "Success"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# These thresholds were selected using validation results.
# Do not change them based on test-set performance.
SELECTED_THRESHOLDS = {
    "Logistic Regression": 0.40,
    "Random Forest": 0.50,
    "XGBoost": 0.24,
}

MODEL_FILES = {
    "Logistic Regression": "prelaunch_logistic_regression.joblib",
    "Random Forest": "prelaunch_random_forest.joblib",
    "XGBoost": "prelaunch_xgboost.joblib",
}


# --------------------------------------------------
# 2. Load the untouched test set
# --------------------------------------------------

test_path = DATA_DIR / "test_preprocessed.csv"

if not test_path.exists():
    raise FileNotFoundError(
        f"Test dataset not found: {test_path}"
    )

test_df = pd.read_csv(test_path, low_memory=False)

if TARGET_COLUMN not in test_df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' is missing."
    )

if test_df[TARGET_COLUMN].isna().any():
    raise ValueError("Test data contains missing target values.")

y_test = test_df[TARGET_COLUMN].astype("int8")

if y_test.nunique() != 2:
    raise ValueError(
        "Test data must contain both target classes, 0 and 1."
    )

print("Tuned pre-launch test evaluation")
print("=" * 65)
print(f"Test rows: {len(test_df):,}")
print(f"Successful games: {(y_test == 1).sum():,}")
print(f"Unsuccessful games: {(y_test == 0).sum():,}")
print("\nThresholds selected before test evaluation:")

for model_name, threshold in SELECTED_THRESHOLDS.items():
    print(f"  {model_name}: {threshold:.2f}")


# --------------------------------------------------
# 3. Evaluate each model at its selected threshold
# --------------------------------------------------

results = []
summary_lines = [
    "TUNED PRE-LAUNCH TEST SET EVALUATION",
    "=" * 70,
    f"Test rows: {len(test_df):,}",
    "Thresholds selected using validation data only.",
    "",
]

for model_name, filename in MODEL_FILES.items():

    model_path = MODEL_DIR / filename

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    print(f"\nEvaluating {model_name}...")

    # Models were saved as dictionaries by the training script.
    artifact = joblib.load(model_path)

    if not isinstance(artifact, dict):
        raise TypeError(
            f"Expected a dictionary model artifact: {model_path}"
        )

    if artifact.get("experiment") != "prelaunch":
        raise ValueError(
            f"Unexpected experiment type for {model_name}."
        )

    if "model" not in artifact or "feature_columns" not in artifact:
        raise ValueError(
            f"Model artifact is missing required entries: {model_path}"
        )

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    missing_columns = [
        col for col in feature_columns
        if col not in test_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing test features for {model_name}: {missing_columns}"
        )

    # Preserve the exact feature order used during training.
    X_test = test_df[feature_columns].copy()

    # Obtain probabilities, then apply the validation-selected threshold.
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            f"{model_name} does not support predict_proba."
        )

    probabilities = model.predict_proba(X_test)[:, 1]

    threshold = SELECTED_THRESHOLDS[model_name]

    predictions = (probabilities >= threshold).astype("int8")

    # Calculate metrics at the selected threshold.
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test, predictions
    )

    # ROC-AUC uses probabilities, so it does not depend
    # on the selected classification threshold.
    roc_auc = roc_auc_score(y_test, probabilities)

    cm = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    results.append({
        "Model": model_name,
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Balanced Accuracy": balanced_accuracy,
        "ROC-AUC": roc_auc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Predicted Success Count": int(predictions.sum()),
    })

    print(f"  Threshold         = {threshold:.2f}")
    print(f"  Accuracy          = {accuracy:.4f}")
    print(f"  Precision         = {precision:.4f}")
    print(f"  Recall             = {recall:.4f}")
    print(f"  F1 Score           = {f1:.4f}")
    print(f"  Balanced Accuracy  = {balanced_accuracy:.4f}")
    print(f"  ROC-AUC            = {roc_auc:.4f}")
    print(f"  TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}")

    # --------------------------------------------------
    # 4. Save confusion matrix
    # --------------------------------------------------

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Success", "Success"],
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    display.plot(
        ax=ax,
        values_format="d",
        colorbar=False,
    )

    ax.set_title(
        f"{model_name}\nTest Confusion Matrix (threshold={threshold:.2f})"
    )

    fig.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")

    figure_path = (
        FIGURES_DIR
        / f"{safe_name}_tuned_prelaunch_test_confusion_matrix.png"
    )

    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------
    # 5. Add detailed results to summary
    # --------------------------------------------------

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Not Success", "Success"],
        zero_division=0,
    )

    summary_lines.extend([
        f"MODEL: {model_name}",
        "-" * 70,
        f"Selected threshold: {threshold:.2f}",
        f"Accuracy: {accuracy:.4f}",
        f"Precision: {precision:.4f}",
        f"Recall: {recall:.4f}",
        f"F1 Score: {f1:.4f}",
        f"Balanced Accuracy: {balanced_accuracy:.4f}",
        f"ROC-AUC: {roc_auc:.4f}",
        f"TN={tn}, FP={fp}, FN={fn}, TP={tp}",
        "",
        "Classification report:",
        report,
        "",
        f"Confusion matrix image: {figure_path}",
        "",
    ])


# --------------------------------------------------
# 6. Save metrics and comparison
# --------------------------------------------------

results_df = pd.DataFrame(results)

metrics_path = (
    OUTPUT_DIR / "prelaunch_tuned_test_metrics.csv"
)

results_df.to_csv(metrics_path, index=False)

# Rank by F1-score for comparison. This ranking is descriptive;
# do not use it to change the already selected thresholds.
ranked_results = results_df.sort_values(
    by="F1 Score",
    ascending=False,
)

summary_lines.extend([
    "=" * 70,
    "TEST RESULTS AT VALIDATION-SELECTED THRESHOLDS",
    "=" * 70,
    ranked_results.to_string(index=False),
    "",
    "Interpretation notes:",
    "- Thresholds were selected before evaluating the test set.",
    "- Precision, recall, F1 and accuracy depend on the threshold.",
    "- ROC-AUC evaluates probability ranking and is threshold-independent.",
    "- Do not select new thresholds using these test results.",
])

summary_path = (
    OUTPUT_DIR / "prelaunch_tuned_test_evaluation_summary.txt"
)

summary_path.write_text(
    "\n".join(summary_lines) + "\n",
    encoding="utf-8",
)


# --------------------------------------------------
# 7. Print final comparison
# --------------------------------------------------

print("\n" + "=" * 100)
print("FINAL TUNED PRE-LAUNCH TEST RESULTS")
print("=" * 100)

print(
    ranked_results.to_string(
        index=False,
        formatters={
            "Threshold": "{:.2f}".format,
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1 Score": "{:.4f}".format,
            "Balanced Accuracy": "{:.4f}".format,
            "ROC-AUC": "{:.4f}".format,
        },
    )
)

print("\nEvaluation completed successfully.")
print(f"Metrics saved to: {metrics_path}")
print(f"Summary saved to: {summary_path}")
print(f"Confusion matrices saved in: {FIGURES_DIR}")
