
"""Evaluate before-launch models on the untouched test set."""

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
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data" / "processed" / "prelaunch"
MODELS_DIR = ROOT_DIR / "results" / "models" / "prelaunch"

METRICS_DIR = ROOT_DIR / "results" / "metrics" / "prelaunch"
FIGURES_DIR = ROOT_DIR / "results" / "figures" / "prelaunch"

TARGET_COLUMN = "Success"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load test dataset
# --------------------------------------------------

test_path = DATA_DIR / "test_preprocessed.csv"

if not test_path.exists():
    raise FileNotFoundError(f"Test dataset not found: {test_path}")

test_df = pd.read_csv(test_path, low_memory=False)

if TARGET_COLUMN not in test_df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' is missing from the test data."
    )

if test_df[TARGET_COLUMN].isna().any():
    raise ValueError("The test dataset contains missing target values.")

y_test = test_df[TARGET_COLUMN].astype("int8")

if y_test.nunique() != 2:
    raise ValueError("Expected both target classes, 0 and 1, in the test set.")

print("Pre-launch test-set evaluation")
print("-" * 50)
print(f"Test rows: {len(test_df):,}")
print(f"Successful games: {(y_test == 1).sum():,}")
print(f"Unsuccessful games: {(y_test == 0).sum():,}")


# --------------------------------------------------
# 3. Model filenames
# --------------------------------------------------

model_files = {
    "Logistic Regression": "prelaunch_logistic_regression.joblib",
    "Random Forest": "prelaunch_random_forest.joblib",
    "XGBoost": "prelaunch_xgboost.joblib",
}


# --------------------------------------------------
# 4. Evaluate each model
# --------------------------------------------------

results = []

summary_lines = [
    "STEAM GAMES SUCCESS PREDICTION",
    "BEFORE-LAUNCH TEST SET EVALUATION",
    "=" * 64,
    f"Test rows: {len(test_df):,}",
    f"Success cases: {(y_test == 1).sum():,}",
    f"Not Success cases: {(y_test == 0).sum():,}",
    "",
]

for model_name, filename in model_files.items():

    model_path = MODELS_DIR / filename

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"\nEvaluating {model_name}...")

    # Load the dictionary saved by src/11_train_prelaunch_models.py.
    saved_artifact = joblib.load(model_path)

    if not isinstance(saved_artifact, dict):
        raise TypeError(
            f"Expected a dictionary in {model_path}, "
            f"but found {type(saved_artifact).__name__}."
        )

    required_keys = {"model", "feature_columns", "experiment"}

    if not required_keys.issubset(saved_artifact.keys()):
        raise ValueError(
            f"Unexpected model artifact structure in {model_path}. "
            f"Expected keys: {required_keys}"
        )

    if saved_artifact["experiment"] != "prelaunch":
        raise ValueError(
            f"{model_name} is not marked as a pre-launch model."
        )

    # Extract the actual trained estimator.
    model = saved_artifact["model"]
    feature_columns = saved_artifact["feature_columns"]

    # Ensure all expected features exist in the test set.
    missing_features = [
        col for col in feature_columns
        if col not in test_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing test features for {model_name}: {missing_features}"
        )

    # Use the exact feature names and order from training.
    X_test = test_df[feature_columns].copy()

    print(f"  Loaded estimator: {type(model).__name__}")
    print(f"  Predictor columns: {X_test.shape[1]}")

    # Generate predictions and positive-class scores.
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        raise ValueError(
            f"{model_name} cannot calculate ROC-AUC scores."
        )

    # Calculate metrics.
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, y_score)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
    })

    print(f"  Accuracy  = {accuracy:.4f}")
    print(f"  Precision = {precision:.4f}")
    print(f"  Recall    = {recall:.4f}")
    print(f"  F1 Score  = {f1:.4f}")
    print(f"  ROC-AUC   = {roc_auc:.4f}")

    # --------------------------------------------------
    # 5. Save confusion matrix
    # --------------------------------------------------

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

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
        f"{model_name} - Pre-launch Test Confusion Matrix"
    )

    fig.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")

    figure_path = (
        FIGURES_DIR
        / f"{safe_name}_prelaunch_test_confusion_matrix.png"
    )

    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------
    # 6. Save detailed classification report
    # --------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Not Success", "Success"],
        zero_division=0,
    )

    summary_lines.extend([
        f"MODEL: {model_name}",
        "-" * 64,
        f"Accuracy:  {accuracy:.4f}",
        f"Precision: {precision:.4f}",
        f"Recall:    {recall:.4f}",
        f"F1 Score:  {f1:.4f}",
        f"ROC-AUC:   {roc_auc:.4f}",
        "",
        "Confusion matrix (rows = actual, columns = predicted):",
        str(cm),
        "",
        "Classification report:",
        report,
        "",
        f"Confusion matrix image: {figure_path}",
        "",
    ])


# --------------------------------------------------
# 7. Save comparison metrics
# --------------------------------------------------

results_df = pd.DataFrame(results)

metrics_path = METRICS_DIR / "prelaunch_test_set_metrics.csv"
results_df.to_csv(metrics_path, index=False)

ranked_results = results_df.sort_values(
    by="ROC-AUC",
    ascending=False,
)

summary_lines.extend([
    "=" * 64,
    "MODEL COMPARISON - RANKED BY ROC-AUC",
    "=" * 64,
    ranked_results.to_string(index=False),
    "",
    "The test set was used only for final evaluation.",
    "Use validation results for model selection and threshold tuning.",
])

summary_path = (
    METRICS_DIR / "prelaunch_test_set_evaluation_summary.txt"
)

summary_path.write_text(
    "\n".join(summary_lines) + "\n",
    encoding="utf-8",
)


# --------------------------------------------------
# 8. Display final results
# --------------------------------------------------

print("\n" + "=" * 75)
print("PRE-LAUNCH TEST SET RESULTS")
print("=" * 75)

print(
    ranked_results.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1 Score": "{:.4f}".format,
            "ROC-AUC": "{:.4f}".format,
        },
    )
)

print("\nEvaluation completed successfully.")
print(f"Metrics saved to: {metrics_path}")
print(f"Summary saved to: {summary_path}")
print(f"Confusion matrices saved in: {FIGURES_DIR}")
