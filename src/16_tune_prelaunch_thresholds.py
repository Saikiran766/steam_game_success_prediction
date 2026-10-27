
"""Tune classification thresholds using validation data only."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
)


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data" / "processed" / "prelaunch"
MODEL_DIR = ROOT_DIR / "results" / "models" / "prelaunch"
OUTPUT_DIR = ROOT_DIR / "results" / "metrics" / "prelaunch"

TARGET_COLUMN = "Success"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_FILE = DATA_DIR / "validation_preprocessed.csv"

# Thresholds from 0.10 to 0.90 in increments of 0.01.
THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.01), 2)

MODEL_FILES = {
    "Logistic Regression": "prelaunch_logistic_regression.joblib",
    "Random Forest": "prelaunch_random_forest.joblib",
    "XGBoost": "prelaunch_xgboost.joblib",
}


# --------------------------------------------------
# 2. Load validation data
# --------------------------------------------------

if not VALIDATION_FILE.exists():
    raise FileNotFoundError(
        f"Validation file not found: {VALIDATION_FILE}"
    )

validation_df = pd.read_csv(
    VALIDATION_FILE,
    low_memory=False,
)

if TARGET_COLUMN not in validation_df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' is missing."
    )

if validation_df[TARGET_COLUMN].isna().any():
    raise ValueError("Validation targets contain missing values.")

y_validation = validation_df[TARGET_COLUMN].astype("int8")

if y_validation.nunique() != 2:
    raise ValueError("Validation data must contain both classes.")

print("Pre-launch threshold tuning")
print("-" * 60)
print(f"Validation rows: {len(validation_df):,}")


# --------------------------------------------------
# 3. Calculate metrics at each threshold
# --------------------------------------------------

all_results = []

for model_name, filename in MODEL_FILES.items():

    model_path = MODEL_DIR / filename

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    saved_artifact = joblib.load(model_path)

    if not isinstance(saved_artifact, dict):
        raise TypeError(
            f"Expected a dictionary model artifact: {model_path}"
        )

    model = saved_artifact["model"]
    feature_columns = saved_artifact["feature_columns"]

    missing_columns = [
        col for col in feature_columns
        if col not in validation_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing validation features for {model_name}: "
            f"{missing_columns}"
        )

    X_validation = validation_df[feature_columns].copy()

    # Calculate scores once. Thresholds change predictions,
    # not the trained model.
    if hasattr(model, "predict_proba"):
        positive_scores = model.predict_proba(
            X_validation
        )[:, 1]
    else:
        raise ValueError(
            f"{model_name} does not support predict_proba."
        )

    auc = roc_auc_score(y_validation, positive_scores)

    print(f"\nTuning {model_name} (ROC-AUC={auc:.4f})")

    for threshold in THRESHOLDS:

        predictions = (
            positive_scores >= threshold
        ).astype("int8")

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        accuracy = accuracy_score(
            y_validation,
            predictions,
        )

        balanced_acc = balanced_accuracy_score(
            y_validation,
            predictions,
        )

        all_results.append({
            "Model": model_name,
            "Threshold": float(threshold),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Balanced Accuracy": balanced_acc,
            "ROC-AUC": auc,
            "Predicted Success Count": int(predictions.sum()),
        })


# --------------------------------------------------
# 4. Save all threshold results
# --------------------------------------------------

results_df = pd.DataFrame(all_results)

all_results_path = (
    OUTPUT_DIR / "prelaunch_threshold_tuning_all_results.csv"
)

results_df.to_csv(all_results_path, index=False)


# --------------------------------------------------
# 5. Select thresholds using validation results
# --------------------------------------------------
# Primary selection: highest F1-score.
# Tie-breaker: higher recall, then higher precision.

ranked = results_df.sort_values(
    by=["F1 Score", "Recall", "Precision"],
    ascending=[False, False, False],
)

best_f1_rows = (
    ranked.groupby("Model", sort=False)
    .head(1)
    .sort_values("Model")
)

# Also find the best threshold for a recall target of 70%.
# This is an alternative operating point, not a universal rule.
recall_target = 0.70

recall_candidates = results_df[
    results_df["Recall"] >= recall_target
].copy()

if not recall_candidates.empty:
    recall_candidates = recall_candidates.sort_values(
        by=["Precision", "F1 Score"],
        ascending=[False, False],
    )

    best_recall_rows = (
        recall_candidates.groupby("Model", sort=False)
        .head(1)
        .sort_values("Model")
    )
else:
    best_recall_rows = pd.DataFrame(columns=results_df.columns)


# --------------------------------------------------
# 6. Save selected thresholds and summary
# --------------------------------------------------

best_f1_path = (
    OUTPUT_DIR / "prelaunch_best_thresholds_by_f1.csv"
)

best_f1_rows.to_csv(best_f1_path, index=False)

best_recall_path = (
    OUTPUT_DIR / "prelaunch_thresholds_recall_at_least_70.csv"
)

best_recall_rows.to_csv(best_recall_path, index=False)

summary_lines = [
    "PRE-LAUNCH CLASSIFICATION THRESHOLD TUNING",
    "=" * 70,
    f"Validation rows: {len(validation_df):,}",
    f"Threshold search: {THRESHOLDS.min():.2f} to "
    f"{THRESHOLDS.max():.2f}, step 0.01",
    "",
    "BEST THRESHOLD PER MODEL BY F1-SCORE",
    "-" * 70,
    best_f1_rows.to_string(index=False),
    "",
    "ALTERNATIVE: BEST PRECISION WITH RECALL >= 0.70",
    "-" * 70,
]

if best_recall_rows.empty:
    summary_lines.append(
        "No tested threshold achieved recall >= 0.70 for any model."
    )
else:
    summary_lines.append(best_recall_rows.to_string(index=False))

summary_lines.extend([
    "",
    "IMPORTANT:",
    "Thresholds were selected using validation data only.",
    "Do not select thresholds based on test-set performance.",
    "A recall target of 0.70 is an illustrative operating goal; "
    "choose a target that fits the project's intended use.",
])

summary_path = (
    OUTPUT_DIR / "prelaunch_threshold_tuning_summary.txt"
)

summary_path.write_text(
    "\n".join(summary_lines) + "\n",
    encoding="utf-8",
)


# --------------------------------------------------
# 7. Print results
# --------------------------------------------------

print("\n" + "=" * 80)
print("BEST THRESHOLDS BY F1-SCORE")
print("=" * 80)

print(
    best_f1_rows.to_string(
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

print("\n" + "=" * 80)
print("BEST PRECISION WITH RECALL >= 0.70")
print("=" * 80)

if best_recall_rows.empty:
    print("No tested threshold reached recall >= 0.70.")
else:
    print(
        best_recall_rows.to_string(
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

print("\nThreshold tuning completed.")
print(f"All results: {all_results_path}")
print(f"Best F1 thresholds: {best_f1_path}")
print(f"Recall-target results: {best_recall_path}")
print(f"Summary: {summary_path}")
