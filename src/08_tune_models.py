# Tune the main models using the validation set while keeping the test set
# completely untouched. Hyperparameter selection is based on ROC-AUC.

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError(
        "XGBoost is not installed. Run: python -m pip install xgboost"
    ) from exc


PROJECT_ROOT = Path(r"D:\Steam_Games_Success_Prediction")

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"

TRAIN_FILE = DATA_DIR / "train_features.csv"
VALIDATION_FILE = DATA_DIR / "validation_features.csv"

RANDOM_SEED = 42
TARGET = "Success"
PRIMARY_METRIC = "ROC-AUC"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading training and validation data...")

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET].astype(int)

X_validation = validation.drop(columns=[TARGET])
y_validation = validation[TARGET].astype(int)

print(f"Training rows: {len(train)}")
print(f"Validation rows: {len(validation)}")
print(f"Features: {X_train.shape[1]}")

# Keep the tuning grids deliberately small so the experiment is reproducible
# and practical on a normal local machine.
param_grids = {
    "logistic_regression": {
        "C": [0.1, 1.0, 10.0],
        "class_weight": [None],
    },
    "random_forest": {
        "n_estimators": [200, 400],
        "max_depth": [None, 15],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt"],
        "class_weight": [None],
    },
    "xgboost": {
        "n_estimators": [200, 400],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    },
}

def create_model(model_name, params):
    # Construct a fresh model for every parameter combination.
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
            **params,
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            **params,
        )

    if model_name == "xgboost":
        return XGBClassifier(
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            n_jobs=-1,
            **params,
        )

    raise ValueError(f"Unknown model: {model_name}")

def get_probabilities(model, X):
    # Use the probability of the Success class for ROC-AUC.
    return model.predict_proba(X)[:, 1]

def evaluate(model, X, y, threshold=0.5):
    probabilities = get_probabilities(model, X)
    predictions = (probabilities >= threshold).astype(int)

    return {
        "Accuracy": float(
            accuracy_score(y, predictions)
        ),
        "Precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "Recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "F1": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "ROC-AUC": float(
            roc_auc_score(y, probabilities)
        ),
    }, probabilities, predictions

def find_best_threshold(y_true, probabilities):
    # Select the threshold using validation data only.
    # F1 is used for threshold selection because threshold does not change
    # ROC-AUC; it provides a practical operating point for binary decisions.
    thresholds = np.linspace(0.10, 0.90, 81)

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


tuning_results = {}
best_models = {}
validation_probabilities = {}
threshold_results = []

for model_name, grid in param_grids.items():
    print(f"\nTuning: {model_name}")

    best_score = -np.inf
    best_params = None
    best_model = None
    combinations = list(ParameterGrid(grid))

    print(f"Parameter combinations: {len(combinations)}")

    for combination_number, params in enumerate(combinations, start=1):
        model = create_model(model_name, params)

        model.fit(X_train, y_train)

        probabilities = get_probabilities(
            model,
            X_validation
        )

        score = roc_auc_score(
            y_validation,
            probabilities
        )

        print(
            f"  [{combination_number}/{len(combinations)}] "
            f"ROC-AUC={score:.4f} | {params}"
        )

        if score > best_score:
            best_score = score
            best_params = params
            best_model = model

    best_models[model_name] = best_model

    metrics, probabilities, predictions = evaluate(
        best_model,
        X_validation,
        y_validation,
        threshold=0.50
    )

    best_threshold, threshold_f1 = find_best_threshold(
        y_validation,
        probabilities
    )

    tuned_threshold_metrics, _, _ = evaluate(
        best_model,
        X_validation,
        y_validation,
        threshold=best_threshold
    )

    tuning_results[model_name] = {
        "best_params": best_params,
        "validation_metrics_at_0.50": metrics,
        "validation_metrics_at_selected_threshold": tuned_threshold_metrics,
        "selected_threshold": best_threshold,
        "threshold_selection_f1": threshold_f1,
        "best_validation_roc_auc": best_score,
    }

    validation_probabilities[model_name] = probabilities

    threshold_results.append({
        "model": model_name,
        "threshold": best_threshold,
        "F1": tuned_threshold_metrics["F1"],
        "Accuracy": tuned_threshold_metrics["Accuracy"],
        "Precision": tuned_threshold_metrics["Precision"],
        "Recall": tuned_threshold_metrics["Recall"],
        "ROC-AUC": tuned_threshold_metrics["ROC-AUC"],
    })

    model_file = MODELS_DIR / f"{model_name}_tuned.joblib"
    joblib.dump(best_model, model_file)

    print(f"Best parameters: {best_params}")
    print(f"Best validation ROC-AUC: {best_score:.4f}")
    print(f"Selected validation threshold: {best_threshold:.2f}")
    print(
        f"Validation F1 at selected threshold: "
        f"{threshold_f1:.4f}"
    )
    print(f"Saved tuned model: {model_file}")

# Save tuning results.
results_file = METRICS_DIR / "tuning_results.json"
results_file.write_text(
    json.dumps(tuning_results, indent=2),
    encoding="utf-8"
)

threshold_file = METRICS_DIR / "validation_threshold_results.csv"
pd.DataFrame(threshold_results).to_csv(
    threshold_file,
    index=False
)

# Create a compact model comparison using the tuned validation results.
comparison_rows = []

for model_name, result in tuning_results.items():
    metrics = result["validation_metrics_at_selected_threshold"]

    comparison_rows.append({
        "Model": model_name,
        "ROC-AUC": metrics["ROC-AUC"],
        "Accuracy": metrics["Accuracy"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1": metrics["F1"],
        "Selected Threshold": result["selected_threshold"],
    })

comparison = pd.DataFrame(comparison_rows)

comparison_file = METRICS_DIR / "tuned_validation_comparison.csv"
comparison.to_csv(
    comparison_file,
    index=False
)

# Save a readable report.
report_lines = [
    "Model Tuning Summary",
    "====================",
    "",
    f"Primary tuning metric: {PRIMARY_METRIC}",
    f"Random seed: {RANDOM_SEED}",
    "",
    "The test set was not loaded or used.",
    "Hyperparameters were selected using validation ROC-AUC.",
    "Decision thresholds were selected using validation data only.",
    "",
]

for model_name, result in tuning_results.items():
    metrics = result["validation_metrics_at_selected_threshold"]

    report_lines.extend([
        model_name,
        "-" * len(model_name),
        f"Best parameters: {result['best_params']}",
        f"Best validation ROC-AUC: {result['best_validation_roc_auc']:.4f}",
        f"Selected threshold: {result['selected_threshold']:.2f}",
        f"Accuracy: {metrics['Accuracy']:.4f}",
        f"Precision: {metrics['Precision']:.4f}",
        f"Recall: {metrics['Recall']:.4f}",
        f"F1: {metrics['F1']:.4f}",
        f"ROC-AUC: {metrics['ROC-AUC']:.4f}",
        "",
    ])

report_lines.extend([
    f"JSON results: {results_file}",
    f"Threshold results: {threshold_file}",
    f"Comparison: {comparison_file}",
])

report_file = METRICS_DIR / "tuning_summary.txt"
report_file.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print("\nTuning completed successfully.")
print("\nTuned validation comparison:")
print(comparison.to_string(index=False))
print(f"\nTuning results saved to:\n{results_file}")
print(f"Threshold results saved to:\n{threshold_file}")
print(f"Comparison saved to:\n{comparison_file}")
print(f"Tuning summary saved to:\n{report_file}")
