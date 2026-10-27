"""
Train the same three classifiers on the fully preprocessed feature splits.

Inputs:
    data/processed/train_preprocessed.csv
    data/processed/validation_preprocessed.csv
    data/processed/test_preprocessed.csv

Outputs:
    results/models/preprocessed_logistic_regression.joblib
    results/models/preprocessed_random_forest.joblib
    results/models/preprocessed_xgboost.joblib
    results/metrics/preprocessed_model_validation_metrics.csv
    results/metrics/preprocessed_model_training_summary.txt

The preprocessing was fitted by 06_preprocess_data.py on the training split only.
This script trains on transformed data and evaluates on validation data only.
"""

from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "results" / "models"
METRICS_DIR = ROOT_DIR / "results" / "metrics"

INPUT_FILES = {
    "train": PROCESSED_DIR / "train_preprocessed.csv",
    "validation": PROCESSED_DIR / "validation_preprocessed.csv",
    "test": PROCESSED_DIR / "test_preprocessed.csv",
}

TARGET_COLUMN = "Success"
RANDOM_SEED = 42


def load_splits():
    missing_files = [str(path) for path in INPUT_FILES.values() if not path.exists()]
    if missing_files:
        print("ERROR: Missing preprocessed split file(s):")
        for path in missing_files:
            print(f" - {path}")
        print("Run src/06_preprocess_data.py first.")
        sys.exit(1)

    frames = {
        name: pd.read_csv(path, index_col=False, low_memory=False)
        for name, path in INPUT_FILES.items()
    }

    for name, frame in frames.items():
        if TARGET_COLUMN not in frame.columns:
            print(f"ERROR: '{TARGET_COLUMN}' was not found in {name} split.")
            sys.exit(1)
        if frame[TARGET_COLUMN].isna().any():
            print(f"ERROR: Missing target values found in {name} split.")
            sys.exit(1)

    train_columns = list(frames["train"].columns)
    for name, frame in frames.items():
        if list(frame.columns) != train_columns:
            print(f"ERROR: Column mismatch between train and {name} preprocessed files.")
            sys.exit(1)

    return frames


def get_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("ERROR: XGBoost is not installed. Install it with: pip install xgboost")
        sys.exit(1)

    models["XGBoost"] = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
    )
    return models


def evaluate_model(model, x, y):
    predictions = model.predict(x)
    probabilities = model.predict_proba(x)[:, 1]
    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    frames = load_splits()
    feature_columns = [column for column in frames["train"].columns if column != TARGET_COLUMN]

    train_x = frames["train"][feature_columns].copy()
    train_y = frames["train"][TARGET_COLUMN].astype("int8")
    validation_x = frames["validation"][feature_columns].copy()
    validation_y = frames["validation"][TARGET_COLUMN].astype("int8")

    # XGBoost rejects feature names containing [, ], or <. Normalize all names
    # consistently before fitting and save the mapping with each model artifact.
    original_feature_columns = list(feature_columns)
    safe_feature_columns = []
    used_names = set()
    for index, name in enumerate(original_feature_columns):
        safe = str(name).replace("[", "_").replace("]", "_").replace("<", "_")
        safe = safe.replace(">", "_")
        base = safe or f"feature_{index}"
        candidate = base
        suffix = 1
        while candidate in used_names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used_names.add(candidate)
        safe_feature_columns.append(candidate)

    train_x.columns = safe_feature_columns
    validation_x.columns = safe_feature_columns

    all_metrics = []
    summary_lines = [
        "STEAM GAMES SUCCESS PREDICTION — PREPROCESSED MODEL TRAINING",
        "=" * 68,
        "Input data: train_preprocessed.csv and validation_preprocessed.csv.",
        "Preprocessing was fitted on training data only by 06_preprocess_data.py.",
        f"Number of transformed predictor columns: {len(feature_columns)}",
        "Feature names are sanitized for XGBoost compatibility; values are unchanged.",
        f"Random seed: {RANDOM_SEED}",
        "",
        "VALIDATION METRICS",
        "-" * 68,
    ]

    models = get_models()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        for model_name, model in models.items():
            print(f"Training {model_name} on preprocessed features...")
            model.fit(train_x, train_y)
            metrics = evaluate_model(model, validation_x, validation_y)
            all_metrics.append({
                "experiment": "preprocessed_features",
                "model": model_name,
                **metrics,
            })

            safe_name = model_name.lower().replace(" ", "_")
            model_path = MODELS_DIR / f"preprocessed_{safe_name}.joblib"
            joblib.dump({
                "model": model,
                "feature_columns": safe_feature_columns,
                "original_feature_columns": original_feature_columns,
                "experiment": "preprocessed_features",
            }, model_path)

            summary_lines.append(
                f"{model_name}: Accuracy={metrics['accuracy']:.4f}, "
                f"Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}, "
                f"F1={metrics['f1']:.4f}, "
                f"ROC-AUC={metrics['roc_auc']:.4f}"
            )
            print(
                f"  Accuracy={metrics['accuracy']:.4f}, "
                f"Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}, "
                f"F1={metrics['f1']:.4f}, "
                f"ROC-AUC={metrics['roc_auc']:.4f}"
            )

    metrics_file = METRICS_DIR / "preprocessed_model_validation_metrics.csv"
    pd.DataFrame(all_metrics).to_csv(metrics_file, index=False)

    summary_lines.extend([
        "",
        "NOTES",
        "-" * 68,
        "- Validation data was used for evaluation; test data was not used here.",
        "- The same train/validation/test row splits are used as in the raw-feature experiment.",
        "- Compare these metrics with raw_model_validation_metrics.csv.",
        f"Metrics CSV: {metrics_file.relative_to(ROOT_DIR)}",
    ])

    summary_file = METRICS_DIR / "preprocessed_model_training_summary.txt"
    summary_file.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\nPreprocessed-feature model training completed.")
    print(f"Metrics saved to: {metrics_file}")
    print(f"Summary saved to: {summary_file}")
    print(f"Models saved in: {MODELS_DIR}")


if __name__ == "__main__":
    main()
