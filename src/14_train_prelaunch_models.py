
"""Train classifiers for the separate before-launch experiment."""

from pathlib import Path
import sys
import warnings

import joblib
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
DATA_DIR = ROOT_DIR / "data" / "processed" / "prelaunch"
MODELS_DIR = ROOT_DIR / "results" / "models" / "prelaunch"
METRICS_DIR = ROOT_DIR / "results" / "metrics"

TARGET_COLUMN = "Success"
RANDOM_SEED = 42

INPUT_FILES = {
    "train": DATA_DIR / "train_preprocessed.csv",
    "validation": DATA_DIR / "validation_preprocessed.csv",
}

METRICS_FILE = METRICS_DIR / "prelaunch_model_validation_metrics.csv"
SUMMARY_FILE = METRICS_DIR / "prelaunch_model_training_summary.txt"


def load_splits():
    frames = {}

    for name, path in INPUT_FILES.items():
        if not path.exists():
            print(f"ERROR: Missing input file: {path}")
            sys.exit(1)

        frames[name] = pd.read_csv(path, low_memory=False)

        if TARGET_COLUMN not in frames[name].columns:
            print(f"ERROR: Target column missing in {path}")
            sys.exit(1)

        if frames[name][TARGET_COLUMN].isna().any():
            print(f"ERROR: Missing target values in {path}")
            sys.exit(1)

    if list(frames["train"].columns) != list(
        frames["validation"].columns
    ):
        print("ERROR: Training and validation columns do not match.")
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
        print("ERROR: XGBoost is not installed. Install it with:")
        print("python -m pip install xgboost")
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

    feature_columns = [
        column for column in frames["train"].columns
        if column != TARGET_COLUMN
    ]

    train_x = frames["train"][feature_columns].copy()
    train_y = frames["train"][TARGET_COLUMN].astype("int8")

    validation_x = frames["validation"][feature_columns].copy()
    validation_y = frames["validation"][TARGET_COLUMN].astype("int8")

    all_metrics = []
    summary_lines = [
        "STEAM GAMES SUCCESS PREDICTION",
        "BEFORE-LAUNCH MODEL TRAINING SUMMARY",
        "=" * 64,
        "Training data: prelaunch/train_preprocessed.csv",
        "Validation data: prelaunch/validation_preprocessed.csv",
        f"Training rows: {len(train_x):,}",
        f"Validation rows: {len(validation_x):,}",
        f"Predictor columns: {len(feature_columns)}",
        f"Random seed: {RANDOM_SEED}",
        "",
        "VALIDATION METRICS",
        "-" * 64,
    ]

    models = get_models()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        for model_name, model in models.items():
            print(f"Training {model_name} on before-launch features...")

            model.fit(train_x, train_y)
            metrics = evaluate_model(
                model, validation_x, validation_y
            )

            safe_name = model_name.lower().replace(" ", "_")
            model_path = MODELS_DIR / f"prelaunch_{safe_name}.joblib"

            joblib.dump(
                {
                    "model": model,
                    "feature_columns": feature_columns,
                    "experiment": "prelaunch",
                },
                model_path,
            )

            all_metrics.append({
                "experiment": "prelaunch",
                "model": model_name,
                **metrics,
            })

            summary_lines.append(
                f"{model_name}: "
                f"Accuracy={metrics['accuracy']:.4f}, "
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

    pd.DataFrame(all_metrics).to_csv(METRICS_FILE, index=False)
    SUMMARY_FILE.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("\nBefore-launch model training completed.")
    print(f"Metrics saved to: {METRICS_FILE}")
    print(f"Summary saved to: {SUMMARY_FILE}")
    print(f"Models saved in: {MODELS_DIR}")


if __name__ == "__main__":
    main()
