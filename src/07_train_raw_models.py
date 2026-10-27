"""
Train baseline models on the original (not imputed/scaled/one-hot encoded) feature splits.

Inputs:
    data/processed/train.csv
    data/processed/validation.csv
    data/processed/test.csv

Outputs:
    results/models/raw_logistic_regression.joblib
    results/models/raw_random_forest.joblib
    results/models/raw_xgboost.joblib
    results/metrics/raw_model_validation_metrics.csv
    results/metrics/raw_model_training_summary.txt

Categorical values are converted to pandas category codes using mappings learned
from the training split only. Numeric missing values are imputed with training
medians. These are minimal model-compatibility steps, not the full preprocessing
pipeline. Validation/test mappings never influence training.
"""

from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.class_weight import compute_class_weight


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "results" / "models"
METRICS_DIR = ROOT_DIR / "results" / "metrics"

INPUT_FILES = {
    "train": PROCESSED_DIR / "train.csv",
    "validation": PROCESSED_DIR / "validation.csv",
    "test": PROCESSED_DIR / "test.csv",
}
TARGET_COLUMN = "Success"
RANDOM_SEED = 42


def load_splits():
    missing = [str(path) for path in INPUT_FILES.values() if not path.exists()]
    if missing:
        print("ERROR: Missing split file(s):")
        for path in missing:
            print(f" - {path}")
        print("Run src/05_split_data.py first.")
        sys.exit(1)

    frames = {
        name: pd.read_csv(path, index_col=False, low_memory=False)
        for name, path in INPUT_FILES.items()
    }
    for name, frame in frames.items():
        if TARGET_COLUMN not in frame.columns:
            print(f"ERROR: '{TARGET_COLUMN}' not found in {name} split.")
            sys.exit(1)
        if frame[TARGET_COLUMN].isna().any():
            print(f"ERROR: Missing target values in {name} split.")
            sys.exit(1)
    return frames


def build_model_preprocessor(train_x):
    numeric_columns = [
        col for col in train_x.columns
        if pd.api.types.is_numeric_dtype(train_x[col])
    ]
    categorical_columns = [col for col in train_x.columns if col not in numeric_columns]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )),
    ])

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric_columns, categorical_columns


def get_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
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
    feature_columns = [c for c in frames["train"].columns if c != TARGET_COLUMN]

    for split_name, frame in frames.items():
        missing_cols = [c for c in feature_columns if c not in frame.columns]
        if missing_cols:
            print(f"ERROR: {split_name} split missing columns: {missing_cols}")
            sys.exit(1)

    train_x = frames["train"][feature_columns].copy()
    train_y = frames["train"][TARGET_COLUMN].astype("int8")
    validation_x = frames["validation"][feature_columns].copy()
    validation_y = frames["validation"][TARGET_COLUMN].astype("int8")

    preprocessor, numeric_columns, categorical_columns = build_model_preprocessor(train_x)
    models = get_models()

    all_metrics = []
    summary = [
        "STEAM GAMES SUCCESS PREDICTION — RAW-FEATURE MODEL TRAINING",
        "=" * 68,
        "Input data: original train.csv and validation.csv splits.",
        "Compatibility processing: numeric median imputation + scaling; categorical mode imputation + ordinal encoding.",
        "These steps are minimal model-compatibility transformations; compare against the full preprocessing pipeline separately.",
        f"Numeric columns: {len(numeric_columns)}",
        f"Categorical columns: {len(categorical_columns)}",
        f"Random seed: {RANDOM_SEED}",
        "",
        "VALIDATION METRICS",
        "-" * 68,
    ]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        preprocessor.fit(train_x)
        train_x_ready = preprocessor.transform(train_x)
        validation_x_ready = preprocessor.transform(validation_x)

        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(train_x_ready, train_y)
            metrics = evaluate_model(model, validation_x_ready, validation_y)
            metrics_row = {"experiment": "raw_features", "model": name, **metrics}
            all_metrics.append(metrics_row)

            model_path = MODELS_DIR / f"raw_{name.lower().replace(' ', '_')}.joblib"
            joblib.dump(
                {
                    "model": model,
                    "preprocessor": preprocessor,
                    "feature_columns": feature_columns,
                    "numeric_columns": numeric_columns,
                    "categorical_columns": categorical_columns,
                    "experiment": "raw_features",
                },
                model_path,
            )

            summary.append(
                f"{name}: Accuracy={metrics['accuracy']:.4f}, "
                f"Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}, "
                f"F1={metrics['f1']:.4f}, "
                f"ROC-AUC={metrics['roc_auc']:.4f}"
            )
            print(
                f"  Accuracy={metrics['accuracy']:.4f}, "
                f"F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}"
            )

    metrics_file = METRICS_DIR / "raw_model_validation_metrics.csv"
    pd.DataFrame(all_metrics).to_csv(metrics_file, index=False)
    summary.extend([
        "",
        "NOTES",
        "-" * 68,
        "- Validation data was used for evaluation; the test set was not used here.",
        "- Minimal ordinal encoding makes categorical fields numeric for model compatibility.",
        "- Ordinal encoding can impose artificial order on categories; this is a limitation of this raw-feature baseline.",
        "- Reuse the same split files for the preprocessed-feature experiments.",
        f"Metrics CSV: {metrics_file.relative_to(ROOT_DIR)}",
    ])
    summary_file = METRICS_DIR / "raw_model_training_summary.txt"
    summary_file.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("\nRaw-feature model training completed.")
    print(f"Metrics saved to: {metrics_file}")
    print(f"Summary saved to: {summary_file}")
    print(f"Models saved in: {MODELS_DIR}")


if __name__ == "__main__":
    main()
