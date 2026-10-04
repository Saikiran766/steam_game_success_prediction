# Train the baseline and three main classification models using the
# training set only for fitting and the validation set for initial comparison.

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
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
        "XGBoost is not installed. Install it with: pip install xgboost"
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

# ROC-AUC is fixed as the primary validation metric before model comparison.
PRIMARY_METRIC = "ROC-AUC"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading preprocessed training and validation data...")

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET].astype(int)

X_validation = validation.drop(columns=[TARGET])
y_validation = validation[TARGET].astype(int)

print(f"Training rows: {len(train)}")
print(f"Validation rows: {len(validation)}")
print(f"Training features: {X_train.shape[1]}")

# Define the baseline and the three required main models.
models = {
    "baseline": DummyClassifier(
        strategy="most_frequent"
    ),
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED
    ),
    "random_forest": RandomForestClassifier(
        random_state=RANDOM_SEED
    ),
    "xgboost": XGBClassifier(
        random_state=RANDOM_SEED,
        eval_metric="logloss"
    ),
}

def evaluate_model(model, X, y):
    # Generate class predictions and probability estimates for validation.
    predictions = model.predict(X)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        probabilities = model.decision_function(X)

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
    }, predictions, probabilities


validation_results = {}
validation_predictions = {}

for model_name, model in models.items():
    print(f"\nTraining: {model_name}")

    model.fit(X_train, y_train)

    metrics, predictions, probabilities = evaluate_model(
        model,
        X_validation,
        y_validation
    )

    validation_results[model_name] = metrics

    validation_predictions[model_name] = {
        "predictions": predictions,
        "probabilities": probabilities,
    }

    model_file = MODELS_DIR / f"{model_name}.joblib"
    joblib.dump(model, model_file)

    print(f"Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall:    {metrics['Recall']:.4f}")
    print(f"F1:        {metrics['F1']:.4f}")
    print(f"ROC-AUC:   {metrics['ROC-AUC']:.4f}")
    print(f"Saved: {model_file}")

# Create a validation comparison table.
comparison = pd.DataFrame(validation_results).T
comparison.index.name = "Model"

comparison_file = METRICS_DIR / "validation_model_comparison.csv"
comparison.to_csv(comparison_file)

# Save a JSON copy for reproducibility.
results_file = METRICS_DIR / "validation_metrics.json"
results_file.write_text(
    json.dumps(validation_results, indent=2),
    encoding="utf-8"
)

# Save validation predictions/probabilities for the later tuning step.
prediction_rows = []

for model_name, output in validation_predictions.items():
    for index, (prediction, probability) in enumerate(
        zip(
            output["predictions"],
            output["probabilities"]
        )
    ):
        prediction_rows.append({
            "row_index": index,
            "model": model_name,
            "actual": int(y_validation.iloc[index]),
            "prediction": int(prediction),
            "probability_success": float(probability),
        })

prediction_file = METRICS_DIR / "validation_predictions.csv"

pd.DataFrame(prediction_rows).to_csv(
    prediction_file,
    index=False
)

# Write a concise training report.
report_lines = [
    "Model Training Summary",
    "======================",
    "",
    f"Primary validation metric: {PRIMARY_METRIC}",
    f"Random seed: {RANDOM_SEED}",
    "",
    "Models:",
    "1. Majority-class baseline",
    "2. Logistic Regression",
    "3. Random Forest",
    "4. XGBoost",
    "",
    "The baseline is included for reference and is not counted as one of",
    "the three main machine-learning models.",
    "",
    "Training data is used for model fitting.",
    "Validation data is used for initial model comparison.",
    "The test set is not loaded or used in this script.",
    "",
    "Validation results:",
]

for model_name, metrics in validation_results.items():
    report_lines.extend([
        "",
        model_name,
        f"Accuracy:  {metrics['Accuracy']:.4f}",
        f"Precision: {metrics['Precision']:.4f}",
        f"Recall:    {metrics['Recall']:.4f}",
        f"F1:        {metrics['F1']:.4f}",
        f"ROC-AUC:   {metrics['ROC-AUC']:.4f}",
    ])

report_lines.extend([
    "",
    f"Comparison table: {comparison_file}",
    f"Metrics JSON: {results_file}",
    f"Validation predictions: {prediction_file}",
])

report_file = METRICS_DIR / "model_training_summary.txt"
report_file.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print("\nModel training completed successfully.")
print(f"\nValidation comparison:\n{comparison}")
print(f"\nComparison saved to:\n{comparison_file}")
print(f"Metrics saved to:\n{results_file}")
print(f"Validation predictions saved to:\n{prediction_file}")
print(f"Training summary saved to:\n{report_file}")
