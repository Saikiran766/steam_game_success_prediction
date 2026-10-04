# Interpret the final trained models using training-derived model information.
# This step does not retrain models and does not change the held-out test results.

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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
FIGURES_DIR = RESULTS_DIR / "figures"

TRAIN_FILE = DATA_DIR / "train_features.csv"
VALIDATION_FILE = DATA_DIR / "validation_features.csv"
FEATURE_NAMES_FILE = METRICS_DIR / "feature_names.json"

RANDOM_SEED = 42
TARGET = "Success"

MODELS = [
    "logistic_regression",
    "random_forest",
    "xgboost",
]

TOP_N = 20

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading training and validation data for interpretation...")

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET].astype(int)

X_validation = validation.drop(columns=[TARGET])
y_validation = validation[TARGET].astype(int)

with open(FEATURE_NAMES_FILE, "r", encoding="utf-8") as file:
    feature_names = json.load(file)

# Confirm that the saved feature names match the processed matrices.
if list(X_train.columns) != feature_names:
    raise ValueError(
        "Feature names do not match train_features.csv."
    )

# -------------------------------------------------------------------------
# Logistic Regression: coefficient-based interpretation.
# -------------------------------------------------------------------------

logistic_model = joblib.load(
    MODELS_DIR / "logistic_regression_tuned.joblib"
)

if not isinstance(logistic_model, LogisticRegression):
    raise TypeError("Loaded logistic regression model has an unexpected type.")

logistic_coefficients = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": logistic_model.coef_[0],
})

logistic_coefficients["Absolute Coefficient"] = (
    logistic_coefficients["Coefficient"].abs()
)

logistic_coefficients = logistic_coefficients.sort_values(
    "Absolute Coefficient",
    ascending=False
)

logistic_coefficients.to_csv(
    METRICS_DIR / "logistic_coefficients.csv",
    index=False
)

# Save the strongest positive and negative coefficients.
positive_coefficients = (
    logistic_coefficients
    .sort_values("Coefficient", ascending=False)
    .head(TOP_N)
)

negative_coefficients = (
    logistic_coefficients
    .sort_values("Coefficient", ascending=True)
    .head(TOP_N)
)

positive_coefficients.to_csv(
    METRICS_DIR / "logistic_positive_coefficients.csv",
    index=False
)

negative_coefficients.to_csv(
    METRICS_DIR / "logistic_negative_coefficients.csv",
    index=False
)

# -------------------------------------------------------------------------
# Tree models: impurity-based feature importance.
# These are supplementary interpretation measures.
# -------------------------------------------------------------------------

tree_importance_results = {}

for model_name in ["random_forest", "xgboost"]:
    model = joblib.load(
        MODELS_DIR / f"{model_name}_tuned.joblib"
    )

    if not hasattr(model, "feature_importances_"):
        raise TypeError(
            f"{model_name} does not expose feature_importances_."
        )

    importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_,
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    tree_importance_results[model_name] = importance

    importance.to_csv(
        METRICS_DIR / f"{model_name}_feature_importance.csv",
        index=False
    )

# -------------------------------------------------------------------------
# Permutation importance on validation data.
# This is used as the main model-agnostic interpretation method.
# -------------------------------------------------------------------------

permutation_results = {}

for model_name in MODELS:
    print(f"\nCalculating permutation importance: {model_name}")

    model = joblib.load(
        MODELS_DIR / f"{model_name}_tuned.joblib"
    )

    result = permutation_importance(
        model,
        X_validation,
        y_validation,
        scoring="roc_auc",
        n_repeats=5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    importance = pd.DataFrame({
        "Feature": X_validation.columns,
        "Mean Importance": result.importances_mean,
        "Std Importance": result.importances_std,
    })

    importance = importance.sort_values(
        "Mean Importance",
        ascending=False
    )

    permutation_results[model_name] = importance

    importance.to_csv(
        METRICS_DIR / f"{model_name}_permutation_importance.csv",
        index=False
    )

# -------------------------------------------------------------------------
# Create top-feature plots.
# -------------------------------------------------------------------------

def save_top_importance_plot(data, value_column, title, output_file):
    top = data.head(TOP_N).sort_values(
        value_column,
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.barh(
        top["Feature"].astype(str),
        top[value_column]
    )

    ax.set_xlabel(value_column)
    ax.set_ylabel("Feature")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


save_top_importance_plot(
    logistic_coefficients,
    "Absolute Coefficient",
    "Logistic Regression - Top Feature Coefficients",
    FIGURES_DIR / "logistic_top_coefficients.png"
)

for model_name, importance in tree_importance_results.items():
    save_top_importance_plot(
        importance,
        "Importance",
        f"{model_name.replace('_', ' ').title()} - Top Feature Importance",
        FIGURES_DIR / f"{model_name}_top_feature_importance.png"
    )

for model_name, importance in permutation_results.items():
    save_top_importance_plot(
        importance,
        "Mean Importance",
        f"{model_name.replace('_', ' ').title()} - Validation Permutation Importance",
        FIGURES_DIR / f"{model_name}_top_permutation_importance.png"
    )

# -------------------------------------------------------------------------
# Segment-error analysis using validation predictions.
# This identifies broad error patterns without touching the test set.
# -------------------------------------------------------------------------

segment_data = validation.copy()

segment_results = []

# Analyze release year when available in the original validation data.
if "Release date" in segment_data.columns:
    release_dates = pd.to_datetime(
        segment_data["Release date"],
        errors="coerce"
    )

    segment_data["Release Year"] = release_dates.dt.year

    segment_data["Predicted Success"] = (
        validation["Success"].values
    )

    # The processed model input no longer contains the original release date,
    # so use the selected tuned model probabilities to construct errors.
    xgb_model = joblib.load(
        MODELS_DIR / "xgboost_tuned.joblib"
    )

    xgb_probabilities = xgb_model.predict_proba(
        X_validation
    )[:, 1]

    xgb_threshold = 0.38

    segment_data["Predicted Success"] = (
        xgb_probabilities >= xgb_threshold
    ).astype(int)

    segment_data["Correct"] = (
        segment_data["Predicted Success"]
        == segment_data["Success"]
    )

    year_summary = (
        segment_data
        .dropna(subset=["Release Year"])
        .groupby("Release Year")
        .agg(
            Samples=("Success", "size"),
            Accuracy=("Correct", "mean"),
            Actual_Success_Rate=("Success", "mean"),
        )
        .reset_index()
    )

    year_summary.to_csv(
        METRICS_DIR / "xgboost_validation_error_by_release_year.csv",
        index=False
    )

# -------------------------------------------------------------------------
# Save a readable interpretation report.
# -------------------------------------------------------------------------

report_lines = [
    "Model Interpretation Summary",
    "============================",
    "",
    "Interpretation uses training/validation data only.",
    "The held-out test set is not used for feature interpretation.",
    "",
    "Primary interpretation method:",
    "Permutation importance using validation ROC-AUC.",
    "",
    "Supplementary interpretation:",
    "Logistic Regression coefficients.",
    "Random Forest and XGBoost impurity-based feature importance.",
    "",
    "Logistic Regression strongest positive coefficients:",
]

for _, row in positive_coefficients.head(10).iterrows():
    report_lines.append(
        f"{row['Feature']}: {row['Coefficient']:.6f}"
    )

report_lines.extend([
    "",
    "Logistic Regression strongest negative coefficients:",
])

for _, row in negative_coefficients.head(10).iterrows():
    report_lines.append(
        f"{row['Feature']}: {row['Coefficient']:.6f}"
    )

for model_name, importance in permutation_results.items():
    report_lines.extend([
        "",
        f"{model_name.replace('_', ' ').title()} top validation permutation features:",
    ])

    for _, row in importance.head(10).iterrows():
        report_lines.append(
            f"{row['Feature']}: "
            f"{row['Mean Importance']:.6f} "
            f"(std={row['Std Importance']:.6f})"
        )

report_lines.extend([
    "",
    "Interpretation files are stored under:",
    str(METRICS_DIR),
    "",
    "Interpretation figures are stored under:",
    str(FIGURES_DIR),
])

report_file = METRICS_DIR / "interpretation_summary.txt"
report_file.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print("\nInterpretation completed successfully.")
print(f"Top logistic coefficients saved to: {METRICS_DIR}")
print(f"Tree importance files saved to: {METRICS_DIR}")
print(f"Permutation importance files saved to: {METRICS_DIR}")
print(f"Interpretation figures saved to: {FIGURES_DIR}")
print(f"Summary saved to: {report_file}")
