"""
Feature-importance analysis for Steam Games Success Prediction.

What this script does:
1. Loads the six saved model bundles (three algorithms, raw/preprocessed features).
2. Calculates feature importance:
   - Logistic Regression: absolute model coefficients
   - Random Forest and XGBoost: built-in feature importance
3. Saves the complete importance table and a top-20 chart for each experiment.

Run from the project root:
    python src/11_feature_importance.py
"""

from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd


# Project folders
ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "results" / "models"
METRICS_DIR = ROOT_DIR / "results" / "metrics"
FIGURES_DIR = ROOT_DIR / "results" / "figures"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Model files created by the earlier training scripts.
EXPERIMENTS = [
    ("Raw features", "Logistic Regression", "raw_logistic_regression.joblib"),
    ("Raw features", "Random Forest", "raw_random_forest.joblib"),
    ("Raw features", "XGBoost", "raw_xgboost.joblib"),
    ("Preprocessed features", "Logistic Regression", "preprocessed_logistic_regression.joblib"),
    ("Preprocessed features", "Random Forest", "preprocessed_random_forest.joblib"),
    ("Preprocessed features", "XGBoost", "preprocessed_xgboost.joblib"),
]


def unpack_model_bundle(loaded):
    """Support either a saved dictionary bundle or a model saved directly."""
    if isinstance(loaded, dict):
        model = loaded.get("model")
        if model is None:
            # Accommodate bundles that use another common model key.
            for key in ("estimator", "classifier", "best_model"):
                if key in loaded:
                    model = loaded[key]
                    break
        if model is None:
            raise ValueError(
                "Could not find the model object in this bundle. "
                "Expected a 'model' key."
            )

        feature_names = (
            loaded.get("feature_columns")
            or loaded.get("original_feature_columns")
        )
        return model, feature_names

    return loaded, None


def calculate_importance(model, feature_names):
    """Return feature names and importance values for a fitted model."""
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        method = "Built-in feature importance"
    elif hasattr(model, "coef_"):
        # Absolute coefficients provide a simple importance estimate for
        # Logistic Regression. The preprocessed model uses scaled features.
        values = abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else abs(model.coef_)
        method = "Absolute coefficient"
    else:
        raise ValueError(
            f"{type(model).__name__} does not expose feature_importances_ or coef_."
        )

    values = list(values)

    if feature_names is None or len(feature_names) != len(values):
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
        else:
            feature_names = [f"feature_{i + 1}" for i in range(len(values))]

    if len(feature_names) != len(values):
        raise ValueError(
            f"Feature-name count ({len(feature_names)}) does not match "
            f"importance count ({len(values)})."
        )

    return pd.DataFrame({
        "feature": [str(name) for name in feature_names],
        "importance": [float(value) for value in values],
        "importance_method": method,
    })


def save_top_features_chart(data, experiment, model_name, output_path):
    """Save a horizontal bar chart for the 20 most important features."""
    top = data.sort_values("importance", ascending=False).head(20)
    top = top.sort_values("importance", ascending=True)

    height = max(6, min(12, 0.32 * len(top) + 1.5))
    fig, ax = plt.subplots(figsize=(11, height))
    ax.barh(top["feature"], top["importance"])
    ax.set_title(f"Top 20 Features — {model_name}\n{experiment}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    all_results = []

    for experiment, model_name, filename in EXPERIMENTS:
        model_path = MODELS_DIR / filename

        if not model_path.exists():
            warnings.warn(f"Model file not found; skipping: {model_path}")
            continue

        print(f"Analyzing {model_name} — {experiment}...")
        loaded = joblib.load(model_path)
        model, feature_names = unpack_model_bundle(loaded)
        importance_df = calculate_importance(model, feature_names)

        importance_df.insert(0, "model", model_name)
        importance_df.insert(0, "experiment", experiment)
        importance_df = importance_df.sort_values(
            "importance", ascending=False
        ).reset_index(drop=True)

        all_results.append(importance_df)

        safe_model_name = model_name.lower().replace(" ", "_")
        safe_experiment = experiment.lower().replace(" ", "_")
        chart_path = FIGURES_DIR / (
            f"feature_importance_{safe_experiment}_{safe_model_name}.png"
        )
        save_top_features_chart(
            importance_df, experiment, model_name, chart_path
        )
        print(f"  Chart saved: {chart_path}")

    if not all_results:
        raise FileNotFoundError(
            f"No model files were found in {MODELS_DIR}. "
            "Check that the earlier training scripts completed successfully."
        )

    combined = pd.concat(all_results, ignore_index=True)
    csv_path = METRICS_DIR / "feature_importance.csv"
    combined.to_csv(csv_path, index=False)

    # Also save a concise table of the top 20 features for each experiment.
    top20 = (
        combined.sort_values("importance", ascending=False)
        .groupby(["experiment", "model"], sort=False, group_keys=False)
        .head(20)
        .reset_index(drop=True)
    )
    top20_path = METRICS_DIR / "top_20_features.csv"
    top20.to_csv(top20_path, index=False)

    print("\nFeature-importance analysis completed.")
    print(f"Full results: {csv_path}")
    print(f"Top-20 results: {top20_path}")
    print(f"Charts folder: {FIGURES_DIR}")
    print(
        "\nNote: importance scores are model-specific. "
        "For Logistic Regression, absolute coefficients are used; "
        "for tree models, built-in feature importance is used. "
        "These scores show model reliance, not proof of causation."
    )


if __name__ == "__main__":
    main()
