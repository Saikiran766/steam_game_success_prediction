
"""
Preprocess Steam Games Success Prediction data for the baseline experiment.

Inputs:
    data/processed/train.csv
    data/processed/validation.csv
    data/processed/test.csv

Outputs:
    data/processed/train_preprocessed.csv
    data/processed/validation_preprocessed.csv
    data/processed/test_preprocessed.csv
    data/processed/preprocessor.joblib
    results/metrics/baseline_preprocessing_summary.txt

The preprocessor is fitted on training predictors only.
The Success target is never used as a predictor.
"""

from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results" / "metrics"

TARGET_COLUMN = "Success"
MAX_CATEGORIES_PER_COLUMN = 100

INPUT_FILES = {
    "train": PROCESSED_DIR / "train.csv",
    "validation": PROCESSED_DIR / "validation.csv",
    "test": PROCESSED_DIR / "test.csv",
}

OUTPUT_FILES = {
    "train": PROCESSED_DIR / "train_preprocessed.csv",
    "validation": PROCESSED_DIR / "validation_preprocessed.csv",
    "test": PROCESSED_DIR / "test_preprocessed.csv",
}

PREPROCESSOR_FILE = PROCESSED_DIR / "preprocessor.joblib"

REPORT_FILE = (
    RESULTS_DIR / "baseline_preprocessing_summary.txt"
)


def stop_with_error(message):
    print(f"\nERROR: {message}")
    sys.exit(1)


# --------------------------------------------------
# 2. LOAD AND VALIDATE SPLITS
# --------------------------------------------------

def main():

    missing_files = [
        str(path)
        for path in INPUT_FILES.values()
        if not path.exists()
    ]

    if missing_files:
        stop_with_error(
            "Required split files are missing:\n"
            + "\n".join(missing_files)
            + "\nRun src/05_split_data.py first."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    split_data = {
        name: pd.read_csv(
            path,
            low_memory=False,
        )
        for name, path in INPUT_FILES.items()
    }

    for name, frame in split_data.items():

        if TARGET_COLUMN not in frame.columns:
            stop_with_error(
                f"'{TARGET_COLUMN}' is missing from {name}.csv."
            )

        if frame.empty:
            stop_with_error(
                f"The {name} split is empty."
            )

        if frame[TARGET_COLUMN].isna().any():
            stop_with_error(
                f"Missing target values found in {name}.csv."
            )

        invalid_targets = (
            set(frame[TARGET_COLUMN].unique()) - {0, 1}
        )

        if invalid_targets:
            stop_with_error(
                f"Unexpected target values in {name}.csv: "
                f"{invalid_targets}"
            )

    # Ensure that the same columns occur in all three splits.
    train_columns = list(split_data["train"].columns)

    for name, frame in split_data.items():
        if list(frame.columns) != train_columns:
            stop_with_error(
                f"Column names or order differ in {name}.csv."
            )

    # The target must never enter the predictor matrix.
    feature_columns = [
        column
        for column in train_columns
        if column != TARGET_COLUMN
    ]

    if not feature_columns:
        stop_with_error("No predictor columns were found.")

    # Keep predictors and target separate.
    for name, frame in split_data.items():
        split_data[name] = frame[
            feature_columns + [TARGET_COLUMN]
        ].copy()

    train_x = split_data["train"][feature_columns].copy()

    # --------------------------------------------------
    # 3. IDENTIFY NUMERIC AND CATEGORICAL FEATURES
    # --------------------------------------------------

    numeric_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(train_x[column])
    ]

    categorical_columns = [
        column
        for column in feature_columns
        if column not in numeric_columns
    ]

    # Convert numeric values safely and replace infinities.
    for name, frame in split_data.items():

        for column in numeric_columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

            frame[column] = frame[column].replace(
                [np.inf, -np.inf],
                np.nan,
            )

        # Consistent categorical representation across splits.
        for column in categorical_columns:
            frame[column] = frame[column].map(
                lambda value: (
                    np.nan if pd.isna(value) else str(value)
                )
            )

    train_x = split_data["train"][feature_columns].copy()

    # --------------------------------------------------
    # 4. BUILD PREPROCESSING PIPELINES
    # --------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                    keep_empty_features=True,
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    max_categories=MAX_CATEGORIES_PER_COLUMN,
                ),
            ),
        ]
    )

    transformers = []

    if numeric_columns:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # --------------------------------------------------
    # 5. FIT ON TRAINING DATA ONLY
    # --------------------------------------------------

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        preprocessor.fit(train_x)

    feature_names = preprocessor.get_feature_names_out()

    # --------------------------------------------------
    # 6. TRANSFORM AND SAVE EACH SPLIT
    # --------------------------------------------------

    output_frames = {}

    for name, frame in split_data.items():

        x = frame[feature_columns].copy()

        transformed = preprocessor.transform(x)

        transformed_df = pd.DataFrame(
            transformed,
            columns=feature_names,
            index=frame.index,
        )

        transformed_df[TARGET_COLUMN] = (
            frame[TARGET_COLUMN].astype("int8").to_numpy()
        )

        output_frames[name] = transformed_df

        transformed_df.to_csv(
            OUTPUT_FILES[name],
            index=False,
        )

        print(
            f"Saved {name}: "
            f"{len(transformed_df):,} rows, "
            f"{len(transformed_df.columns):,} columns"
        )

    # Save the fitted preprocessor separately from prelaunch.
    joblib.dump(
        preprocessor,
        PREPROCESSOR_FILE,
    )

    # --------------------------------------------------
    # 7. VALIDATE OUTPUTS
    # --------------------------------------------------

    total_missing = sum(
        int(frame.isna().sum().sum())
        for frame in output_frames.values()
    )

    if total_missing:
        stop_with_error(
            f"{total_missing:,} missing cells remain "
            "after preprocessing."
        )

    for name, frame in output_frames.items():

        if list(frame.columns) != list(
            output_frames["train"].columns
        ):
            stop_with_error(
                f"Output column mismatch in {name}."
            )

        if len(frame) != len(split_data[name]):
            stop_with_error(
                f"Row count changed in {name}."
            )

    # --------------------------------------------------
    # 8. WRITE SUMMARY REPORT
    # --------------------------------------------------

    report_lines = [
        "STEAM GAMES SUCCESS PREDICTION",
        "BASELINE PREPROCESSING SUMMARY",
        "=" * 64,
        "Experiment: baseline preprocessed features",
        "Preprocessor fitted on training predictors only.",
        f"Original predictor columns: {len(feature_columns)}",
        f"Transformed predictor columns: {len(feature_names)}",
        f"Numeric columns: {len(numeric_columns)}",
        f"Categorical columns: {len(categorical_columns)}",
        f"Missing cells after preprocessing: {total_missing}",
        "",
        "NUMERIC FEATURES",
        "-" * 64,
        *[f"  - {column}" for column in numeric_columns],
        "",
        "CATEGORICAL FEATURES",
        "-" * 64,
        *[f"  - {column}" for column in categorical_columns],
        "",
        "SPLIT SHAPES",
        "-" * 64,
    ]

    for name, frame in output_frames.items():
        report_lines.append(
            f"{name.title()}: "
            f"{frame.shape[0]:,} rows x "
            f"{frame.shape[1]:,} columns"
        )

    report_lines.extend([
        "",
        "PREPROCESSING DETAILS",
        "-" * 64,
        "- Target column: Success.",
        "- Success is excluded from all predictor transformations.",
        "- Numeric features use median imputation and standard scaling.",
        "- Categorical features use most-frequent imputation.",
        "- Categorical features are one-hot encoded.",
        "- Unknown validation/test categories are ignored.",
        "- The preprocessor was fitted using training data only.",
        "- Baseline outputs are stored outside the prelaunch folder.",
        "",
        "OUTPUT FILES",
        "-" * 64,
        *[
            str(path.relative_to(ROOT_DIR))
            for path in OUTPUT_FILES.values()
        ],
        str(PREPROCESSOR_FILE.relative_to(ROOT_DIR)),
    ])

    REPORT_FILE.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\nBaseline preprocessing completed successfully.")
    print(f"Numeric features: {len(numeric_columns)}")
    print(f"Categorical features: {len(categorical_columns)}")
    print(f"Transformed features: {len(feature_names)}")
    print(f"Missing cells: {total_missing}")
    print(f"Preprocessor: {PREPROCESSOR_FILE}")
    print(f"Summary: {REPORT_FILE}")


if __name__ == "__main__":
    main()
