
"""
Preprocess Steam Games Success Prediction data for a before-launch experiment.

Inputs:
    data/processed/train.csv
    data/processed/validation.csv
    data/processed/test.csv

Outputs (kept separate from the baseline experiment):
    data/processed/prelaunch/train_preprocessed.csv
    data/processed/prelaunch/validation_preprocessed.csv
    data/processed/prelaunch/test_preprocessed.csv
    data/processed/prelaunch/preprocessor.joblib
    results/metrics/prelaunch_preprocessing_summary.txt

Preprocessing is fitted only on the training split and then applied to
validation and test splits. The Success target is never used as a predictor.
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


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = PROCESSED_DIR / "prelaunch"
RESULTS_DIR = ROOT_DIR / "results" / "metrics"

REPORT_FILE = RESULTS_DIR / "prelaunch_preprocessing_summary.txt"
PREPROCESSOR_FILE = OUTPUT_DIR / "preprocessor.joblib"

INPUT_FILES = {
    "train": PROCESSED_DIR / "train.csv",
    "validation": PROCESSED_DIR / "validation.csv",
    "test": PROCESSED_DIR / "test.csv",
}

OUTPUT_FILES = {
    "train": OUTPUT_DIR / "train_preprocessed.csv",
    "validation": OUTPUT_DIR / "validation_preprocessed.csv",
    "test": OUTPUT_DIR / "test_preprocessed.csv",
}

TARGET_COLUMN = "Success"
MAX_CATEGORIES_PER_COLUMN = 100

# Candidate features for prediction before launch.
# Confirm that each value was available at the intended prediction time.
PRELAUNCH_FEATURES = [
    "Required age",
    "Price",
    "DiscountDLC count",
    "Windows",
    "Mac",
    "Linux",
    "Achievements",
]


def main():
    # 1. Verify that input files exist.
    missing_files = [
        str(path)
        for path in INPUT_FILES.values()
        if not path.exists()
    ]

    if missing_files:
        print("ERROR: Required input split file(s) were not found:")
        for path in missing_files:
            print(f" - {path}")
        print("Run src/05_split_data.py first.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Load the original train, validation, and test splits.
    split_data = {
        name: pd.read_csv(
            path,
            index_col=False,
            low_memory=False,
        )
        for name, path in INPUT_FILES.items()
    }

    # 3. Validate target and predictor columns.
    for name, frame in split_data.items():
        if TARGET_COLUMN not in frame.columns:
            print(
                f"ERROR: '{TARGET_COLUMN}' is missing "
                f"from {name}.csv."
            )
            sys.exit(1)

        if frame[TARGET_COLUMN].isna().any():
            print(
                f"ERROR: Missing target values found "
                f"in {name}.csv."
            )
            sys.exit(1)

        invalid_targets = set(frame[TARGET_COLUMN].unique()) - {0, 1}
        if invalid_targets:
            print(
                f"ERROR: {name}.csv contains unexpected "
                f"target values: {invalid_targets}"
            )
            sys.exit(1)

    missing_features = [
        column
        for column in PRELAUNCH_FEATURES
        if column not in split_data["train"].columns
    ]

    if missing_features:
        print(
            "ERROR: Expected before-launch feature(s) "
            f"not found in train.csv: {missing_features}"
        )
        sys.exit(1)

    for name, frame in split_data.items():
        missing = [
            column
            for column in PRELAUNCH_FEATURES
            if column not in frame.columns
        ]

        if missing:
            print(
                f"ERROR: {name}.csv is missing "
                f"predictor columns: {missing}"
            )
            sys.exit(1)

    # Use only the explicit allowlist and the target.
    for name, frame in split_data.items():
        split_data[name] = frame[
            PRELAUNCH_FEATURES + [TARGET_COLUMN]
        ].copy()

    # 4. Determine numeric and categorical columns from training data.
    train_x = split_data["train"][PRELAUNCH_FEATURES].copy()

    numeric_columns = [
        column
        for column in PRELAUNCH_FEATURES
        if pd.api.types.is_numeric_dtype(train_x[column])
    ]

    categorical_columns = [
        column
        for column in PRELAUNCH_FEATURES
        if column not in numeric_columns
    ]

    # 5. Convert numeric values and non-finite values safely.
    for name, frame in split_data.items():
        for column in numeric_columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan)

    train_x = split_data["train"][PRELAUNCH_FEATURES].copy()

    numeric_pipeline = Pipeline(steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            ),
        ),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
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
    ])

    transformers = []

    if numeric_columns:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # 6. Fit ONLY on training predictors.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        preprocessor.fit(train_x)

    feature_names = preprocessor.get_feature_names_out()

    # 7. Transform each split using the fitted training preprocessor.
    output_frames = {}

    for name, frame in split_data.items():
        x = frame[PRELAUNCH_FEATURES].copy()

        for column in numeric_columns:
            x[column] = pd.to_numeric(
                x[column],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan)

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

        # Write only to the separate prelaunch directory.
        transformed_df.to_csv(
            OUTPUT_FILES[name],
            index=False,
        )

        print(
            f"Saved {name}: "
            f"{len(transformed_df):,} rows, "
            f"{len(transformed_df.columns):,} columns"
        )

    # 8. Save the fitted preprocessor.
    joblib.dump(preprocessor, PREPROCESSOR_FILE)

    # 9. Validate the transformed output.
    total_missing_after = sum(
        int(frame.isna().sum().sum())
        for frame in output_frames.values()
    )

    if total_missing_after != 0:
        print(
            "WARNING: Missing values remain after preprocessing: "
            f"{total_missing_after:,}"
        )

    # 10. Write a summary report.
    report_lines = [
        "STEAM GAMES SUCCESS PREDICTION",
        "BEFORE-LAUNCH PREPROCESSING SUMMARY",
        "=" * 64,
        "Experiment: prelaunch",
        "The explicit feature allowlist was used.",
        "Preprocessor was fitted on training predictors only.",
        f"Candidate predictor columns ({len(PRELAUNCH_FEATURES)}):",
        *[f"  - {column}" for column in PRELAUNCH_FEATURES],
        "",
        "IMPORTANT TIMING LIMITATION",
        "-" * 64,
        "- Confirm that every predictor was available at the intended",
        "  prediction time before describing this as a strict pre-launch model.",
        "- The Success target represents the existing eventual-ownership",
        "  threshold and is not a predictor.",
        "",
        "FEATURE TYPES",
        "-" * 64,
        f"Numeric ({len(numeric_columns)}): "
        f"{', '.join(numeric_columns) or 'None'}",
        f"Categorical ({len(categorical_columns)}): "
        f"{', '.join(categorical_columns) or 'None'}",
        "",
        "SPLIT SIZES AND OUTPUT SHAPES",
        "-" * 64,
    ]

    for name in ["train", "validation", "test"]:
        frame = output_frames[name]
        report_lines.append(
            f"{name.title()}: {frame.shape[0]:,} rows x "
            f"{frame.shape[1]:,} columns"
        )

    report_lines.extend([
        f"Transformed predictor columns: {len(feature_names):,}",
        f"Total missing cells after preprocessing: {total_missing_after:,}",
        "",
        "OUTPUT FILES",
        "-" * 64,
        *(f"{name}: {path.relative_to(ROOT_DIR)}"
          for name, path in OUTPUT_FILES.items()),
        f"Fitted preprocessor: {PREPROCESSOR_FILE.relative_to(ROOT_DIR)}",
        "",
        "PREPROCESSING NOTES",
        "-" * 64,
        "- Numeric features use median imputation and standard scaling.",
        "- Categorical features use most-frequent imputation and one-hot encoding.",
        "- Unknown validation/test categories are ignored by the fitted encoder.",
        "- Original train/validation/test split files were not modified.",
        "- Outputs are stored separately to preserve baseline preprocessed files.",
    ])

    REPORT_FILE.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\nBefore-launch preprocessing completed successfully.")
    print(f"Numeric predictor columns: {len(numeric_columns)}")
    print(f"Categorical predictor columns: {len(categorical_columns)}")
    print(f"Transformed feature columns: {len(feature_names)}")
    print(f"Missing cells after preprocessing: {total_missing_after:,}")
    print(f"Preprocessor saved to: {PREPROCESSOR_FILE}")
    print(f"Summary saved to: {REPORT_FILE}")
    print("\nAll baseline preprocessed files were preserved.")


if __name__ == "__main__":
    main()
