"""
Create reproducible train, validation, and test splits for Steam Games Success Prediction.

Input:
    data/processed/steam_games_features.csv

Outputs:
    data/processed/train.csv
    data/processed/validation.csv
    data/processed/test.csv
    results/metrics/data_split_summary.txt

The split is stratified by Success and uses a fixed random seed. Reuse these same
split files for all models and raw-vs-preprocessed comparisons.
"""

from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT_DIR / "data" / "processed" / "steam_games_features.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORT_FILE = ROOT_DIR / "results" / "metrics" / "data_split_summary.txt"

TARGET_COLUMN = "Success"
RANDOM_SEED = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20  # fraction of the full dataset


def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file was not found:\n{INPUT_FILE}")
        print("Run src/04_prepare_features.py first.")
        sys.exit(1)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, index_col=False, low_memory=False)

    if TARGET_COLUMN not in df.columns:
        print(f"ERROR: Required target column '{TARGET_COLUMN}' was not found.")
        sys.exit(1)
    if df[TARGET_COLUMN].isna().any():
        print("ERROR: The target column contains missing values.")
        sys.exit(1)

    target_values = set(pd.to_numeric(df[TARGET_COLUMN], errors="coerce").unique())
    if not target_values.issubset({0, 1}):
        print(f"ERROR: Target must contain only 0 and 1; found: {sorted(target_values)}")
        sys.exit(1)

    # Reserve 20% for testing, then 25% of the remaining 80% for validation.
    train_validation_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df[TARGET_COLUMN],
    )
    validation_fraction_of_remaining = VALIDATION_SIZE / (1 - TEST_SIZE)
    train_df, validation_df = train_test_split(
        train_validation_df,
        test_size=validation_fraction_of_remaining,
        random_state=RANDOM_SEED,
        stratify=train_validation_df[TARGET_COLUMN],
    )

    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_file = PROCESSED_DIR / "train.csv"
    validation_file = PROCESSED_DIR / "validation.csv"
    test_file = PROCESSED_DIR / "test.csv"
    train_df.to_csv(train_file, index=False)
    validation_df.to_csv(validation_file, index=False)
    test_df.to_csv(test_file, index=False)

    total_rows = len(df)
    split_rows = len(train_df) + len(validation_df) + len(test_df)

    def class_counts(frame):
        counts = frame[TARGET_COLUMN].value_counts().to_dict()
        return int(counts.get(0, 0)), int(counts.get(1, 0))

    def format_split(name, frame):
        not_success, success = class_counts(frame)
        size_pct = len(frame) / total_rows * 100 if total_rows else 0
        not_success_pct = not_success / len(frame) * 100 if len(frame) else 0
        success_pct = success / len(frame) * 100 if len(frame) else 0
        return (
            f"{name}: {len(frame):,} rows ({size_pct:.2f}%) | "
            f"Not Success (0): {not_success:,} ({not_success_pct:.2f}%) | "
            f"Success (1): {success:,} ({success_pct:.2f}%)"
        )

    report_lines = [
        "STEAM GAMES SUCCESS PREDICTION — DATA SPLIT SUMMARY",
        "=" * 62,
        f"Input: {INPUT_FILE.relative_to(ROOT_DIR)}",
        f"Random seed: {RANDOM_SEED}",
        "Method: stratified train/validation/test split",
        "Target: Success (0 = Not Success, 1 = Success)",
        "",
        "SPLIT SIZES",
        "-" * 62,
        "Requested proportions: 60% train / 20% validation / 20% test",
        format_split("Train", train_df),
        format_split("Validation", validation_df),
        format_split("Test", test_df),
        f"Total split rows: {split_rows:,}",
        f"Original rows: {total_rows:,}",
        f"All rows assigned exactly once: {'Yes' if split_rows == total_rows else 'NO'}",
        "",
        "OUTPUT FILES",
        "-" * 62,
        f"Train: {train_file.relative_to(ROOT_DIR)}",
        f"Validation: {validation_file.relative_to(ROOT_DIR)}",
        f"Test: {test_file.relative_to(ROOT_DIR)}",
        "",
        "METHODOLOGY NOTES",
        "-" * 62,
        "- Stratification preserves target class proportions in each split.",
        "- Reuse these split files for every model and preprocessing experiment.",
        "- Fit imputers, encoders, scalers, and other learned preprocessing on training data only.",
        "- Do not use the test set for model selection or hyperparameter tuning.",
    ]
    REPORT_FILE.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("Data splitting completed successfully.")
    print(f"Original rows: {total_rows:,}")
    print(f"Train rows: {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Total split rows: {split_rows:,}")
    print("\nFiles saved:")
    print(f"Train: {train_file}")
    print(f"Validation: {validation_file}")
    print(f"Test: {test_file}")
    print(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
