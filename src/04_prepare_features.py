"""
Prepare a leakage-safe feature dataset for Steam Games Success Prediction.

Input:
    data/cleaned/steam_games_targeted.csv

Outputs:
    data/processed/steam_games_features.csv
    results/metrics/feature_preparation_summary.txt

This step removes identifiers, URLs, free-text fields, and selected high-cardinality
fields from the predictor dataset. Raw feature values are retained where possible
so the same eligible rows can later be used for a fair raw-vs-preprocessed comparison.
"""

from pathlib import Path
import sys

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT_DIR / "data" / "cleaned" / "steam_games_targeted.csv"
OUTPUT_FILE = ROOT_DIR / "data" / "processed" / "steam_games_features.csv"
REPORT_FILE = ROOT_DIR / "results" / "metrics" / "feature_preparation_summary.txt"

TARGET_COLUMN = "Success"
EXPECTED_TARGETS = {0, 1}

EXCLUDED_COLUMNS = {
    "AppID",
    "Name",
    "Release date",
    "Estimated owners",
    "Estimated owners lower bound",
    "About the game",
    "Header image",
    "Website",
    "Support url",
    "Support email",
    "Metacritic url",
    "Notes",
    "Developers",
    "Publishers",
    "Categories",
    "Genres",
    "Tags",
    "Screenshots",
    "Movies",
    "Supported languages",
    "Full audio languages",
}


def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file was not found:\n{INPUT_FILE}")
        print("Run src/03_create_target.py first.")
        sys.exit(1)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, index_col=False, low_memory=False)

    if TARGET_COLUMN not in df.columns:
        print(f"ERROR: Required target column '{TARGET_COLUMN}' was not found.")
        sys.exit(1)

    target_values = set(pd.to_numeric(df[TARGET_COLUMN], errors="coerce").dropna().unique())
    if not target_values.issubset(EXPECTED_TARGETS):
        print(f"ERROR: Unexpected values in target column: {sorted(target_values)}")
        sys.exit(1)

    present_exclusions = sorted(EXCLUDED_COLUMNS.intersection(df.columns))
    absent_exclusions = sorted(EXCLUDED_COLUMNS.difference(df.columns))

    columns_to_keep = [column for column in df.columns if column not in EXCLUDED_COLUMNS]
    if TARGET_COLUMN not in columns_to_keep:
        columns_to_keep.append(TARGET_COLUMN)

    features_df = df[columns_to_keep].copy()
    features_df[TARGET_COLUMN] = pd.to_numeric(
        features_df[TARGET_COLUMN], errors="raise"
    ).astype("int8")

    leakage_columns = {"Estimated owners", "Estimated owners lower bound"}.intersection(
        features_df.columns
    )
    if leakage_columns:
        print("ERROR: Target leakage columns remain: " + ", ".join(sorted(leakage_columns)))
        sys.exit(1)

    features_df.to_csv(OUTPUT_FILE, index=False)

    rows, columns = features_df.shape
    counts = features_df[TARGET_COLUMN].value_counts().to_dict()
    not_success_count = int(counts.get(0, 0))
    success_count = int(counts.get(1, 0))

    report_lines = [
        "STEAM GAMES SUCCESS PREDICTION — FEATURE PREPARATION SUMMARY",
        "=" * 66,
        f"Input file: {INPUT_FILE.relative_to(ROOT_DIR)}",
        f"Output file: {OUTPUT_FILE.relative_to(ROOT_DIR)}",
        "",
        "DATASET SHAPE",
        "-" * 66,
        f"Rows: {rows:,}",
        f"Columns including target: {columns:,}",
        f"Predictor columns: {columns - 1:,}",
        "",
        "EXCLUDED COLUMNS",
        "-" * 66,
        "Columns present and excluded:",
    ]
    report_lines.extend([f"- {column}" for column in present_exclusions])
    report_lines.extend([
        "",
        "Expected exclusion columns not found in input:",
    ])
    report_lines.extend([f"- {column}" for column in absent_exclusions] if absent_exclusions else ["None"])
    report_lines.extend([
        "",
        "TARGET DISTRIBUTION",
        "-" * 66,
        f"Not Success (0): {not_success_count:,} ({not_success_count / rows * 100:.2f}%)" if rows else "Not Success (0): 0",
        f"Success (1): {success_count:,} ({success_count / rows * 100:.2f}%)" if rows else "Success (1): 0",
        "",
        "MISSING VALUES",
        "-" * 66,
        f"Total missing cells in retained dataset: {int(features_df.isna().sum().sum()):,}",
        "Missing values are intentionally not imputed in this step.",
        "",
        "IMPORTANT METHODOLOGY NOTES",
        "-" * 66,
        "- The target column 'Success' is retained for later training/evaluation.",
        "- 'Estimated owners' and 'Estimated owners lower bound' are excluded to prevent target leakage.",
        "- Raw predictor values are retained; encoding, scaling, and imputation belong to preprocessing.",
        "- The same rows should be used for raw and preprocessed experiments.",
    ])

    REPORT_FILE.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("Feature preparation completed successfully.")
    print(f"Rows: {rows:,}")
    print(f"Predictor columns: {columns - 1:,}")
    print(
        f"Target counts — Not Success (0): {not_success_count:,}; "
        f"Success (1): {success_count:,}"
    )
    print(f"\nDataset saved to: {OUTPUT_FILE}")
    print(f"Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
