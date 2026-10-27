"""
Create the target variable for the Steam Games Success Prediction project.

Input:
    data/cleaned/steam_games_cleaned.csv

Outputs:
    data/cleaned/steam_games_targeted.csv
    results/metrics/target_definition.txt

Target definition:
    Success (1): estimated owners' lower bound is at least 20,000.
    Not Success (0): estimated owners' lower bound is below 20,000.

Only games released before 2025-01-01 are retained.
The original raw dataset is never modified.
"""

from pathlib import Path
import re
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT_DIR / "data" / "cleaned" / "steam_games_cleaned.csv"
OUTPUT_FILE = ROOT_DIR / "data" / "cleaned" / "steam_games_targeted.csv"
REPORT_FILE = ROOT_DIR / "results" / "metrics" / "target_definition.txt"

RELEASE_CUTOFF = pd.Timestamp("2025-01-01")
OWNER_THRESHOLD = 20_000


def parse_owner_lower_bound(value):
    """Return the first number in an Estimated owners value, or None."""
    if pd.isna(value):
        return None
    match = re.search(r"([\d,]+)", str(value).strip())
    if not match:
        return None
    try:
        return int(match.group(1).replace(",", ""))
    except ValueError:
        return None


def parse_release_dates(series):
    """Parse mixed release-date strings without inventing dates."""
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="coerce")


def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file was not found:\n{INPUT_FILE}")
        print("Run src/02_clean_data.py first.")
        sys.exit(1)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, index_col=False, low_memory=False)
    rows_before = len(df)
    required_columns = ["Release date", "Estimated owners"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        print("ERROR: Required column(s) missing: " + ", ".join(missing_columns))
        sys.exit(1)

    parsed_dates = parse_release_dates(df["Release date"])
    owner_lower_bounds = df["Estimated owners"].apply(parse_owner_lower_bound)

    invalid_date_count = int(parsed_dates.isna().sum())
    invalid_owner_count = int(owner_lower_bounds.isna().sum())
    valid_date_mask = parsed_dates.notna()
    before_cutoff_mask = valid_date_mask & (parsed_dates < RELEASE_CUTOFF)
    recent_or_cutoff_count = int((valid_date_mask & (parsed_dates >= RELEASE_CUTOFF)).sum())
    keep_mask = before_cutoff_mask & owner_lower_bounds.notna()

    result = df.loc[keep_mask].copy()
    result["Release date"] = parsed_dates.loc[keep_mask].dt.strftime("%Y-%m-%d")
    result["Estimated owners lower bound"] = owner_lower_bounds.loc[keep_mask].astype("int64")
    result["Success"] = (result["Estimated owners lower bound"] >= OWNER_THRESHOLD).astype("int8")
    result.to_csv(OUTPUT_FILE, index=False)

    total_retained = len(result)
    success_count = int((result["Success"] == 1).sum())
    not_success_count = int((result["Success"] == 0).sum())
    success_pct = success_count / total_retained * 100 if total_retained else 0.0
    not_success_pct = not_success_count / total_retained * 100 if total_retained else 0.0

    report_lines = [
        "STEAM GAMES SUCCESS PREDICTION — TARGET DEFINITION REPORT",
        "=" * 62,
        f"Input file: {INPUT_FILE.relative_to(ROOT_DIR)}",
        f"Output file: {OUTPUT_FILE.relative_to(ROOT_DIR)}",
        "",
        "TARGET RULES",
        "-" * 62,
        f"Release-date cutoff: before {RELEASE_CUTOFF.strftime('%Y-%m-%d')}",
        f"Success: Estimated owners lower bound >= {OWNER_THRESHOLD:,}",
        f"Not Success: Estimated owners lower bound < {OWNER_THRESHOLD:,}",
        "Estimated owners lower bound is parsed from the first number in the source range.",
        "",
        "ROW COUNTS",
        "-" * 62,
        f"Rows before target filtering: {rows_before:,}",
        f"Rows with invalid/unparseable release date: {invalid_date_count:,}",
        f"Rows released on or after cutoff: {recent_or_cutoff_count:,}",
        f"Rows with invalid/unparseable Estimated owners: {invalid_owner_count:,}",
        f"Rows retained for modeling: {total_retained:,}",
        f"Rows excluded in total: {rows_before - total_retained:,}",
        "",
        "TARGET DISTRIBUTION",
        "-" * 62,
        f"Success (1): {success_count:,} ({success_pct:.2f}%)",
        f"Not Success (0): {not_success_count:,} ({not_success_pct:.2f}%)",
        "",
        "OUTPUT COLUMNS",
        "-" * 62,
        "Original columns are retained, plus:",
        "- Estimated owners lower bound",
        "- Success (target: 1 = Success, 0 = Not Success)",
        "",
        "IMPORTANT",
        "-" * 62,
        "Estimated owners is used to construct the target and must not be used as a model input.",
        "The raw dataset and cleaned input file were not modified.",
    ]
    REPORT_FILE.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("Target creation completed successfully.")
    print(f"Rows before filtering: {rows_before:,}")
    print(f"Rows retained: {total_retained:,}")
    print(f"Success (1): {success_count:,} ({success_pct:.2f}%)")
    print(f"Not Success (0): {not_success_count:,} ({not_success_pct:.2f}%)")
    print(f"\nDataset saved to: {OUTPUT_FILE}")
    print(f"Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
