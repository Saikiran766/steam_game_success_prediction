# Import the libraries required for release-maturity target analysis

import pandas as pd
from pathlib import Path

# Define the project paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_FILE = PROJECT_ROOT / "data" / "cleaned" / "steam_games_cleaned.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Define the release-maturity cutoff for the analysis

MATURITY_DATE = pd.Timestamp("2025-01-01")

# Check whether the cleaned dataset exists

if not CLEANED_FILE.exists():
    raise FileNotFoundError(f"Cleaned dataset not found: {CLEANED_FILE}")

print("Cleaned dataset found:")
print(CLEANED_FILE)

# Load the cleaned dataset

df = pd.read_csv(
    CLEANED_FILE,
    low_memory=False
)

# Parse release dates

df["Release date"] = pd.to_datetime(
    df["Release date"],
    errors="coerce"
)

# Parse Estimated owners into lower and upper bounds

owner_parts = (
    df["Estimated owners"]
    .astype("string")
    .str.replace(",", "", regex=False)
    .str.extract(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
)

df["owners_lower"] = pd.to_numeric(owner_parts[0], errors="coerce")
df["owners_upper"] = pd.to_numeric(owner_parts[1], errors="coerce")

# Identify games released before the maturity cutoff

mature_mask = df["Release date"] < MATURITY_DATE
mature_df = df.loc[mature_mask].copy()
recent_df = df.loc[~mature_mask].copy()

print(f"Total games: {len(df)}")
print(f"Maturity cutoff: {MATURITY_DATE.date()}")
print(f"Mature games before cutoff: {len(mature_df)}")
print(f"Games on/after cutoff: {len(recent_df)}")

# Show owner-range distribution for mature games

range_counts = (
    mature_df["Estimated owners"]
    .value_counts(dropna=False)
    .sort_index()
    .rename("game_count")
    .reset_index()
    .rename(columns={"index": "Estimated owners"})
)

range_counts["percentage"] = (
    range_counts["game_count"] / len(mature_df) * 100
).round(2)

print("\nOwner-range distribution among mature games:")
print(range_counts.to_string(index=False))

# Analyze target candidates using the lower bound of the owner range

thresholds = [
    20_000,
    50_000,
    100_000,
    200_000,
    500_000,
    1_000_000,
    2_000_000,
    5_000_000
]

threshold_rows = []

for threshold in thresholds:
    success_mask = mature_df["owners_lower"] >= threshold
    success_count = success_mask.sum()
    not_success_count = (~success_mask).sum()

    threshold_rows.append({
        "target_rule": f"owners_lower >= {threshold:,}",
        "success_count": success_count,
        "not_success_count": not_success_count,
        "success_percentage": round(
            success_count / len(mature_df) * 100, 2
        ),
        "not_success_percentage": round(
            not_success_count / len(mature_df) * 100, 2
        )
    })

threshold_summary = pd.DataFrame(threshold_rows)

print("\nTarget candidates using the LOWER owner-range bound:")
print(threshold_summary.to_string(index=False))

# Analyze a conservative target using the entire owner range

print("\nOwner ranges that are completely above selected thresholds:")

conservative_rows = []

for threshold in thresholds:
    success_mask = mature_df["owners_lower"] >= threshold
    success_count = success_mask.sum()

    conservative_rows.append({
        "threshold": threshold,
        "definite_success_count": success_count,
        "definite_success_percentage": round(
            success_count / len(mature_df) * 100, 2
        )
    })

conservative_summary = pd.DataFrame(conservative_rows)

print(conservative_summary.to_string(index=False))

# Compare recent games with mature games

recent_distribution = pd.DataFrame({
    "group": ["Mature games", "Recent games"],
    "count": [len(mature_df), len(recent_df)]
})

print("\nRelease-maturity comparison:")
print(recent_distribution.to_string(index=False))

# Save the complete analysis report

report_file = RESULTS_DIR / "target_maturity_analysis.txt"

with open(report_file, "w", encoding="utf-8") as file:
    file.write("Steam Games Dataset - Target and Release-Maturity Analysis\n")
    file.write("=" * 70 + "\n")
    file.write(f"Maturity cutoff: {MATURITY_DATE.date()}\n")
    file.write(f"Total games: {len(df)}\n")
    file.write(f"Mature games: {len(mature_df)}\n")
    file.write(f"Games on/after cutoff: {len(recent_df)}\n\n")

    file.write("Owner-range distribution among mature games:\n")
    file.write(range_counts.to_string(index=False))
    file.write("\n\n")

    file.write("Target candidates using the lower owner-range bound:\n")
    file.write(threshold_summary.to_string(index=False))
    file.write("\n\n")

    file.write("Definite success counts:\n")
    file.write(conservative_summary.to_string(index=False))
    file.write("\n")

print("\nAnalysis saved to:")
print(report_file)

print("\nNo target column or balanced dataset has been created.")
print("This script is only for selecting the target rule and maturity cutoff.")
