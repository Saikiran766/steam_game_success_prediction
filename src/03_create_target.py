# Create the binary target using only information available in the cleaned dataset.

from pathlib import Path
import pandas as pd
import numpy as np
import re

PROJECT_ROOT = Path(r"D:\Steam_Games_Success_Prediction")
INPUT_FILE = PROJECT_ROOT / "data" / "cleaned" / "steam_games_cleaned.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "cleaned" / "steam_games_targeted.csv"
REPORT_FILE = PROJECT_ROOT / "results" / "metrics" / "target_definition.txt"

MATURITY_DATE = pd.Timestamp("2025-01-01")
SUCCESS_OWNER_THRESHOLD = 20_000

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE, low_memory=False)

rows_before = len(df)

# Parse release dates and remove rows that cannot support the maturity rule.
df["Release date"] = pd.to_datetime(df["Release date"], errors="coerce")
df = df.dropna(subset=["Release date"]).copy()

# Extract the lower and upper bounds from Estimated owners.
def parse_owner_range(value):
    if pd.isna(value):
        return np.nan, np.nan

    text = str(value).replace(",", "").strip()
    numbers = re.findall(r"\d+", text)

    if len(numbers) >= 2:
        return float(numbers[0]), float(numbers[1])
    if len(numbers) == 1:
        number = float(numbers[0])
        return number, number

    return np.nan, np.nan

owner_bounds = df["Estimated owners"].apply(parse_owner_range)
df["owners_lower"] = owner_bounds.apply(lambda x: x[0])
df["owners_upper"] = owner_bounds.apply(lambda x: x[1])

invalid_target_rows = int(
    df[["owners_lower", "owners_upper"]].isna().any(axis=1).sum()
)

df = df.dropna(subset=["owners_lower", "owners_upper"]).copy()

# Keep only games old enough for the target to be meaningfully observed.
recent_games = int((df["Release date"] >= MATURITY_DATE).sum())
df = df[df["Release date"] < MATURITY_DATE].copy()

# Define success from the meaning of the target, not from class balance.
df["Success"] = (
    df["owners_lower"] >= SUCCESS_OWNER_THRESHOLD
).astype(int)

df["Success Label"] = df["Success"].map({
    1: "Success",
    0: "Not Success"
})

success_count = int(df["Success"].sum())
not_success_count = int((df["Success"] == 0).sum())

# Keep the owner bounds for auditability at this stage.
df.to_csv(OUTPUT_FILE, index=False)

success_pct = success_count / len(df) * 100
not_success_pct = not_success_count / len(df) * 100

report = [
    "Target Definition",
    "=================",
    f"Maturity cutoff: {MATURITY_DATE.date()}",
    f"Success owner threshold: >= {SUCCESS_OWNER_THRESHOLD:,} estimated owners",
    "",
    "Meaning:",
    "Success = game has at least the selected estimated-owner threshold.",
    "Not Success = game is below the selected estimated-owner threshold.",
    "",
    "Target-source field:",
    "Estimated owners",
    "",
    "Rows before target filtering: " + str(rows_before),
    "Rows removed because target fields were invalid: " + str(invalid_target_rows),
    "Recent games excluded by maturity filter: " + str(recent_games),
    "Mature games retained: " + str(len(df)),
    "",
    f"Success: {success_count} ({success_pct:.2f}%)",
    f"Not Success: {not_success_count} ({not_success_pct:.2f}%)",
    "",
    f"Target dataset saved to: {OUTPUT_FILE}",
]

REPORT_FILE.write_text("\n".join(report), encoding="utf-8")

print(f"Rows before target filtering: {rows_before}")
print(f"Rows removed because target fields were invalid: {invalid_target_rows}")
print(f"Maturity cutoff: {MATURITY_DATE.date()}")
print(f"Recent games excluded by maturity filter: {recent_games}")
print(f"Mature games retained: {len(df)}")
print(f"Success: {success_count} ({success_pct:.2f}%)")
print(f"Not Success: {not_success_count} ({not_success_pct:.2f}%)")
print(f"\nTarget dataset saved to:\n{OUTPUT_FILE}")
print(f"Target definition report saved to:\n{REPORT_FILE}")
