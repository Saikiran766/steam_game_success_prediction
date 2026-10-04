# Load the raw Steam dataset and clean it without losing valid platform values.

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(r"D:\Steam_Games_Success_Prediction")

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "steam_games.csv"
CLEANED_FILE = PROJECT_ROOT / "data" / "cleaned" / "steam_games_cleaned.csv"
SUMMARY_FILE = PROJECT_ROOT / "results" / "metrics" / "cleaning_summary.txt"

# Create required output directories if they do not already exist.
CLEANED_FILE.parent.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)

# Load the raw CSV with index_col=False so the first column remains AppID.
df = pd.read_csv(
    RAW_FILE,
    index_col=False,
    low_memory=False
)

rows_before = len(df)
cols_before = len(df.columns)

# Keep a copy of the original platform columns so we can verify that cleaning
# does not accidentally turn valid Windows, Mac, or Linux values into NaN.
platform_columns = ["Windows", "Mac", "Linux"]
platform_before = {
    col: df[col].copy() for col in platform_columns if col in df.columns
}

# Remove completely duplicated rows.
duplicate_rows = int(df.duplicated().sum())
df = df.drop_duplicates().copy()

# Remove rows with missing AppID.
missing_appid = int(df["AppID"].isna().sum())
df = df.dropna(subset=["AppID"]).copy()

# Convert AppID to numeric and remove invalid IDs.
df["AppID"] = pd.to_numeric(df["AppID"], errors="coerce")
invalid_appid = int(df["AppID"].isna().sum())
df = df.dropna(subset=["AppID"]).copy()
df["AppID"] = df["AppID"].astype("int64")

# Remove duplicate AppIDs while keeping the first occurrence.
duplicate_appids = int(df["AppID"].duplicated().sum())
df = df.drop_duplicates(subset=["AppID"], keep="first").copy()

# Remove rows where the game name is missing.
missing_name = int(df["Name"].isna().sum())
df = df.dropna(subset=["Name"]).copy()

# Parse Release date into a real datetime column.
df["Release date"] = pd.to_datetime(
    df["Release date"],
    errors="coerce"
)

invalid_release_dates = int(df["Release date"].isna().sum())

# Keep only rows with a valid release date because the target definition
# later uses release maturity.
df = df.dropna(subset=["Release date"]).copy()

# Clean boolean platform columns without using astype(bool), which can
# incorrectly convert strings and can also cause conversion errors.
def clean_boolean_column(series):
    # Preserve actual boolean values.
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    # Normalize textual/numeric representations.
    normalized = series.astype("string").str.strip().str.lower()

    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f"}

    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result.loc[normalized.isin(true_values)] = True
    result.loc[normalized.isin(false_values)] = False

    return result

for col in platform_columns:
    if col in df.columns:
        df[col] = clean_boolean_column(df[col])

# Remove columns that contain no usable information after cleaning.
all_missing_columns = [
    col for col in df.columns
    if df[col].isna().all()
]

# Do not drop the platform columns merely because they are currently missing.
# Their source values need to be preserved for auditing. If a source column
# is genuinely all-missing, it will be documented rather than fabricated.
safe_drop_columns = [
    col for col in all_missing_columns
    if col not in platform_columns
]

df = df.drop(columns=safe_drop_columns)

# Identify post-release outcome fields that are intentionally retained
# because Step 3 must keep them available for target creation in Step 4.
post_release_outcome_columns = [
    "Estimated owners",
    "Peak CCU",
    "Reviews",
    "Positive",
    "Negative",
    "Recommendations",
    "Average playtime two weeks",
    "Median playtime forever",
    "Median playtime two weeks",
]

post_release_outcome_columns = [
    col for col in post_release_outcome_columns
    if col in df.columns
]

# Verify platform preservation after cleaning.
platform_after = {}
for col in platform_columns:
    if col in df.columns:
        platform_after[col] = {
            "true": int((df[col] == True).sum()),
            "false": int((df[col] == False).sum()),
            "missing": int(df[col].isna().sum()),
        }

# Save cleaned dataset.
df.to_csv(CLEANED_FILE, index=False)

# Write an audit-friendly cleaning summary.
summary_lines = [
    f"Rows before cleaning: {rows_before}",
    f"Columns before cleaning: {cols_before}",
    f"Completely duplicated rows removed: {duplicate_rows}",
    f"Rows removed because AppID was missing: {missing_appid}",
    f"Rows removed because AppID was invalid: {invalid_appid}",
    f"Duplicate AppID rows removed: {duplicate_appids}",
    f"Rows removed because Name was missing: {missing_name}",
    f"Invalid release dates after parsing: {invalid_release_dates}",
    f"All-missing columns removed: {safe_drop_columns}",
    f"Post-release outcome columns retained for target creation: {post_release_outcome_columns}",
    "",
    "Platform values after cleaning:",
]

for col in platform_columns:
    if col in platform_after:
        info = platform_after[col]
        summary_lines.append(
            f"{col}: True={info['true']}, False={info['false']}, Missing={info['missing']}"
        )
    else:
        summary_lines.append(f"{col}: column not present")

summary_lines.extend([
    "",
    f"Rows after cleaning: {len(df)}",
    f"Columns after cleaning: {len(df.columns)}",
    "",
    f"Cleaned dataset saved to: {CLEANED_FILE}",
])

SUMMARY_FILE.write_text(
    "\n".join(summary_lines),
    encoding="utf-8"
)

print(f"Rows before cleaning: {rows_before}")
print(f"Columns before cleaning: {cols_before}")
print(f"Completely duplicated rows removed: {duplicate_rows}")
print(f"Rows removed because AppID was missing: {missing_appid}")
print(f"Rows removed because AppID was invalid: {invalid_appid}")
print(f"Duplicate AppID rows removed: {duplicate_appids}")
print(f"Rows removed because Name was missing: {missing_name}")
print(f"Invalid release dates after parsing: {invalid_release_dates}")
print(f"All-missing columns removed: {safe_drop_columns}")
print(
    "Post-release outcome columns retained for target creation:",
    post_release_outcome_columns
)

print("\nPlatform values after cleaning:")
for col in platform_columns:
    if col in platform_after:
        print(
            f"{col}: "
            f"True={platform_after[col]['true']}, "
            f"False={platform_after[col]['false']}, "
            f"Missing={platform_after[col]['missing']}"
        )

print(f"\nRows after cleaning: {len(df)}")
print(f"Columns after cleaning: {len(df.columns)}")
print(f"\nCleaned dataset saved to:\n{CLEANED_FILE}")
print(f"Cleaning summary saved to:\n{SUMMARY_FILE}")
