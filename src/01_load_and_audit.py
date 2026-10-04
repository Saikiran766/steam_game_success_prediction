# Import the libraries required for the dataset audit

import pandas as pd
from pathlib import Path

# Define the project paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "steam_games.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Check whether the raw dataset exists

if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"Raw dataset not found:\n{RAW_FILE}"
    )

print("Dataset found:")
print(RAW_FILE)

# Load the raw Steam games dataset without allowing pandas to infer an index

df = pd.read_csv(
    RAW_FILE,
    index_col=False,
    low_memory=False
)

print("Dataset loaded successfully.")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# Verify that the first column is being read as AppID

print("\nInitial parsing check:")
print(f"First column name: {df.columns[0]}")
print(f"First row AppID: {df.iloc[0, 0]}")
print(f"First row Name: {df.iloc[0, 1]}")
print(f"DataFrame index type: {type(df.index).__name__}")

if df.columns[0] != "AppID":
    raise ValueError(
        "CSV parsing is still incorrect: the first column is not AppID."
    )

# Display the column names

print("\nColumns in the dataset:")
for number, column in enumerate(df.columns, start=1):
    print(f"{number}. {column}")

# Display data types

print("\nData types:")
print(df.dtypes.to_string())

# Display memory usage

memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
print(f"\nApproximate memory usage: {memory_mb:.2f} MB")

# Display missing-value summary

missing = df.isna().sum()
missing_percentage = (missing / len(df)) * 100

missing_summary = pd.DataFrame({
    "missing_count": missing,
    "missing_percentage": missing_percentage
})

missing_summary = missing_summary.sort_values(
    "missing_count",
    ascending=False
)

print("\nMissing value summary:")
print(missing_summary.to_string())

# Check duplicate rows and duplicate AppID values

duplicate_rows = df.duplicated().sum()
print(f"\nCompletely duplicated rows: {duplicate_rows}")

if "AppID" in df.columns:
    duplicate_appids = df["AppID"].duplicated().sum()
    print(f"Duplicate AppID values: {duplicate_appids}")
else:
    print("AppID column not found.")

# Count unique values in each column

unique_summary = (
    df.nunique(dropna=False)
    .sort_values(ascending=False)
    .to_frame("unique_values")
)

print("\nUnique values per column:")
print(unique_summary.to_string())

# Identify numeric and categorical columns

numeric_columns = df.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()

print("\nNumeric columns:")
print(numeric_columns)

print("\nCategorical/text columns:")
print(categorical_columns)

# Display the first five rows

print("\nFirst five rows:")
print(df.head().to_string())

# Display the statistical summary

print("\nBasic statistical summary:")
print(df.describe(include="all").transpose().to_string())

# Save the audit report

report_file = RESULTS_DIR / "data_audit.txt"

with open(report_file, "w", encoding="utf-8") as file:
    file.write("STEAM GAMES DATASET AUDIT REPORT\n")
    file.write("=" * 80 + "\n\n")

    file.write(f"Dataset path: {RAW_FILE}\n")
    file.write(f"Number of rows: {df.shape[0]}\n")
    file.write(f"Number of columns: {df.shape[1]}\n")
    file.write(f"Approximate memory usage: {memory_mb:.2f} MB\n\n")

    file.write("COLUMNS\n")
    file.write("-" * 80 + "\n")
    for number, column in enumerate(df.columns, start=1):
        file.write(f"{number}. {column}\n")

    file.write("\nDATA TYPES\n")
    file.write("-" * 80 + "\n")
    file.write(df.dtypes.to_string())

    file.write("\n\nMISSING VALUE SUMMARY\n")
    file.write("-" * 80 + "\n")
    file.write(missing_summary.to_string())

    file.write("\n\nDUPLICATES\n")
    file.write("-" * 80 + "\n")
    file.write(f"Completely duplicated rows: {duplicate_rows}\n")
    if "AppID" in df.columns:
        file.write(f"Duplicate AppID values: {df['AppID'].duplicated().sum()}\n")

    file.write("\n\nUNIQUE VALUES\n")
    file.write("-" * 80 + "\n")
    file.write(unique_summary.to_string())

    file.write("\n\nNUMERIC COLUMNS\n")
    file.write("-" * 80 + "\n")
    file.write(str(numeric_columns))

    file.write("\n\nCATEGORICAL/TEXT COLUMNS\n")
    file.write("-" * 80 + "\n")
    file.write(str(categorical_columns))

    file.write("\n\nFIRST FIVE ROWS\n")
    file.write("-" * 80 + "\n")
    file.write(df.head().to_string())

    file.write("\n\nSTATISTICAL SUMMARY\n")
    file.write("-" * 80 + "\n")
    file.write(df.describe(include="all").transpose().to_string())

print("\nAudit report saved to:")
print(report_file)
