from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "steam_games.csv"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
CLEANED_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = CLEANED_DIR / "steam_games_cleaned.csv"
SUMMARY_FILE = METRICS_DIR / "cleaning_summary.txt"


def main():
    print("=" * 65)
    print("STEAM GAMES DATASET - CLEANING")
    print("=" * 65)

    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_FILE}")

    print(f"Reading: {RAW_FILE}")
    print("Loading dataset; this may take a few minutes...")
    df = pd.read_csv(RAW_FILE, index_col=False, low_memory=False)

    rows_before = len(df)
    columns_before = len(df.columns)
    summary = []

    def log(message=""):
        message = str(message)
        print(message)
        summary.append(message)

    log("\n1. INITIAL DATASET")
    log(f"Rows: {rows_before:,}")
    log(f"Columns: {columns_before:,}")

    # Standardize column labels only by removing surrounding whitespace.
    df.columns = [str(column).strip() for column in df.columns]

    # Remove fully duplicated records, if any.
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        df = df.drop_duplicates().copy()
    log(f"Completely duplicated rows removed: {duplicate_rows:,}")

    # Validate AppID without treating the identifier as a predictive feature here.
    missing_appid = 0
    invalid_appid = 0
    duplicate_appid_rows = 0
    if "AppID" in df.columns:
        missing_appid = int(df["AppID"].isna().sum())
        numeric_appid = pd.to_numeric(df["AppID"], errors="coerce")
        invalid_appid = int((numeric_appid.isna() & df["AppID"].notna()).sum())
        df = df.loc[numeric_appid.notna()].copy()
        df["AppID"] = numeric_appid.loc[df.index].astype("int64")
        duplicate_appid_rows = int(df["AppID"].duplicated().sum())
        if duplicate_appid_rows:
            df = df.drop_duplicates(subset=["AppID"], keep="first").copy()
    log(f"Missing AppID values found: {missing_appid:,}")
    log(f"Invalid AppID values removed: {invalid_appid:,}")
    log(f"Duplicate AppID rows removed: {duplicate_appid_rows:,}")

    # A game name is required to identify the record meaningfully.
    missing_names = 0
    if "Name" in df.columns:
        missing_names = int(df["Name"].isna().sum())
        df = df.loc[df["Name"].notna()].copy()
        df = df.loc[df["Name"].astype(str).str.strip().ne("")].copy()
    else:
        raise KeyError("Required column 'Name' was not found in the dataset.")
    log(f"Rows with missing/blank Name removed: {missing_names:,}")

    # Validate release dates. Keep the original field as text to avoid silently
    # changing its representation; later scripts can parse it explicitly.
    invalid_release_dates = 0
    if "Release date" in df.columns:
        parsed_dates = pd.to_datetime(df["Release date"], errors="coerce", format="mixed")
        invalid_release_dates = int(parsed_dates.isna().sum())
        # Do not drop rows here based solely on parsing: retain the evidence and
        # report the count for a deliberate downstream decision.
    log(f"Release dates that could not be parsed (reported, not removed): {invalid_release_dates:,}")

    # Do not automatically remove high-missingness columns or coerce ambiguous
    # fields such as 'About the game' / 'Metacritic score'. Record missingness
    # for downstream feature-selection decisions instead.
    log("\n2. COLUMN HANDLING")
    log("No columns were removed automatically.")
    log("Ambiguous column data types were retained for explicit downstream inspection.")

    rows_after = len(df)
    log("\n3. CLEANING SUMMARY")
    log(f"Rows before cleaning: {rows_before:,}")
    log(f"Rows after cleaning: {rows_after:,}")
    log(f"Rows removed overall: {rows_before - rows_after:,}")
    log(f"Columns retained: {len(df.columns):,}")
    log(f"Output file: {OUTPUT_FILE}")

    print("\nWriting cleaned dataset. This may take a few minutes...")
    df.to_csv(OUTPUT_FILE, index=False)
    SUMMARY_FILE.write_text("\n".join(summary), encoding="utf-8")

    print("\nCleaning completed successfully.")
    print(f"Cleaned dataset: {OUTPUT_FILE}")
    print(f"Cleaning summary: {SUMMARY_FILE}")
    print("The original raw CSV was not modified.")


if __name__ == "__main__":
    main()
