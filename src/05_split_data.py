# Split the balanced dataset into stratified train, validation, and test sets.

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(r"D:\Steam_Games_Success_Prediction")

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "steam_games_balanced.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_FILE = PROJECT_ROOT / "results" / "metrics" / "split_summary.txt"

RANDOM_SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE, low_memory=False)

# First split: 80% temporary data and 20% final test data.
train_val, test = train_test_split(
    df,
    test_size=0.20,
    stratify=df["Success"],
    random_state=RANDOM_SEED
)

# Second split: 25% of the remaining 80% becomes validation.
# This produces 60% train, 20% validation, and 20% test overall.
train, validation = train_test_split(
    train_val,
    test_size=0.25,
    stratify=train_val["Success"],
    random_state=RANDOM_SEED
)

train = train.reset_index(drop=True)
validation = validation.reset_index(drop=True)
test = test.reset_index(drop=True)

train_file = OUTPUT_DIR / "train.csv"
validation_file = OUTPUT_DIR / "validation.csv"
test_file = OUTPUT_DIR / "test.csv"

train.to_csv(train_file, index=False)
validation.to_csv(validation_file, index=False)
test.to_csv(test_file, index=False)

def class_summary(data):
    counts = data["Success"].value_counts().sort_index()
    total = len(data)
    return {
        "Not Success": (
            int(counts.get(0, 0)),
            float(counts.get(0, 0) / total * 100)
        ),
        "Success": (
            int(counts.get(1, 0)),
            float(counts.get(1, 0) / total * 100)
        ),
    }

train_summary = class_summary(train)
validation_summary = class_summary(validation)
test_summary = class_summary(test)

report = [
    "Train / Validation / Test Split Summary",
    "========================================",
    f"Random seed: {RANDOM_SEED}",
    "",
    "Split ratio:",
    "Train = 60%",
    "Validation = 20%",
    "Test = 20%",
    "",
    f"Train rows: {len(train)}",
    f"Validation rows: {len(validation)}",
    f"Test rows: {len(test)}",
    "",
    "Train class distribution:",
    f"Not Success: {train_summary['Not Success'][0]} ({train_summary['Not Success'][1]:.2f}%)",
    f"Success: {train_summary['Success'][0]} ({train_summary['Success'][1]:.2f}%)",
    "",
    "Validation class distribution:",
    f"Not Success: {validation_summary['Not Success'][0]} ({validation_summary['Not Success'][1]:.2f}%)",
    f"Success: {validation_summary['Success'][0]} ({validation_summary['Success'][1]:.2f}%)",
    "",
    "Test class distribution:",
    f"Not Success: {test_summary['Not Success'][0]} ({test_summary['Not Success'][1]:.2f}%)",
    f"Success: {test_summary['Success'][0]} ({test_summary['Success'][1]:.2f}%)",
    "",
    "The test set is held out and is not used during preprocessing, tuning,",
    "threshold selection, or model selection.",
]

REPORT_FILE.write_text("\n".join(report), encoding="utf-8")

print(f"Train rows: {len(train)}")
print(f"Validation rows: {len(validation)}")
print(f"Test rows: {len(test)}")
print("\nTrain class distribution:")
print(train["Success"].value_counts().sort_index())
print("\nValidation class distribution:")
print(validation["Success"].value_counts().sort_index())
print("\nTest class distribution:")
print(test["Success"].value_counts().sort_index())
print(f"\nFiles saved to:\n{OUTPUT_DIR}")
