# Create a balanced modeling dataset by retaining all Success games and
# randomly sampling the same number of Not Success games.

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"D:\Steam_Games_Success_Prediction")

INPUT_FILE = PROJECT_ROOT / "data" / "cleaned" / "steam_games_targeted.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "steam_games_balanced.csv"
REPORT_FILE = PROJECT_ROOT / "results" / "metrics" / "balancing_summary.txt"

RANDOM_SEED = 42

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE, low_memory=False)

success_df = df[df["Success"] == 1].copy()
not_success_df = df[df["Success"] == 0].copy()

success_count = len(success_df)
not_success_count = len(not_success_df)

sample_size = min(success_count, not_success_count)

# Keep every observation from the smaller class and randomly sample the
# larger class. The fixed seed makes the selection reproducible.
if success_count <= not_success_count:
    selected_success = success_df
    selected_not_success = not_success_df.sample(
        n=sample_size,
        random_state=RANDOM_SEED
    )
else:
    selected_success = success_df.sample(
        n=sample_size,
        random_state=RANDOM_SEED
    )
    selected_not_success = not_success_df

balanced_df = pd.concat(
    [selected_success, selected_not_success],
    ignore_index=True
)

balanced_df = balanced_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

balanced_df.to_csv(OUTPUT_FILE, index=False)

final_success = int((balanced_df["Success"] == 1).sum())
final_not_success = int((balanced_df["Success"] == 0).sum())

report = [
    "Dataset Balancing Summary",
    "=========================",
    f"Random seed: {RANDOM_SEED}",
    "",
    f"Success available: {success_count}",
    f"Not Success available: {not_success_count}",
    f"Rows selected per class: {sample_size}",
    "",
    f"Final rows: {len(balanced_df)}",
    f"Final Success: {final_success}",
    f"Final Not Success: {final_not_success}",
    "",
    "Balancing is performed after target definition.",
    "The target threshold itself is not changed to force class balance.",
    "",
    f"Balanced dataset saved to: {OUTPUT_FILE}",
]

REPORT_FILE.write_text("\n".join(report), encoding="utf-8")

print(f"Success available: {success_count}")
print(f"Not Success available: {not_success_count}")
print(f"Rows selected per class: {sample_size}")
print(f"Final rows: {len(balanced_df)}")
print(f"Final Success: {final_success}")
print(f"Final Not Success: {final_not_success}")
print(f"\nBalanced dataset saved to:\n{OUTPUT_FILE}")
