
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
file_path = root / "data" / "cleaned" / "steam_games_targeted.csv"

print("Loading sample rows...", flush=True)

df = pd.read_csv(file_path, nrows=20, low_memory=False)

print("\nDataset columns:", flush=True)

for column in ["Genres", "Categories", "Tags"]:
    print(f"\n--- {column} ---", flush=True)

    if column not in df.columns:
        print("Column not found", flush=True)
        continue

    for value in df[column].dropna().head(10):
        print(repr(value)[:300], flush=True)

print("\nInspection completed.", flush=True)
