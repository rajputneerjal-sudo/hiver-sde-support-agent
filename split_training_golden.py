import pandas as pd
import os

INPUT_FILE = "data/processed/labeling_dataset.csv"

TRAIN_FILE = "data/processed/training_dataset.csv"
GOLDEN_FILE = "data/processed/golden_set.csv"

df = pd.read_csv(INPUT_FILE)

print(f"Total examples: {len(df):,}")

# Shuffle once with a fixed seed so the split is reproducible.
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Required golden evaluation size.
golden_size = 203

golden = df.iloc[:golden_size].copy()
train = df.iloc[golden_size:].copy()

# Keep the intent column empty.
golden["intent"] = ""
train["intent"] = ""

# Add a column explaining that labels will be manually reviewed.
golden["label_notes"] = ""
train["label_notes"] = ""

os.makedirs("data/processed", exist_ok=True)

golden.to_csv(GOLDEN_FILE, index=False)
train.to_csv(TRAIN_FILE, index=False)

print()
print("=" * 70)
print("TRAINING / GOLDEN SPLIT CREATED")
print("=" * 70)

print(f"Training examples: {len(train):,}")
print(f"Golden examples:   {len(golden):,}")

print()
print(f"Training file: {TRAIN_FILE}")
print(f"Golden file:   {GOLDEN_FILE}")