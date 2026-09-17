import pandas as pd
import os

INPUT_FILE = "data/processed/training_dataset.csv"
OUTPUT_FILE = "data/processed/training_labeling_sample.csv"

df = pd.read_csv(INPUT_FILE)

print(f"Available training examples: {len(df):,}")

# Create a reproducible random sample for manual labelling.
sample_size = min(800, len(df))

sample = df.sample(
    n=sample_size,
    random_state=123
).reset_index(drop=True)

# Remove any previous labels.
sample["intent"] = ""
sample["label_notes"] = ""

os.makedirs("data/processed", exist_ok=True)

sample.to_csv(OUTPUT_FILE, index=False)

print()
print("=" * 70)
print("TRAINING LABELING SAMPLE CREATED")
print("=" * 70)

print(f"Examples selected: {len(sample):,}")
print(f"Output file: {OUTPUT_FILE}")