import pandas as pd
import os

INPUT_FILE = "data/processed/amazonhelp_clean.csv"
OUTPUT_FILE = "data/processed/labeling_dataset.csv"

df = pd.read_csv(INPUT_FILE)

# Remove duplicate customer messages
df = df.drop_duplicates(
    subset=["customer_message"]
).copy()

# We want a manageable dataset for manual labeling.
# Sample randomly from the real AmazonHelp conversations.
sample_size = min(1600, len(df))

sample = df.sample(
    n=sample_size,
    random_state=42
).copy()

# Create empty label columns.
sample["intent"] = ""
sample["label_notes"] = ""

# Keep only the columns needed for labeling.
sample = sample[
    [
        "customer_tweet_id",
        "customer_message",
        "amazonhelp_response",
        "intent",
        "label_notes"
    ]
]

os.makedirs("data/processed", exist_ok=True)

sample.to_csv(
    OUTPUT_FILE,
    index=False
)

print("=" * 70)
print("REAL AMAZONHELP LABELING DATASET CREATED")
print("=" * 70)

print(f"Examples: {len(sample):,}")
print(f"Saved to: {OUTPUT_FILE}")

print()
print("Intents to use:")
print("1. delivery_issue")
print("2. refund_return")
print("3. payment_billing")
print("4. account_access")
print("5. product_problem")
print("6. order_management")
print("7. other")