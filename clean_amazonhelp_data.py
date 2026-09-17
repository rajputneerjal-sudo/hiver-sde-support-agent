import pandas as pd
import os
import re

INPUT_FILE = "data/processed/amazonhelp_kb.csv"
OUTPUT_FILE = "data/processed/amazonhelp_clean.csv"

print("Loading AmazonHelp knowledge base...")

df = pd.read_csv(INPUT_FILE)

print(f"Original conversation pairs: {len(df):,}")


def clean_text(text):
    text = str(text)

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove escaped newlines
    text = text.replace("\\n", " ")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["customer_message"] = df["customer_message"].apply(clean_text)
df["amazonhelp_response"] = df["amazonhelp_response"].apply(clean_text)

# Remove empty messages
df = df[
    (df["customer_message"].str.len() >= 10)
    &
    (df["amazonhelp_response"].str.len() >= 10)
].copy()

# Remove exact duplicate customer-response pairs
df = df.drop_duplicates(
    subset=["customer_message", "amazonhelp_response"]
)

# Remove cases where customer and response are identical
df = df[
    df["customer_message"].str.lower()
    != df["amazonhelp_response"].str.lower()
]

os.makedirs("data/processed", exist_ok=True)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 70)
print("Cleaning completed")
print("=" * 70)

print(f"Clean conversation pairs: {len(df):,}")
print(f"Removed: {100232 - len(df):,}")
print(f"Saved to: {OUTPUT_FILE}")

print()
print("Sample cleaned conversations:")
print("-" * 70)

print(
    df[
        ["customer_message", "amazonhelp_response"]
    ]
    .head(10)
    .to_string(index=False)
)