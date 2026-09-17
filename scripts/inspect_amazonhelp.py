import pandas as pd

INPUT_FILE = "data/processed/amazonhelp_clean.csv"

print("Loading cleaned AmazonHelp data...")

df = pd.read_csv(INPUT_FILE)

print(f"Total clean conversations: {len(df):,}")
print()

print("=" * 80)
print("SAMPLE REAL AMAZONHELP CUSTOMER MESSAGES")
print("=" * 80)

# Show 100 random real customer messages
sample = df.sample(
    n=min(100, len(df)),
    random_state=42
)

for i, text in enumerate(sample["customer_message"], start=1):
    print()
    print(f"{i}. {text}")

print()
print("=" * 80)
print("SAMPLE AMAZONHELP RESPONSES")
print("=" * 80)

for i, text in enumerate(
    sample["amazonhelp_response"].head(20),
    start=1
):
    print()
    print(f"{i}. {text}")