import pandas as pd

INPUT_FILE = "data/processed/golden_set.csv"

VALID_INTENTS = {
    "delivery_issue",
    "refund_return",
    "payment_billing",
    "account_access",
    "product_problem",
    "order_management",
    "other"
}

print("Loading golden evaluation set...")

df = pd.read_csv(INPUT_FILE)

print()
print("=" * 70)
print("GOLDEN SET VALIDATION")
print("=" * 70)

print(f"Total rows: {len(df)}")

# Check blank labels
blank_labels = df["intent"].isna() | (
    df["intent"].astype(str).str.strip() == ""
)

print(f"Blank labels: {blank_labels.sum()}")

# Check invalid labels
clean_labels = (
    df["intent"]
    .fillna("")
    .astype(str)
    .str.strip()
)

invalid_labels = sorted(
    set(clean_labels) - VALID_INTENTS - {""}
)

print(f"Invalid labels: {len(invalid_labels)}")

if invalid_labels:
    print("Invalid values:")
    for label in invalid_labels:
        print(f"  - {label}")

# Count each intent
print()
print("INTENT DISTRIBUTION")
print("-" * 70)

counts = clean_labels.value_counts()

for intent in sorted(VALID_INTENTS):
    print(
        f"{intent:<25} {counts.get(intent, 0):>4}"
    )

# Overall result
print()
print("=" * 70)

if (
    len(df) == 203
    and blank_labels.sum() == 0
    and len(invalid_labels) == 0
):
    print("VALIDATION PASSED")
    print("All 203 golden examples have valid labels.")
else:
    print("VALIDATION FAILED")
    print("Please fix the issue before continuing.")