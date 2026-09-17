import pandas as pd
import os

INPUT_FILE = "data/raw/twcs.csv"
OUTPUT_FILE = "data/processed/amazonhelp_kb.csv"

print("Loading Twitter Support dataset...")

df = pd.read_csv(
    INPUT_FILE,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ]
)

print(f"Loaded {len(df):,} tweets")

# Make tweet IDs strings
df["tweet_id"] = df["tweet_id"].astype(str)

# Clean response ID columns
df["response_tweet_id"] = (
    df["response_tweet_id"]
    .fillna("")
    .astype(str)
)

# Create tweet ID -> row lookup
tweet_lookup = df.set_index("tweet_id")

records = []

print("Finding real AmazonHelp customer -> response pairs...")

# Customer tweets:
# inbound == True
# and the customer mentions AmazonHelp
customers = df[
    (df["inbound"] == True)
    &
    (
        df["text"]
        .astype(str)
        .str.contains("@AmazonHelp", case=False, na=False)
    )
].copy()

print(f"AmazonHelp customer tweets found: {len(customers):,}")

for _, customer in customers.iterrows():

    customer_text = str(customer["text"])

    response_ids = str(customer["response_tweet_id"]).strip()

    if not response_ids or response_ids == "nan":
        continue

    # response_tweet_id can contain multiple IDs
    ids = [
        x.strip()
        for x in response_ids.split(",")
        if x.strip()
    ]

    for response_id in ids:

        if response_id not in tweet_lookup.index:
            continue

        response = tweet_lookup.loc[response_id]

        # We only want outbound/support responses
        if response["inbound"] != False:
            continue

        response_text = str(response["text"])

        records.append({
            "customer_tweet_id": customer["tweet_id"],
            "response_tweet_id": response_id,
            "customer_message": customer_text,
            "amazonhelp_response": response_text,
            "customer_created_at": customer["created_at"],
            "response_created_at": response["created_at"]
        })

kb = pd.DataFrame(records)

os.makedirs("data/processed", exist_ok=True)

kb.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 70)
print("AmazonHelp historical knowledge base created")
print("=" * 70)
print(f"Conversation pairs: {len(kb):,}")
print(f"Saved to: {OUTPUT_FILE}")

if len(kb) > 0:
    print()
    print("Sample real conversations:")
    print("-" * 70)

    print(
        kb[
            [
                "customer_message",
                "amazonhelp_response"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
else:
    print()
    print("WARNING: No conversation pairs were found.")