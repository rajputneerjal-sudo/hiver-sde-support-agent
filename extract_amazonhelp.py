import pandas as pd
import os

INPUT_FILE = "data/raw/twcs.csv"
OUTPUT_FILE = "data/raw/amazonhelp_tweets.csv"

print("Reading Twitter Support dataset...")

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

print(f"Total tweets loaded: {len(df):,}")

# Find tweets that mention AmazonHelp
amazon = df[
    df["text"]
    .astype(str)
    .str.contains("@AmazonHelp", case=False, na=False)
].copy()

print(f"Tweets mentioning @AmazonHelp: {len(amazon):,}")

# Save them
os.makedirs("data/raw", exist_ok=True)

amazon.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"Saved to: {OUTPUT_FILE}")