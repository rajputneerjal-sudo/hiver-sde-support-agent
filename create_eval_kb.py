from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

KB_PATH = DATA_DIR / "amazonhelp_kb.csv"
GOLDEN_PATH = DATA_DIR / "golden_set.csv"
OUTPUT_PATH = DATA_DIR / "kb_for_eval.csv"


kb = pd.read_csv(KB_PATH)
golden = pd.read_csv(GOLDEN_PATH)

print(f"Original KB rows: {len(kb):,}")
print(f"Golden rows: {len(golden):,}")

if "customer_tweet_id" not in kb.columns:
    raise ValueError("KB is missing customer_tweet_id column.")

if "customer_tweet_id" not in golden.columns:
    raise ValueError("Golden set is missing customer_tweet_id column.")

golden_ids = set(
    golden["customer_tweet_id"]
    .dropna()
    .astype(str)
)

kb_ids = kb["customer_tweet_id"].astype(str)

kb_for_eval = kb[
    ~kb_ids.isin(golden_ids)
].copy()

kb_for_eval.to_csv(
    OUTPUT_PATH,
    index=False,
)

print(f"Evaluation KB rows: {len(kb_for_eval):,}")
print(f"Removed rows: {len(kb) - len(kb_for_eval):,}")
print(f"Saved to: {OUTPUT_PATH}")