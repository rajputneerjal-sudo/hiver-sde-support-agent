import pandas as pd
import os

INPUT_FILE = "data/processed/training_dataset.csv"
OUTPUT_FILE = "data/processed/training_labeled.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("CREATING IMPROVED TRAINING LABELS")
print("=" * 70)

print(f"Training examples loaded: {len(df):,}")


def has_any(text, phrases):
    text = str(text).lower()
    return any(phrase in text for phrase in phrases)


def assign_intent(text):

    text = str(text).lower()

    # ---------------------------------------------------------
    # 1. ACCOUNT / ACCESS
    # ---------------------------------------------------------
    if has_any(text, [
        "can't login",
        "cannot login",
        "can't log in",
        "cannot log in",
        "can't sign in",
        "cannot sign in",
        "unable to login",
        "unable to log in",
        "password",
        "account locked",
        "locked out",
        "hacked account",
        "account hacked",
        "verify my account",
        "verification code",
        "verification problem"
    ]):
        return "account_access"

    # ---------------------------------------------------------
    # 2. PAYMENT / BILLING
    # ---------------------------------------------------------
    if has_any(text, [
        "charged twice",
        "charged two times",
        "double charged",
        "duplicate charge",
        "wrong charge",
        "wrongly charged",
        "unexpected charge",
        "credit card charged",
        "debit card charged",
        "payment failed",
        "payment problem",
        "payment issue",
        "billing problem",
        "billing issue",
        "refund to my card"
    ]):
        return "payment_billing"

    # ---------------------------------------------------------
    # 3. REFUND / RETURN
    # ---------------------------------------------------------
    if has_any(text, [
        "want a refund",
        "need a refund",
        "get a refund",
        "where is my refund",
        "when will i get my refund",
        "refund status",
        "refund hasn't",
        "refund hasnt",
        "money back",
        "return my",
        "return this",
        "return item",
        "return label",
        "returning",
        "send it back",
        "send back"
    ]):
        return "refund_return"

    # ---------------------------------------------------------
    # 4. PRODUCT PROBLEM
    # ---------------------------------------------------------
    if has_any(text, [
        "product is broken",
        "product was broken",
        "item is broken",
        "item was broken",
        "arrived broken",
        "arrived damaged",
        "product damaged",
        "item damaged",
        "defective product",
        "defective item",
        "faulty product",
        "faulty item",
        "doesn't work",
        "doesnt work",
        "not working",
        "won't work",
        "wont work",
        "malfunction",
        "product problem",
        "product issue",
        "wrong item",
        "wrong product"
    ]):
        return "product_problem"

    # ---------------------------------------------------------
    # 5. ORDER MANAGEMENT
    # ---------------------------------------------------------
    if has_any(text, [
        "cancel my order",
        "cancel order",
        "cancel the order",
        "cancelled my order",
        "canceled my order",
        "order cancellation",
        "change my order",
        "change the order",
        "modify my order",
        "edit my order",
        "update my order",
        "change order",
        "replace my order",
        "replacement order",
        "reorder",
        "re-order",
        "wrong order",
        "order number",
        "order details"
    ]):
        return "order_management"

    # ---------------------------------------------------------
    # 6. DELIVERY / SHIPPING
    # ---------------------------------------------------------
    if has_any(text, [
        "where is my package",
        "where is my parcel",
        "package hasn't arrived",
        "package hasnt arrived",
        "parcel hasn't arrived",
        "parcel hasnt arrived",
        "package is late",
        "delivery is late",
        "late delivery",
        "delivery delayed",
        "delivery delay",
        "shipping delay",
        "shipping delayed",
        "tracking number",
        "tracking says",
        "tracking hasn't",
        "tracking hasnt",
        "package tracking",
        "delivery date",
        "delivered but",
        "marked delivered",
        "carrier",
        "courier",
        "usps",
        "ups",
        "fedex",
        "package",
        "parcel",
        "delivery",
        "shipping",
        "shipped"
    ]):
        return "delivery_issue"

    # ---------------------------------------------------------
    # 7. OTHER
    # ---------------------------------------------------------
    return "other"


df["intent"] = df["customer_message"].apply(assign_intent)

df["label_source"] = "weak_keyword_rule_v2"

os.makedirs("data/processed", exist_ok=True)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 70)
print("IMPROVED LABEL DISTRIBUTION")
print("=" * 70)

counts = df["intent"].value_counts()

for intent in [
    "delivery_issue",
    "refund_return",
    "payment_billing",
    "account_access",
    "product_problem",
    "order_management",
    "other"
]:
    print(
        f"{intent:<25} {counts.get(intent, 0):>5}"
    )

print()
print("=" * 70)
print("IMPROVED WEAK LABELING COMPLETE")
print("=" * 70)

print(f"Labeled examples: {len(df):,}")
print(f"Saved to: {OUTPUT_FILE}")