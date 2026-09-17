import pandas as pd
import re

INPUT_FILE = "data/processed/amazonhelp_clean.csv"

df = pd.read_csv(INPUT_FILE)

text = df["customer_message"].astype(str).str.lower()

themes = {
    "delivery / shipping": [
        "delivery", "delivered", "shipping", "ship", "package",
        "parcel", "courier", "usps", "tracking", "arrive", "arrival",
        "late", "missing"
    ],

    "refund / return": [
        "refund", "return", "money back", "returning", "returned"
    ],

    "payment / billing": [
        "payment", "charged", "charge", "billing", "bill",
        "credit card", "debit card", "price"
    ],

    "account / login": [
        "account", "login", "log in", "password", "locked",
        "sign in", "hacked"
    ],

    "product problem": [
        "broken", "damaged", "defective", "not working",
        "doesn't work", "warranty", "product", "device"
    ],

    "order problem": [
        "order", "ordered", "cancel", "cancelled", "canceled",
        "preorder", "pre-order"
    ],

    "customer service": [
        "customer service", "support", "agent", "representative",
        "call", "phone", "chat", "help"
    ]
}

print("=" * 70)
print("PRELIMINARY THEMES IN REAL AMAZONHELP DATA")
print("=" * 70)

for theme, keywords in themes.items():

    pattern = "|".join(
        re.escape(keyword)
        for keyword in keywords
    )

    count = text.str.contains(
        pattern,
        regex=True,
        na=False
    ).sum()

    percentage = count / len(df) * 100

    print(
        f"{theme:<25} {count:>8,} messages "
        f"({percentage:.2f}%)"
    )

print()
print("=" * 70)
print("IMPORTANT")
print("=" * 70)
print("These are only preliminary keyword counts.")
print("We will manually review real examples before finalizing intents.")