import pandas as pd

INPUT_FILE = "data/processed/amazonhelp_clean.csv"

df = pd.read_csv(INPUT_FILE)

themes = {
    "DELIVERY / SHIPPING": [
        "delivery", "delivered", "shipping", "ship",
        "package", "parcel", "courier", "usps",
        "tracking", "arrive", "arrival", "late", "missing"
    ],

    "REFUND / RETURN": [
        "refund", "return", "money back",
        "returning", "returned"
    ],

    "PAYMENT / BILLING": [
        "payment", "charged", "charge", "billing",
        "bill", "credit card", "debit card"
    ],

    "ACCOUNT / LOGIN": [
        "account", "login", "log in", "password",
        "locked", "sign in", "hacked"
    ],

    "PRODUCT PROBLEM": [
        "broken", "damaged", "defective",
        "not working", "doesn't work",
        "warranty"
    ],

    "ORDER PROBLEM": [
        "order", "ordered", "cancel",
        "cancelled", "canceled",
        "preorder", "pre-order"
    ]
}

print("=" * 80)
print("REAL AMAZONHELP EXAMPLES BY POSSIBLE INTENT")
print("=" * 80)

text = df["customer_message"].astype(str).str.lower()

for theme, keywords in themes.items():

    mask = text.apply(
        lambda x: any(keyword in x for keyword in keywords)
    )

    examples = df[mask].sample(
        n=min(10, mask.sum()),
        random_state=42
    )

    print()
    print("=" * 80)
    print(theme)
    print("=" * 80)

    for i, message in enumerate(
        examples["customer_message"],
        start=1
    ):
        print(f"{i}. {message}")