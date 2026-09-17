from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ============================================================
# INTENT TAXONOMY
# ============================================================

INTENTS = [
    "delivery_issue",
    "refund_return",
    "payment_billing",
    "account_access",
    "product_problem",
    "order_management",
    "other",
]


# ============================================================
# TRIVIAL BASELINE
# ============================================================

class TrivialClassifier:
    """Simple majority-class baseline."""

    def fit(self, texts, labels):
        self.majority_ = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, texts):
        return [self.majority_] * len(texts)


# ============================================================
# PREDICTION RESULT
# ============================================================

@dataclass
class MLPrediction:
    intent: str
    confidence: float


# ============================================================
# MAIN ML CLASSIFIER
# ============================================================

class MLIntentClassifier:
    """
    TF-IDF + Logistic Regression intent classifier.

    A small set of high-precision routing rules is applied
    before the ML classifier for clearly identifiable cases.
    """

    def __init__(self):
        self.vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            lowercase=True,
            token_pattern=r"(?u)\b\w[\w']+\b",
        )

        self.clf = LogisticRegression(
            max_iter=1000,
            C=4.0,
            class_weight="balanced",
        )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    def fit(self, texts, labels):
        X = self.vec.fit_transform(texts)
        self.clf.fit(X, labels)
        return self

    # --------------------------------------------------------
    # NORMAL ML PREDICTION
    # --------------------------------------------------------

    def predict(self, texts):
        X = self.vec.transform(texts)
        return list(self.clf.predict(X))

    # --------------------------------------------------------
    # SINGLE MESSAGE PREDICTION
    # --------------------------------------------------------

    def predict_proba_one(self, text: str) -> MLPrediction:

        text_lower = str(text).lower().strip()

        # ====================================================
        # 1. ACCOUNT ACCESS
        # ====================================================

        account_terms = [
            "locked out",
            "account locked",
            "account suspended",
            "account disabled",
            "can't log in",
            "cannot log in",
            "can't login",
            "cannot login",
            "unable to log in",
            "unable to login",
            "login",
            "log in",
            "sign in",
            "signin",
            "password",
            "forgot password",
            "forgot my password",
            "password reset",
            "reset my password",
            "verification code",
            "otp",
            "2fa",
            "two factor",
            "verify my account",
        ]

        # ====================================================
        # 2. PAYMENT / BILLING
        # ====================================================

        payment_terms = [
            "charged twice",
            "charged multiple times",
            "charged extra",
            "wrong charge",
            "incorrect charge",
            "duplicate charge",
            "duplicate payment",
            "payment issue",
            "payment problem",
            "billing",
            "billed twice",
            "credit card charged",
            "debit card charged",
            "promotional credit",
            "invoice",
            "payment failed",
            "revise payment",
            "bank account",
        ]

        # ====================================================
        # 3. REFUND / RETURN
        # ====================================================

        refund_terms = [
            "refund",
            "want a refund",
            "need a refund",
            "waiting for refund",
            "refund status",
            "money back",
            "return policy",
            "return label",
            "return request",
            "return my item",
            "return my product",
            "return this item",
            "return this product",
            "returned item",
            "returned product",
        ]

        # ====================================================
        # 4. PRODUCT PROBLEM
        # ====================================================

        product_terms = [
            "damaged",
            "broken",
            "faulty",
            "defective",
            "duplicate product",
            "fake product",
            "counterfeit",
            "wrong product",
            "incorrect product",
            "product information",
            "pack of",
            "empty package",
            "missing item",
            "replacement",
            "replace it",
            "replace my item",
            "exchange",
            "not working",
            "stopped working",
            "doesn't work",
            "does not work",
            "freezes",
            "freezing",
            "malfunction",
            "malfunctioning",
            "feature unavailable",
            "app support unavailable",
            "spotify on fire tv",
        ]

        # ====================================================
        # 5. DELIVERY ISSUE
        # ====================================================

        delivery_terms = [
            "where is my order",
            "where's my order",
            "where is my package",
            "where's my package",
            "where is my parcel",
            "where's my parcel",
            "not delivered",
            "not received",
            "haven't received",
            "have not received",
            "didn't receive",
            "did not receive",
            "order hasn't arrived",
            "order has not arrived",
            "package hasn't arrived",
            "package has not arrived",
            "parcel hasn't arrived",
            "delivery delayed",
            "delivery is late",
            "expected delivery",
            "tracking number",
            "tracking my order",
            "track my package",
            "track my order",
            "warehouse",
            "courier",
            "driver",
        ]

        # ====================================================
        # 6. ORDER MANAGEMENT
        # ====================================================

        order_terms = [
            "cancel the order",
            "cancel my order",
            "cancel order",
            "want to cancel",
            "need to cancel",
            "change my order",
            "change the order",
            "modify my order",
            "edit my order",
            "change order",
            "order change",
            "reorder",
            "re-order",
        ]

        # ====================================================
        # HIGH-PRIORITY ROUTING
        # ====================================================

        # ----------------------------------------------------
        # ACCOUNT
        # ----------------------------------------------------

        if any(term in text_lower for term in account_terms):
            return MLPrediction(
                intent="account_access",
                confidence=0.99,
            )

        # ----------------------------------------------------
        # PAYMENT / BILLING
        # ----------------------------------------------------

        if any(term in text_lower for term in payment_terms):
            return MLPrediction(
                intent="payment_billing",
                confidence=0.99,
            )

        # ----------------------------------------------------
        # REFUND / RETURN
        # ----------------------------------------------------

        if any(term in text_lower for term in refund_terms):
            return MLPrediction(
                intent="refund_return",
                confidence=0.99,
            )

        # ----------------------------------------------------
        # PRODUCT PROBLEM
        # ----------------------------------------------------

        if any(term in text_lower for term in product_terms):

            # A message explicitly asking to cancel should remain
            # order_management rather than product_problem.
            if any(term in text_lower for term in order_terms):
                return MLPrediction(
                    intent="order_management",
                    confidence=0.99,
                )

            return MLPrediction(
                intent="product_problem",
                confidence=0.99,
            )

        # ----------------------------------------------------
        # DELIVERY
        # ----------------------------------------------------

        if any(term in text_lower for term in delivery_terms):
            return MLPrediction(
                intent="delivery_issue",
                confidence=0.99,
            )

        # ----------------------------------------------------
        # ORDER MANAGEMENT
        # ----------------------------------------------------

        if any(term in text_lower for term in order_terms):
            return MLPrediction(
                intent="order_management",
                confidence=0.99,
            )

        # ====================================================
        # FALLBACK TO MACHINE LEARNING
        # ====================================================

        X = self.vec.transform([text])

        proba = self.clf.predict_proba(X)[0]

        idx = proba.argmax()

        return MLPrediction(
            intent=self.clf.classes_[idx],
            confidence=float(proba[idx]),
        )


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_ml_classifier(training_csv_path: str) -> MLIntentClassifier:
    """
    Load labelled training data and train the classifier.
    """

    df = pd.read_csv(training_csv_path)

    df = df[df["intent"].isin(INTENTS)].copy()

    if len(df) == 0:
        raise ValueError(
            "No valid labelled training examples found."
        )

    print(
        f"Training classifier on {len(df):,} labelled examples..."
    )

    clf = MLIntentClassifier()

    clf.fit(
        df["customer_message"].astype(str).tolist(),
        df["intent"].tolist(),
    )

    return clf