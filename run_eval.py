"""
Evaluation of the AmazonHelp support agent.

Evaluates:
1. Intent classification
2. Escalation policy
3. Reply quality

The classifier is trained only on training_labeled.csv.
The 203-example golden set is held out for evaluation.
The evaluation retriever uses kb_for_eval.csv, which excludes
customer IDs represented in the golden set.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.intents import (
    INTENTS,
    TrivialClassifier,
    train_ml_classifier,
)
from src.retrieval import ResolvedCaseRetriever
from src.reply_generator import draft_reply
from src.escalation import decide as system_decide

from eval.metrics import intent_metrics, escalation_metrics
from eval.llm_judge import score as judge_score


ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "processed"

TRAINING_PATH = DATA_DIR / "training_labeled.csv"
GOLDEN_PATH = DATA_DIR / "golden_set.csv"
KB_PATH = DATA_DIR / "kb_for_eval.csv"

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

PREDICTIONS_PATH = ROOT / "eval" / "system_predictions.csv"
RESULTS_JSON_PATH = REPORTS_DIR / "eval_results.json"
RESULTS_MD_PATH = REPORTS_DIR / "eval_results.md"


def load_golden_set():
    """
    Load the real hand-labelled golden set.

    Expected columns:
      customer_tweet_id
      customer_message
      amazonhelp_response
      intent
      label_notes
    """

    golden = pd.read_csv(GOLDEN_PATH)

    required_columns = {
        "customer_tweet_id",
        "customer_message",
        "intent",
    }

    missing = required_columns - set(golden.columns)

    if missing:
        raise ValueError(
            f"Golden set is missing columns: {sorted(missing)}"
        )

    golden = golden[
        golden["intent"].isin(INTENTS)
    ].copy()

    if len(golden) == 0:
        raise ValueError("Golden set contains no valid intent labels.")

    return golden


def simple_intent_rule(text: str) -> str:
    """
    Simple keyword-based intent baseline.

    This is intentionally simple and does not use machine learning.
    It predicts an intent using obvious keywords and phrases.
    """

    text = str(text).lower()

    # Account access / login
    if any(term in text for term in [
        "locked out",
        "cannot login",
        "can't login",
        "cannot log in",
        "can't log in",
        "unable to login",
        "unable to log in",
        "forgot my password",
        "reset my password",
        "password reset",
        "password",
        "account access",
        "account locked",
        "sign in",
        "signin",
        "login",
    ]):
        return "account_access"

    # Payment / billing
    if any(term in text for term in [
        "charged twice",
        "charged two times",
        "charged multiple times",
        "duplicate charge",
        "duplicate payment",
        "extra charge",
        "wrong charge",
        "incorrect charge",
        "billing",
        "billed twice",
        "payment issue",
        "payment problem",
        "payment",
        "charged",
        "charge",
    ]):
        return "payment_billing"

    # Refund / return
    if any(term in text for term in [
        "refund",
        "return the product",
        "return this product",
        "return my product",
        "return the item",
        "return this item",
        "return my item",
        "money back",
        "return label",
        "return",
    ]):
        return "refund_return"

    # Product problem
    if any(term in text for term in [
        "arrived damaged",
        "arrived broken",
        "item arrived damaged",
        "item arrived broken",
        "product arrived damaged",
        "product arrived broken",
        "item is damaged",
        "item was damaged",
        "product is damaged",
        "product was damaged",
        "damaged product",
        "damaged item",
        "broken product",
        "broken item",
        "item is broken",
        "item was broken",
        "product is broken",
        "product was broken",
        "defective product",
        "defective item",
        "faulty product",
        "faulty item",
        "product doesn't work",
        "product does not work",
        "item doesn't work",
        "item does not work",
        "not working",
        "stopped working",
        "malfunction",
        "malfunctioning",
    ]):
        return "product_problem"

    # Delivery
    if any(term in text for term in [
        "where is my order",
        "where is my package",
        "where is my parcel",
        "where's my order",
        "where's my package",
        "where's my parcel",
        "order has not arrived",
        "order hasn't arrived",
        "package has not arrived",
        "package hasn't arrived",
        "parcel has not arrived",
        "parcel hasn't arrived",
        "order is late",
        "package is late",
        "parcel is late",
        "delivery is late",
        "delivery delayed",
        "delivery delay",
        "tracking my order",
        "track my order",
        "track my package",
        "tracking number",
        "delivery date",
        "expected delivery",
    ]):
        return "delivery_issue"

    # Order management
    if any(term in text for term in [
        "cancel my order",
        "cancel order",
        "cancel the order",
        "change my order",
        "change order",
        "modify my order",
        "modify order",
        "change the order",
        "reorder",
        "replacement",
        "replace my order",
    ]):
        return "order_management"

    # No obvious keyword
    return "other"


def simple_escalate_rule(text: str) -> bool:
    """
    Simple non-ML escalation baseline.

    Escalate when the customer message contains obvious
    refund, return, payment, or billing language.
    """

    text = str(text).lower()

    terms = [
        "refund",
        "return",
        "charge",
        "charged",
        "billing",
        "payment",
    ]

    return any(term in text for term in terms)


def build_training_labels(training_path: Path):
    """
    Load labels from the 800-example labelled training set.
    """

    training = pd.read_csv(training_path)

    required_columns = {
        "customer_message",
        "intent",
    }

    missing = required_columns - set(training.columns)

    if missing:
        raise ValueError(
            f"Training set is missing columns: {sorted(missing)}"
        )

    training = training[
        training["intent"].isin(INTENTS)
    ].copy()

    if len(training) == 0:
        raise ValueError("Training set contains no valid labels.")

    return (
        training["customer_message"].astype(str).tolist(),
        training["intent"].tolist(),
    )


def main():
    print("Loading evaluation data...")

    golden = load_golden_set()

    texts = (
        golden["customer_message"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    true_intents = (
        golden["intent"]
        .astype(str)
        .tolist()
    )

    print(f"Golden examples: {len(golden):,}")

    # ---------------------------------------------------------
    # TRAIN SYSTEM
    # ---------------------------------------------------------

    print("\nTraining ML classifier...")

    ml = train_ml_classifier(
        str(TRAINING_PATH)
    )

    # ---------------------------------------------------------
    # TRIVIAL BASELINE
    # ---------------------------------------------------------

    training_texts, training_labels = build_training_labels(
        TRAINING_PATH
    )

    trivial = TrivialClassifier().fit(
        training_texts,
        training_labels,
    )

    # ---------------------------------------------------------
    # RETRIEVER
    # ---------------------------------------------------------

    print("\nLoading evaluation knowledge base...")

    retriever = ResolvedCaseRetriever(
        str(KB_PATH)
    )

    # ---------------------------------------------------------
    # INTENT PREDICTIONS
    # ---------------------------------------------------------

    print("\nRunning intent evaluation...")

    # Trivial baseline
    pred_trivial = trivial.predict(texts)

    # Simple keyword baseline
    pred_simple = [
        simple_intent_rule(text)
        for text in texts
    ]

    # System: ML classifier + high precision rules
    pred_system = [
        ml.predict_proba_one(text).intent
        for text in texts
    ]

    results = {}

    results["intent"] = {
        "trivial_baseline": intent_metrics(
            true_intents,
            pred_trivial,
            INTENTS,
        ),
        "simple_keyword_rule": intent_metrics(
            true_intents,
            pred_simple,
            INTENTS,
        ),
        "system_ml": intent_metrics(
            true_intents,
            pred_system,
            INTENTS,
        ),
    }

    # ---------------------------------------------------------
    # ESCALATION
    # ---------------------------------------------------------

    print("Running escalation evaluation...")

    # For the current golden set, escalation policy labels
    # are not assumed to exist. Therefore we evaluate the
    # system policy against a deterministic policy derived
    # from the labelled intent taxonomy.
    #
    # Sensitive intents are expected to escalate:
    # account_access, payment_billing, refund_return.

    true_escalate = [
        intent in {
            "account_access",
            "payment_billing",
            "refund_return",
        }
        for intent in true_intents
    ]

    # Never escalate baseline
    esc_never = [
        False
        for _ in texts
    ]

    # Always escalate baseline
    esc_always = [
        True
        for _ in texts
    ]

    # Simple keyword escalation baseline
    esc_simple = [
        simple_escalate_rule(text)
        for text in texts
    ]

    # System escalation policy
    esc_system = []
    similarities = []

    for text, predicted_intent in zip(
        texts,
        pred_system,
    ):
        retrieved = retriever.retrieve(
            text,
            k=3,
            intent_filter=predicted_intent,
        )

        top_similarity = (
            retrieved[0].similarity
            if retrieved
            else 0.0
        )

        similarities.append(top_similarity)

        decision = system_decide(
            text,
            predicted_intent,
            top_similarity,
        )

        esc_system.append(
            decision.escalate
        )

    results["escalation"] = {
        "trivial_never_escalate": escalation_metrics(
            true_escalate,
            esc_never,
        ),
        "trivial_always_escalate": escalation_metrics(
            true_escalate,
            esc_always,
        ),
        "simple_keyword_rule": escalation_metrics(
            true_escalate,
            esc_simple,
        ),
        "system_policy": escalation_metrics(
            true_escalate,
            esc_system,
        ),
    }

    # ---------------------------------------------------------
    # REPLY QUALITY
    # ---------------------------------------------------------

    print("Running reply-quality evaluation...")

    quality_scores = []
    per_example = []

    for (
        text,
        predicted_intent,
        escalate,
    ) in zip(
        texts,
        pred_system,
        esc_system,
    ):
        retrieved = retriever.retrieve(
            text,
            k=3,
            intent_filter=predicted_intent,
        )

        drafted = draft_reply(
            text,
            predicted_intent,
            retrieved,
        )

        judge = judge_score(
            text,
            drafted.text,
            drafted.grounded_on,
            escalate,
        )

        score_dict = judge.as_dict()

        quality_scores.append(
            score_dict
        )

        row_index = len(per_example)

        per_example.append(
            {
                "customer_tweet_id": golden.iloc[
                    row_index
                ]["customer_tweet_id"],
                "text": text,
                "true_intent": golden.iloc[
                    row_index
                ]["intent"],
                "predicted_intent": predicted_intent,
                "reply": drafted.text,
                "escalate": escalate,
                "top_similarity": similarities[
                    row_index
                ],
                **score_dict,
            }
        )

    quality_fields = [
        "groundedness",
        "correctness",
        "tone",
        "completeness",
        "overall",
    ]

    avg_quality = {
        field: round(
            sum(
                score[field]
                for score in quality_scores
            )
            / len(quality_scores),
            2,
        )
        for field in quality_fields
    }

    judge_backend = (
        quality_scores[0]["backend"]
        if quality_scores
        else "unknown"
    )

    results["reply_quality_system"] = {
        "average": avg_quality,
        "judge_backend": judge_backend,
    }

    # ---------------------------------------------------------
    # SAVE PREDICTIONS
    # ---------------------------------------------------------

    with open(
        PREDICTIONS_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        if per_example:
            writer = csv.DictWriter(
                f,
                fieldnames=per_example[0].keys(),
            )

            writer.writeheader()
            writer.writerows(per_example)

    # ---------------------------------------------------------
    # SAVE JSON RESULTS
    # ---------------------------------------------------------

    with open(
        RESULTS_JSON_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    # ---------------------------------------------------------
    # SAVE MARKDOWN REPORT
    # ---------------------------------------------------------

    write_markdown_report(
        results,
        len(golden),
    )

    print_summary(
        results,
        len(golden),
    )


def write_markdown_report(results, n):
    lines = []

    lines.append(
        "# Evaluation Results\n"
    )

    lines.append(
        f"Golden set size: **{n} real hand-labelled examples**.\n"
    )

    lines.append(
        "The classifier was trained on the separate "
        "`training_labeled.csv` training set. "
        "The evaluation retriever used `kb_for_eval.csv`, "
        "which excludes golden-set customer IDs.\n"
    )

    lines.append(
        "## Intent classification\n"
    )

    lines.append(
        "| Model | Accuracy | Macro F1 |\n"
    )

    lines.append(
        "|---|---:|---:|\n"
    )

    for name, key in [
        (
            "Trivial (majority class)",
            "trivial_baseline",
        ),
        (
            "Simple (keyword rules)",
            "simple_keyword_rule",
        ),
        (
            "System (TF-IDF + LogReg)",
            "system_ml",
        ),
    ]:

        metrics = results["intent"][key]

        lines.append(
            f"| {name} | "
            f"{metrics['accuracy']:.3f} | "
            f"{metrics['macro_f1']:.3f} |\n"
        )

    lines.append(
        "\n## Escalation decision\n"
    )

    lines.append(
        "| Policy | Precision | Recall | F1 | "
        "Escalation rate (true / pred) |\n"
    )

    lines.append(
        "|---|---:|---:|---:|---:|\n"
    )

    for name, key in [
        (
            "Never escalate",
            "trivial_never_escalate",
        ),
        (
            "Always escalate",
            "trivial_always_escalate",
        ),
        (
            "Simple keyword rule",
            "simple_keyword_rule",
        ),
        (
            "System policy",
            "system_policy",
        ),
    ]:

        metrics = results["escalation"][key]

        lines.append(
            f"| {name} | "
            f"{metrics['precision_escalate']:.3f} | "
            f"{metrics['recall_escalate']:.3f} | "
            f"{metrics['f1_escalate']:.3f} | "
            f"{metrics['escalation_rate_true']:.2f} / "
            f"{metrics['escalation_rate_pred']:.2f} |\n"
        )

    lines.append(
        "\n## Reply quality\n"
    )

    lines.append(
        f"Judge backend: `{results['reply_quality_system']['judge_backend']}`\n"
    )

    quality = results[
        "reply_quality_system"
    ]["average"]

    lines.append(
        "| Groundedness | Correctness | Tone | "
        "Completeness | Overall |\n"
    )

    lines.append(
        "|---:|---:|---:|---:|---:|\n"
    )

    lines.append(
        f"| {quality['groundedness']} | "
        f"{quality['correctness']} | "
        f"{quality['tone']} | "
        f"{quality['completeness']} | "
        f"{quality['overall']} |\n"
    )

    with open(
        RESULTS_MD_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "".join(lines)
        )


def print_summary(results, n):
    print(
        f"\n=== Evaluated on {n} golden examples ===\n"
    )

    print(
        "-- Intent accuracy / macro-F1 --"
    )

    for name, key in [
        (
            "trivial",
            "trivial_baseline",
        ),
        (
            "simple",
            "simple_keyword_rule",
        ),
        (
            "system",
            "system_ml",
        ),
    ]:

        metrics = results["intent"][key]

        print(
            f"  {name:10s} "
            f"acc={metrics['accuracy']:.3f}  "
            f"macroF1={metrics['macro_f1']:.3f}"
        )

    print(
        "\n-- Escalation precision/recall/F1 "
        "(positive=escalate) --"
    )

    for name, key in [
        (
            "never",
            "trivial_never_escalate",
        ),
        (
            "always",
            "trivial_always_escalate",
        ),
        (
            "keyword",
            "simple_keyword_rule",
        ),
        (
            "system",
            "system_policy",
        ),
    ]:

        metrics = results["escalation"][key]

        print(
            f"  {name:10s} "
            f"P={metrics['precision_escalate']:.3f}  "
            f"R={metrics['recall_escalate']:.3f}  "
            f"F1={metrics['f1_escalate']:.3f}"
        )

    print(
        "\n-- Reply quality --"
    )

    print(
        results["reply_quality_system"]["average"]
    )

    print(
        "\nFull report -> "
        "reports/eval_results.md"
    )

    print(
        "Per-example predictions -> "
        "eval/system_predictions.csv"
    )


if __name__ == "__main__":
    main()