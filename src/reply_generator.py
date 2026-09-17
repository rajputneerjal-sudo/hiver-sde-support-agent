from __future__ import annotations

from dataclasses import dataclass

from .retrieval import RetrievedCase


@dataclass
class DraftedReply:
    text: str
    source: str
    grounded_on: list[str]


def _clean_reply(text: str) -> str:
    """Clean a retrieved AmazonHelp response."""

    text = str(text).strip()

    # Remove excessive whitespace.
    text = " ".join(text.split())

    return text


def _fallback_reply(intent: str) -> str:
    """Safe fallback when retrieval does not find a useful response."""

    replies = {
        "delivery_issue": (
            "I'm sorry you're having trouble with your delivery. "
            "Please contact Amazon Customer Support so they can check "
            "the current status of your order."
        ),

        "refund_return": (
            "I'm sorry you're having trouble with your return or refund. "
            "Please contact Amazon Customer Support so they can review "
            "the order and help with the next steps."
        ),

        "payment_billing": (
            "I'm sorry you're experiencing a payment or billing issue. "
            "Please contact Amazon Customer Support so they can review "
            "the charge and help resolve the issue."
        ),

        "account_access": (
            "I'm sorry you're having trouble accessing your account. "
            "Please contact Amazon Customer Support so they can help "
            "you securely resolve the account-access issue."
        ),

        "product_problem": (
            "I'm sorry you're experiencing a problem with the product. "
            "Please contact Amazon Customer Support so they can review "
            "the issue and help with the available options."
        ),

        "order_management": (
            "I'm sorry you're having trouble managing your order. "
            "Please contact Amazon Customer Support so they can check "
            "the order and help with the appropriate next steps."
        ),

        "other": (
            "I'm sorry you're experiencing this issue. "
            "Please contact Amazon Customer Support so we can "
            "look into this for you."
        ),
    }

    return replies.get(
        intent,
        replies["other"],
    )


def draft_reply(
    message: str,
    intent: str,
    retrieved: list[RetrievedCase],
) -> DraftedReply:

    # ========================================================
    # NO RETRIEVAL RESULT
    # ========================================================

    if not retrieved:
        return DraftedReply(
            text=_fallback_reply(intent),
            source="fallback",
            grounded_on=[],
        )

    # ========================================================
    # SELECT BEST RETRIEVED CASE
    # ========================================================

    best = retrieved[0]

    reply = _clean_reply(best.brand_reply)

    # ========================================================
    # EMPTY RETRIEVED RESPONSE
    # ========================================================

    if not reply:
        return DraftedReply(
            text=_fallback_reply(intent),
            source="fallback",
            grounded_on=[],
        )

    # ========================================================
    # EVIDENCE
    # ========================================================

    evidence = []

    for case in retrieved:
        customer_text = _clean_reply(case.customer_text)

        evidence.append(
            (
                f"Retrieved case similarity={case.similarity:.3f}: "
                f"{customer_text[:160]}"
            )
        )

    # ========================================================
    # RETURN GROUNDED RESPONSE
    # ========================================================

    return DraftedReply(
        text=reply,
        source="real_amazonhelp_retrieval",
        grounded_on=evidence,
    )