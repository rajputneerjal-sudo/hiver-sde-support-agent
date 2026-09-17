from __future__ import annotations

from dataclasses import dataclass


ALWAYS_ESCALATE_INTENTS = {
    "account_access",
    "payment_billing",
}


@dataclass
class EscalationDecision:
    escalate: bool
    reason: str


def decide(
    message: str,
    intent: str,
    similarity: float,
) -> EscalationDecision:

    text = message.lower()

    # Account and payment issues are sensitive
    # and should be handled by a human agent.
    if intent in ALWAYS_ESCALATE_INTENTS:
        return EscalationDecision(
            escalate=True,
            reason=f"Sensitive intent: {intent}",
        )

    # Refund/return requests can require order-specific
    # verification or policy decisions.
    if intent == "refund_return":
        return EscalationDecision(
            escalate=True,
            reason="Refund or return request requires order-specific review",
        )

    # Very low retrieval confidence means the system
    # does not have a sufficiently similar resolved case.
    if similarity < 0.10:
        return EscalationDecision(
            escalate=True,
            reason="Low retrieval similarity",
        )

    # Threats, legal action, or serious complaints.
    escalation_terms = [
        "lawyer",
        "court",
        "consumer court",
        "sue",
        "legal action",
        "fraud",
        "police",
    ]

    if any(term in text for term in escalation_terms):
        return EscalationDecision(
            escalate=True,
            reason="Legal, fraud, or serious complaint language detected",
        )

    return EscalationDecision(
        escalate=False,
        reason="No escalation trigger detected",
    )