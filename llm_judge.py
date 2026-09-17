"""
Reply-quality evaluator.

Scores four dimensions from 1-5:

- groundedness: Does the reply use the retrieved evidence appropriately?
- correctness: Does the reply address the customer's actual request?
- tone: Is the reply professional and empathetic?
- completeness: Does it provide an appropriate next step?

The default heuristic judge is deterministic and offline.

If ANTHROPIC_API_KEY is available, the real LLM judge is used instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from src import llm_backend


# ============================================================
# MARKERS
# ============================================================

POLITENESS_MARKERS = [
    "sorry",
    "thanks",
    "thank you",
    "appreciate",
    "happy to",
    "glad to",
    "understand",
]

NEXT_STEP_MARKERS = [
    "please",
    "contact",
    "reach out",
    "support",
    "dm",
    "email",
    "check",
    "review",
    "help",
    "provide",
]


# ============================================================
# RESULT
# ============================================================

@dataclass
class JudgeScore:
    groundedness: int
    correctness: int
    tone: int
    completeness: int
    overall: float
    backend: str
    rationale: str = ""

    def as_dict(self):
        return asdict(self)


# ============================================================
# TEXT HELPERS
# ============================================================

def _words(text: str) -> set[str]:
    return set(
        re.findall(
            r"[a-z']+",
            str(text).lower(),
        )
    )


def _word_overlap(a: str, b: str) -> float:
    wa = _words(a)
    wb = _words(b)

    if not wa or not wb:
        return 0.0

    return len(wa & wb) / len(wa | wb)


def _contains_any(text: str, terms: list[str]) -> bool:
    text = str(text).lower()
    return any(term in text for term in terms)


# ============================================================
# INTENT SIGNALS
# ============================================================

INTENT_SIGNALS = {
    "delivery_issue": [
        "delivery",
        "delivered",
        "package",
        "parcel",
        "order",
        "tracking",
        "shipment",
        "courier",
        "arrived",
        "received",
        "late",
    ],

    "refund_return": [
        "refund",
        "return",
        "money back",
        "returned",
    ],

    "payment_billing": [
        "charge",
        "charged",
        "payment",
        "billing",
        "billed",
        "invoice",
        "credit",
        "debit",
    ],

    "account_access": [
        "account",
        "login",
        "log in",
        "sign in",
        "password",
        "locked",
        "verification",
    ],

    "product_problem": [
        "broken",
        "damaged",
        "defective",
        "faulty",
        "working",
        "malfunction",
        "freeze",
        "replacement",
        "exchange",
        "product",
        "item",
    ],

    "order_management": [
        "cancel",
        "change order",
        "modify order",
        "reorder",
        "replacement",
    ],

    "other": [],
}


# ============================================================
# INTENT INFERENCE FOR JUDGING
# ============================================================

def _infer_customer_intent(message: str) -> str:
    """
    Lightweight intent inference used only by the offline judge.

    This is deliberately conservative. It is not the production
    classifier and does not affect system predictions.
    """

    text = str(message).lower()

    # High-priority explicit requests.

    if any(
        x in text
        for x in [
            "login",
            "log in",
            "sign in",
            "password",
            "locked out",
            "account locked",
        ]
    ):
        return "account_access"

    if any(
        x in text
        for x in [
            "charged twice",
            "duplicate charge",
            "payment",
            "billing",
            "invoice",
            "charged",
            "billed",
        ]
    ):
        return "payment_billing"

    if any(
        x in text
        for x in [
            "refund",
            "money back",
            "return my",
            "return this",
            "return the",
            "return policy",
        ]
    ):
        return "refund_return"

    if any(
        x in text
        for x in [
            "broken",
            "damaged",
            "defective",
            "faulty",
            "not working",
            "doesn't work",
            "does not work",
            "malfunction",
            "freezes",
            "freezing",
        ]
    ):
        return "product_problem"

    if any(
        x in text
        for x in [
            "cancel my order",
            "cancel the order",
            "cancel order",
            "change my order",
            "change the order",
            "modify my order",
            "reorder",
        ]
    ):
        return "order_management"

    if any(
        x in text
        for x in [
            "where is my order",
            "where is my package",
            "where is my parcel",
            "not delivered",
            "not received",
            "haven't received",
            "have not received",
            "delivery",
            "tracking",
            "package",
            "parcel",
        ]
    ):
        return "delivery_issue"

    return "other"


# ============================================================
# GROUNDEDNESS
# ============================================================

def _score_groundedness(
    reply: str,
    evidence: list[str],
) -> int:

    if not evidence:
        return 2

    overlap_scores = [
        _word_overlap(reply, evidence_item)
        for evidence_item in evidence
    ]

    best_overlap = max(overlap_scores)

    if best_overlap >= 0.45:
        return 5

    if best_overlap >= 0.30:
        return 4

    if best_overlap >= 0.18:
        return 3

    if best_overlap >= 0.08:
        return 2

    return 1


# ============================================================
# CORRECTNESS
# ============================================================

def _score_correctness(
    message: str,
    reply: str,
) -> int:

    customer_intent = _infer_customer_intent(message)

    reply_lower = reply.lower()

    signals = INTENT_SIGNALS.get(
        customer_intent,
        [],
    )

    # "other" cannot be strongly matched using generic keywords.
    if customer_intent == "other":
        if len(reply_lower.split()) >= 8:
            return 4
        return 3

    matching_signals = sum(
        1
        for signal in signals
        if signal in reply_lower
    )

    # A reply that contains relevant intent terminology.
    if matching_signals >= 2:
        score = 5

    elif matching_signals == 1:
        score = 4

    else:
        score = 2

    # Explicitly check whether the reply actually offers help.
    if _contains_any(
        reply_lower,
        [
            "help",
            "support",
            "check",
            "review",
            "contact",
            "assist",
        ],
    ):
        score = min(5, score + 1)

    return max(1, score)


# ============================================================
# TONE
# ============================================================

def _score_tone(reply: str) -> int:

    low = reply.lower().strip()

    if not low:
        return 1

    words = low.split()

    score = 3

    if _contains_any(
        low,
        POLITENESS_MARKERS,
    ):
        score += 1

    if len(words) >= 15:
        score += 1

    # Avoid overly aggressive / unprofessional language.
    if any(
        x in low
        for x in [
            "fuck",
            "shit",
            "stupid",
            "idiot",
            "fucking",
        ]
    ):
        score -= 2

    return max(
        1,
        min(5, score),
    )


# ============================================================
# COMPLETENESS
# ============================================================

def _score_completeness(
    message: str,
    reply: str,
    escalate: bool,
) -> int:

    low = reply.lower()

    has_next_step = _contains_any(
        low,
        NEXT_STEP_MARKERS,
    )

    score = 2

    if has_next_step:
        score = 4

    if len(low.split()) >= 20 and has_next_step:
        score = 5

    # If escalated, avoid pretending that a concrete resolution
    # has already happened.
    if escalate and any(
        x in low
        for x in [
            "refund has been processed",
            "refund is complete",
            "issue is resolved",
            "problem is fixed",
            "we have fixed",
        ]
    ):
        score -= 2

    return max(
        1,
        min(5, score),
    )


# ============================================================
# HEURISTIC JUDGE
# ============================================================

def heuristic_score(
    message: str,
    reply: str,
    evidence: list[str],
    escalate: bool,
) -> JudgeScore:

    groundedness = _score_groundedness(
        reply,
        evidence,
    )

    correctness = _score_correctness(
        message,
        reply,
    )

    tone = _score_tone(reply)

    completeness = _score_completeness(
        message,
        reply,
        escalate,
    )

    overall = round(
        (
            groundedness
            + correctness
            + tone
            + completeness
        )
        / 4,
        2,
    )

    rationale = (
        "Offline deterministic rubric based on "
        "evidence overlap, customer intent signals, "
        "tone markers, and next-step completeness."
    )

    return JudgeScore(
        groundedness=groundedness,
        correctness=correctness,
        tone=tone,
        completeness=completeness,
        overall=overall,
        backend="heuristic",
        rationale=rationale,
    )


# ============================================================
# REAL LLM JUDGE
# ============================================================

def llm_score(
    message: str,
    reply: str,
    evidence: list[str],
    escalate: bool,
) -> JudgeScore | None:

    system = (
        "You are grading a customer-support agent's draft reply. "
        "Score each dimension from 1 to 5 as integers. "
        "Groundedness: does the reply stick to the provided evidence "
        "and avoid inventing policy or facts? "
        "Correctness: does it address the customer's actual request? "
        "Tone: is it empathetic, professional and natural? "
        "Completeness: does it provide an appropriate next step without "
        "over-promising a resolution? "
        "Respond ONLY with JSON: "
        "{\"groundedness\":int,\"correctness\":int,"
        "\"tone\":int,\"completeness\":int,"
        "\"rationale\":\"one sentence\"}"
    )

    ev_block = "\n".join(
        f"- {e}"
        for e in evidence
    ) or "(no evidence retrieved)"

    user = (
        f"Customer message: \"{message}\"\n"
        f"Escalated to human: {escalate}\n"
        f"Evidence from previous AmazonHelp cases:\n"
        f"{ev_block}\n\n"
        f"Draft reply: \"{reply}\""
    )

    data = llm_backend.call_llm_json(
        system,
        user,
        max_tokens=250,
    )

    if data is None:
        return None

    try:
        g = int(data["groundedness"])
        c = int(data["correctness"])
        t = int(data["tone"])
        comp = int(data["completeness"])

        # Keep the judge within the declared 1-5 scale.
        g = max(1, min(5, g))
        c = max(1, min(5, c))
        t = max(1, min(5, t))
        comp = max(1, min(5, comp))

        overall = round(
            (g + c + t + comp) / 4,
            2,
        )

        return JudgeScore(
            groundedness=g,
            correctness=c,
            tone=t,
            completeness=comp,
            overall=overall,
            backend="llm",
            rationale=data.get(
                "rationale",
                "",
            ),
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ):
        return None


# ============================================================
# PUBLIC API
# ============================================================

def score(
    message: str,
    reply: str,
    evidence: list[str],
    escalate: bool,
) -> JudgeScore:

    if llm_backend.llm_enabled():

        result = llm_score(
            message,
            reply,
            evidence,
            escalate,
        )

        if result is not None:
            return result

    return heuristic_score(
        message,
        reply,
        evidence,
        escalate,
    )