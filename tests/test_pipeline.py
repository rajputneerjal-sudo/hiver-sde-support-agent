"""
Basic sanity tests.
Run: python -m pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import SupportAgent
from src.escalation import decide


_agent = None


def get_agent():
    global _agent

    if _agent is None:
        _agent = SupportAgent()

    return _agent


def test_delivery_message_returns_valid_intent():
    agent = get_agent()

    r = agent.process(
        "Hi @AmazonHelp, where is my order? It has been 5 days."
    )

    assert r.intent in {
        "delivery_issue",
        "order_management",
        "other",
    }
    assert len(r.reply) > 0


def test_account_access_always_escalates():
    agent = get_agent()

    r = agent.process(
        "I am locked out of my account and cannot log in."
    )

    assert r.intent == "account_access"
    assert r.escalate is True


def test_payment_billing_always_escalates():
    agent = get_agent()

    r = agent.process(
        "I was charged twice for the same order."
    )

    assert r.intent == "payment_billing"
    assert r.escalate is True


def test_refund_return_always_escalates():
    d = decide(
        "I want a refund for my order.",
        "refund_return",
        0.9,
    )

    assert d.escalate is True
    assert "refund" in d.reason.lower()


def test_legal_language_forces_escalation():
    d = decide(
        "I am contacting my lawyer about this.",
        "other",
        0.9,
    )

    assert d.escalate is True
    assert "legal" in d.reason.lower()


def test_low_retrieval_similarity_forces_escalation():
    d = decide(
        "some totally novel unrelated message",
        "other",
        0.05,
    )

    assert d.escalate is True
    assert "similarity" in d.reason.lower()


def test_reply_is_never_empty():
    agent = get_agent()

    messages = [
        "Where is my package?",
        "I want to return my product.",
        "I cannot access my account.",
        "random gibberish qwerty asdf",
    ]

    for message in messages:
        r = agent.process(message)
        assert len(r.reply.strip()) > 0