from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .intents import train_ml_classifier, MLIntentClassifier
from .retrieval import ResolvedCaseRetriever
from .reply_generator import draft_reply
from .escalation import decide


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"


@dataclass
class AgentResult:
    message: str
    intent: str
    intent_confidence: float
    reply: str
    reply_source: str
    escalate: bool
    escalation_reason: str
    evidence: list[str] = field(default_factory=list)
    top_similarity: float = 0.0


class SupportAgent:
    def __init__(
        self,
        training_csv: str | None = None,
        kb_csv: str | None = None,
    ):
        training_csv = training_csv or str(
            DATA_DIR / "training_labeled.csv"
        )

        kb_csv = kb_csv or str(
            DATA_DIR / "amazonhelp_kb.csv"
        )

        print(f"Training data: {training_csv}")
        print(f"Knowledge base: {kb_csv}")

        self.classifier: MLIntentClassifier = train_ml_classifier(
            training_csv
        )

        self.retriever = ResolvedCaseRetriever(kb_csv)

    def process(self, message: str) -> AgentResult:
        prediction = self.classifier.predict_proba_one(message)

        retrieved = self.retriever.retrieve(
            message,
            k=3,
            intent_filter=prediction.intent,
        )

        top_similarity = (
            retrieved[0].similarity
            if retrieved
            else 0.0
        )

        drafted = draft_reply(
            message,
            prediction.intent,
            retrieved,
        )

        escalation = decide(
            message,
            prediction.intent,
            top_similarity,
        )

        return AgentResult(
            message=message,
            intent=prediction.intent,
            intent_confidence=prediction.confidence,
            reply=drafted.text,
            reply_source=drafted.source,
            escalate=escalation.escalate,
            escalation_reason=escalation.reason,
            evidence=drafted.grounded_on,
            top_similarity=top_similarity,
        )