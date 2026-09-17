from __future__ import annotations
from collections import Counter

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)


def intent_metrics(y_true, y_pred, labels):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def escalation_metrics(y_true_escalate, y_pred_escalate):
    """
    Precision/recall treat 'escalate=True' as the positive class. In this
    domain, a false negative (should have escalated, didn't) is worse than a
    false positive (escalated something that was actually fine) -- so recall
    on the escalate class matters more than precision. We report both plus F1
    and call this out explicitly in the report rather than only headlining F1.
    """
    return {
        "precision_escalate": precision_score(y_true_escalate, y_pred_escalate, pos_label=True, zero_division=0),
        "recall_escalate": recall_score(y_true_escalate, y_pred_escalate, pos_label=True, zero_division=0),
        "f1_escalate": f1_score(y_true_escalate, y_pred_escalate, pos_label=True, zero_division=0),
        "escalation_rate_true": sum(y_true_escalate) / len(y_true_escalate),
        "escalation_rate_pred": sum(y_pred_escalate) / len(y_pred_escalate),
    }


def majority_baseline_label(labels: list[str]) -> str:
    return Counter(labels).most_common(1)[0][0]
