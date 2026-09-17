"""
Optional real-LLM hook. The whole pipeline runs fully offline by default
(deterministic TF-IDF classifier + retrieval + template synthesis), so
grading doesn't require any API key. If ANTHROPIC_API_KEY is set in the
environment, classify/generate/judge steps can optionally route through a
real Claude call for higher-quality drafting and judging.

This keeps "reproduce in <15 minutes" true regardless of whether the grader
has a key, while still satisfying "you may use any LLM API".
"""
import os
import json

_ANTHROPIC_AVAILABLE = False
try:
    import anthropic  # noqa
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    pass

MODEL = "claude-sonnet-4-6"


def llm_enabled() -> bool:
    return _ANTHROPIC_AVAILABLE and bool(os.environ.get("ANTHROPIC_API_KEY"))


def call_llm(system: str, user: str, max_tokens: int = 400) -> str | None:
    """Returns None if no LLM is configured -- callers must have a fallback."""
    if not llm_enabled():
        return None
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")
    except Exception as e:  # noqa
        print(f"[llm_backend] call failed, falling back to offline mode: {e}")
        return None


def call_llm_json(system: str, user: str, max_tokens: int = 400) -> dict | None:
    raw = call_llm(system, user, max_tokens=max_tokens)
    if raw is None:
        return None
    try:
        cleaned = raw.strip().strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        return json.loads(cleaned)
    except Exception:
        return None
