"""Reusable validation helpers for the MemoryNPC notebooks.

The notebooks contain the executed assignment results. This module makes the
evaluation logic reusable so the validation is not only hand-written in notebook
cells. It covers retrieval baselines, trace completeness, and lightweight
response-grounding checks.
"""

import re
from typing import Any, Dict, Iterable, List, Sequence


REQUIRED_TRACE_COLUMNS = {
    "turn",
    "player_input",
    "intent",
    "extracted_memory",
    "saved_memory",
    "retrieved_memories",
    "trust_before",
    "trust_score",
    "trust_level",
    "trust_delta",
    "advice_status",
    "advice_delta",
    "active_advice",
    "new_advice",
    "npc_response",
}


def tokenize(text: str) -> set[str]:
    """Tokenize text for a deliberately simple keyword baseline."""
    return set(re.findall(r"[a-zA-Z]+", text.lower()))


def keyword_retrieve(query: str, memories: Sequence[str], k: int = 3) -> List[str]:
    """Retrieve memories by word overlap for a transparent baseline."""
    query_tokens = tokenize(query)
    scored = []
    for memory in memories:
        overlap = len(query_tokens & tokenize(memory))
        if overlap:
            scored.append((overlap, memory))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [memory for _, memory in scored[:k]]


def retrieval_success(top_memories: Sequence[str], expected_keyword: str) -> bool:
    """Return whether an expected keyword appears in retrieved memories."""
    expected = expected_keyword.lower()
    return any(expected in memory.lower() for memory in top_memories)


def missing_trace_columns(event: Dict[str, Any]) -> set[str]:
    """Find missing validation fields in one event-log row."""
    return REQUIRED_TRACE_COLUMNS - set(event)


def automatic_response_checks(event: Dict[str, Any]) -> Dict[str, Any]:
    """Run lightweight response-quality checks on a generated turn.

    These checks do not replace human judgement, but they make response quality
    less subjective by testing trace completeness, trust arithmetic, and whether
    memory answers are grounded in retrieved evidence.
    """
    retrieved = event.get("retrieved_memories") or []
    response = str(event.get("npc_response", ""))
    response_lower = response.lower()
    intent = event.get("intent")

    checks = {
        "trace_complete": not missing_trace_columns(event),
        "trust_delta_matches": event.get("trust_score") - event.get("trust_before") == event.get("trust_delta"),
        "grounding_check_applicable": intent == "ask_memory",
        "memory_answer_grounded": True,
    }

    if intent == "ask_memory":
        if retrieved:
            retrieved_tokens = set()
            for memory in retrieved:
                retrieved_tokens |= {token for token in tokenize(str(memory)) if len(token) > 3}
            response_tokens = tokenize(response)
            checks["memory_answer_grounded"] = bool(retrieved_tokens & response_tokens)
        else:
            uncertainty_markers = [
                "do not recall",
                "don't recall",
                "do not remember",
                "don't remember",
                "no relevant memory",
                "nothing in memory",
            ]
            checks["memory_answer_grounded"] = any(marker in response_lower for marker in uncertainty_markers)

    checks["all_automatic_checks_pass"] = all(
        value for key, value in checks.items() if key != "grounding_check_applicable"
    )
    return checks


def summarize_response_checks(events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize automatic response checks over a validation trace."""
    rows = [automatic_response_checks(event) for event in events]
    if not rows:
        return {"event_count": 0, "pass_rate": 0.0}
    passed = sum(row["all_automatic_checks_pass"] for row in rows)
    return {
        "event_count": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "pass_rate": passed / len(rows),
    }
