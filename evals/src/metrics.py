"""Eval metrics for the RAG pipeline (T-42)."""

from __future__ import annotations

import re


def recall_at_k(
    expected_citations: list[str],
    retrieved_chunks: list[dict],
    k: int,
) -> float:
    """Fraction of *expected_citations* present in the top-*k* retrieved chunks.

    Args:
        expected_citations: List of citation strings like ``["§151.02", "§27-2018"]``.
        retrieved_chunks:   List of dicts with at least a ``"citation"`` key.
        k:                  Number of chunks to consider.

    Returns:
        Float in [0, 1].
    """
    if not expected_citations:
        return 1.0
    top_k_citations = {c.get("citation", "") for c in retrieved_chunks[:k]}
    hits = sum(1 for ec in expected_citations if any(ec in ct for ct in top_k_citations))
    return hits / len(expected_citations)


def citation_accuracy(
    expected_citations: list[str],
    generated_answer: str,
) -> float:
    """Fraction of *expected_citations* found in *generated_answer* text.

    Matches ``§N.N``, ``§N-N``, and ``§N`` patterns case-insensitively.
    """
    if not expected_citations:
        return 1.0
    hits = 0
    for ec in expected_citations:
        # Strip leading § and build a pattern
        bare = re.escape(ec.lstrip("§").strip())
        pattern = rf"§\s*{bare}"
        if re.search(pattern, generated_answer, re.IGNORECASE):
            hits += 1
    return hits / len(expected_citations)


def refusal_calibration(
    items: list[dict],
    responses: list[str],
) -> float:
    """Fraction of *unanswerable* items where the response correctly refused.

    An item is unanswerable if its ``failure_mode`` is ``"refusal_calibration"``.
    A response "correctly refuses" if it does NOT contain any phrase from
    ``must_not_say`` AND contains a phrase like "not answered", "not in the",
    "not covered", or "not specified" (case-insensitive).
    """
    unanswerable = [
        (item, resp)
        for item, resp in zip(items, responses)
        if item.get("failure_mode") == "refusal_calibration"
    ]
    if not unanswerable:
        return 1.0

    _REFUSAL_PATTERNS = re.compile(
        r"not\s+(answered|in\s+the|covered|specified|supported|provided|found)",
        re.IGNORECASE,
    )

    correct = 0
    for item, resp in unanswerable:
        must_not = item.get("must_not_say", [])
        said_bad = any(bad.lower() in resp.lower() for bad in must_not)
        said_refusal = bool(_REFUSAL_PATTERNS.search(resp))
        if not said_bad and said_refusal:
            correct += 1

    return correct / len(unanswerable)
