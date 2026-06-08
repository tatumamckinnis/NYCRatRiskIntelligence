"""Faithfulness judge using Claude Sonnet 4.5 (T-42).

``judge_faithfulness(question, answer, chunks)`` returns 1 if the answer is
supported by the retrieved chunks, 0 otherwise.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

_JUDGE_MODEL = "claude-sonnet-4-5"

_JUDGE_SYSTEM = (
    "You are a faithful-answer evaluator. You will be given a question, "
    "a set of retrieved source chunks, and a candidate answer. "
    "Determine whether every factual claim in the answer is supported by the chunks. "
    "Respond with exactly one line: FAITHFUL: YES or FAITHFUL: NO"
)


async def judge_faithfulness(
    question: str,
    answer: str,
    chunks: list[dict],
    *,
    api_key: str = "",
) -> int:
    """Return 1 if *answer* is faithful to *chunks*, 0 otherwise.

    Args:
        question: The original user question.
        answer:   The generated assistant answer.
        chunks:   List of dicts with at least a ``"content"`` key.
        api_key:  Anthropic API key (falls back to ``ANTHROPIC_API_KEY`` env var).

    Returns:
        1 (faithful) or 0 (not faithful / error).
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        log.warning("ANTHROPIC_API_KEY not set; skipping faithfulness judge.")
        return -1  # sentinel: skipped

    try:
        import anthropic  # noqa: PLC0415
        client = anthropic.AsyncAnthropic(api_key=key)

        context = "\n\n".join(
            f"[{i + 1}] {c.get('citation', '')}: {c.get('content', '')[:600]}"
            for i, c in enumerate(chunks)
        )
        user_content = (
            f"Question: {question}\n\n"
            f"Retrieved chunks:\n{context}\n\n"
            f"Candidate answer: {answer}"
        )

        msg = await client.messages.create(
            model=_JUDGE_MODEL,
            max_tokens=10,
            system=_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
        )
        verdict = msg.content[0].text.strip().upper()
        return 1 if "YES" in verdict else 0

    except Exception as exc:  # noqa: BLE001
        log.warning("Faithfulness judge error: %s", exc)
        return -1
