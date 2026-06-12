"""Streaming LLM generation for the /chat endpoint (T-41).

Uses Groq ``llama-3.3-70b-versatile`` via litellm (free tier).
Wraps the call in an ``llm_span`` that records token counts.
Persists the completed message pair to ``app.chat_messages``.
"""

from __future__ import annotations

import logging
import uuid
from typing import AsyncIterator, TYPE_CHECKING

import asyncpg

from rat_api.config import get_settings
from rat_api.obs.tracing import llm_span
from rat_api.rag.prompts import CHAT_SYSTEM_PROMPT

if TYPE_CHECKING:
    from rat_api.rag.retriever import RetrievedChunk

log = logging.getLogger(__name__)

_MODEL = "groq/llama-3.3-70b-versatile"
_MAX_CONTEXT_CHARS = 4000  # per-chunk truncation for context window safety


def _build_context(chunks: "list[RetrievedChunk]") -> str:
    """Format retrieved chunks as numbered context blocks."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        text = c.content_with_prefix[:_MAX_CONTEXT_CHARS]
        parts.append(f"[{i}] {c.citation} ({c.authority})\n{text}")
    return "\n\n".join(parts)


async def generate_stream(
    query: str,
    chunks: "list[RetrievedChunk]",
    *,
    session_id: uuid.UUID,
    conn: asyncpg.Connection,
) -> AsyncIterator[str]:
    """Stream LLM tokens for *query* grounded in *chunks*.

    Yields token strings as they arrive.  When the stream is complete,
    persists the user message and assistant response to ``app.chat_messages``.

    Args:
        query:      User question.
        chunks:     Retrieved chunks to use as context.
        session_id: Chat session UUID.
        conn:       Active asyncpg connection for message persistence.
    """
    import litellm  # noqa: PLC0415

    settings = get_settings()
    if not settings.groq_api_key:
        yield "Error: GROQ_API_KEY is not configured."
        return

    context = _build_context(chunks)
    user_content = f"Context:\n{context}\n\nQuestion: {query}"

    full_response: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0

    with llm_span("chat_generate") as span:
        span.set_attribute("llm.model_name", _MODEL)
        span.set_attribute("llm.provider", "groq")
        span.set_attribute("llm.input_messages", f"system:{CHAT_SYSTEM_PROMPT[:200]}…")

        try:
            stream = await litellm.acompletion(
                model=_MODEL,
                max_tokens=1024,
                api_key=settings.groq_api_key,
                stream=True,
                messages=[
                    {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full_response.append(delta)
                    yield delta
                # Capture usage from the final chunk if present
                if hasattr(chunk, "usage") and chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens or 0
                    completion_tokens = chunk.usage.completion_tokens or 0

        except Exception as exc:  # noqa: BLE001
            log.error("LLM stream error: %s", exc)
            yield f"\n[Error: {exc}]"
            return

        span.set_attribute("llm.token_count.prompt", prompt_tokens)
        span.set_attribute("llm.token_count.completion", completion_tokens)
        span.set_attribute("llm.token_count.total", prompt_tokens + completion_tokens)
        span.set_attribute("llm.usd_cost", 0.0)  # Groq free tier

    # Persist to app.chat_messages
    assistant_text = "".join(full_response)
    await _persist_messages(
        conn=conn,
        session_id=session_id,
        user_content=query,
        assistant_content=assistant_text,
        chunks=chunks,
        latency_ms=None,
        cost_usd=0.0,
    )


async def _persist_messages(
    *,
    conn: asyncpg.Connection,
    session_id: uuid.UUID,
    user_content: str,
    assistant_content: str,
    chunks: "list[RetrievedChunk]",
    latency_ms: int | None,
    cost_usd: float,
) -> None:
    """Write user + assistant messages to ``app.chat_messages``."""
    import json  # noqa: PLC0415

    retrieved_json = json.dumps(
        [{"chunk_id": c.chunk_id, "score": c.score} for c in chunks]
    )
    sql = """
        INSERT INTO app.chat_messages
            (session_id, role, content, retrieved_chunks, latency_ms, cost_usd)
        VALUES ($1, $2, $3, $4, $5, $6)
    """
    try:
        await conn.execute(sql, session_id, "user", user_content, None, None, None)
        await conn.execute(
            sql,
            session_id,
            "assistant",
            assistant_content,
            retrieved_json,
            latency_ms,
            cost_usd,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to persist chat messages: %s", exc)
