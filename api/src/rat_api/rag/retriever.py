"""Hybrid retrieval pipeline: BM25 + dense (BGE-M3) + RRF + BGE Reranker (T-39).

Steps
-----
1. **Query rewriting** — Groq openai/gpt-oss-20b expands statutory vocabulary (≤ 200 tokens).
2. **Dense retrieval** — BGE-M3 local query embedding → pgvector HNSW cosine top-K.
3. **BM25 retrieval** — ``plainto_tsquery`` + ``ts_rank_cd`` on the ``content_tsv`` index.
4. **RRF fusion** — Reciprocal Rank Fusion (k=60) over dense + BM25 result lists.
5. **Rerank** — BGE Reranker v2-M3 (or Cohere ablation) → top-6.
6. **Parent expansion** — fetch parent chunk and prepend to context (≤ 4000 token budget).

Public API
----------
``retrieve(query, conn, ...)`` — full pipeline, returns ``list[RetrievedChunk]``.
``RetrievedChunk`` — dataclass returned to callers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import asyncpg
from rat_ml.rag.embed import (
    embed_query_bge as embed_query,  # noqa: F401 — module-level so tests can patch it
)

from rat_api.config import get_settings
from rat_api.obs.tracing import reranker_span, retriever_span
from rat_api.rag.reranker import get_reranker  # noqa: F401 — module-level so tests can patch it

log = logging.getLogger(__name__)

# RRF parameter (standard value)
_RRF_K = 60


@dataclass
class RetrievedChunk:
    chunk_id: str
    document: str
    citation: str
    authority: str
    section_path: list[str]
    content: str
    content_with_prefix: str
    score: float
    retrieval_method: str = "dense"
    parent_chunk_id: str | None = None


# ---------------------------------------------------------------------------
# Step 1 — Query rewriting
# ---------------------------------------------------------------------------

def _rewrite_query(query: str, *, api_key: str) -> str:
    """Expand *query* with statutory vocabulary via Groq openai/gpt-oss-20b (free).

    Returns the original query on any failure so retrieval is never blocked.
    """
    if not api_key:
        return query

    try:
        import litellm  # noqa: PLC0415
        resp = litellm.completion(
            model="groq/openai/gpt-oss-20b",
            max_tokens=200,
            # gpt-oss is a reasoning model that spends tokens on a hidden
            # `reasoning` field before the visible answer — without capping
            # effort, short max_tokens budgets get eaten entirely by that
            # hidden reasoning and the call returns empty content.
            reasoning_effort="low",
            num_retries=1,
            api_key=api_key,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a legal search assistant. Expand the user's query with "
                        "relevant statutory terms, synonyms, and cross-references found in "
                        "NYC rodent-control law (Health Code §151, HMC §27-2017, 24 RCNY §81.23). "
                        "Return only the expanded query text — no explanation, no preamble."
                    ),
                },
                {"role": "user", "content": query},
            ],
        )
        rewritten: str = str(resp.choices[0].message.content or "").strip()
        log.debug("Query rewrite: %r → %r", query, rewritten)
        return rewritten or query
    except Exception as exc:  # noqa: BLE001
        log.warning("Query rewrite failed (%s); using original query.", exc)
        return query


# ---------------------------------------------------------------------------
# Step 2 — Dense retrieval
# ---------------------------------------------------------------------------

async def _dense_retrieve(
    query_vec: list[float],
    conn: asyncpg.Connection,
    *,
    top_k: int,
) -> list[RetrievedChunk]:
    """HNSW cosine-similarity search against ``app.health_code_chunks``."""
    vec_str = "[" + ",".join(f"{v:.8f}" for v in query_vec) + "]"
    sql = """
        SELECT
            chunk_id, document, citation, authority, section_path,
            content, content_with_prefix, parent_chunk_id,
            1 - (embedding <=> $1::vector) AS score
        FROM app.health_code_chunks
        ORDER BY embedding <=> $1::vector
        LIMIT $2
    """
    rows = await conn.fetch(sql, vec_str, top_k)
    return [
        RetrievedChunk(
            chunk_id=str(r["chunk_id"]),
            document=r["document"],
            citation=r["citation"],
            authority=r["authority"],
            section_path=list(r["section_path"]),
            content=r["content"],
            content_with_prefix=r["content_with_prefix"],
            score=float(r["score"]),
            retrieval_method="dense",
            parent_chunk_id=str(r["parent_chunk_id"]) if r["parent_chunk_id"] else None,
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Step 3 — BM25 retrieval
# ---------------------------------------------------------------------------

async def _bm25_retrieve(
    query: str,
    conn: asyncpg.Connection,
    *,
    top_k: int,
) -> list[RetrievedChunk]:
    """Full-text search using the ``content_tsv`` GIN index.

    ``plainto_tsquery`` ANDs every lexeme together, which sounds right but
    means a natural-language question (with 8-15+ significant words after
    stemming) essentially never matches any single chunk — verified
    empirically: even a short real question like "What does active rat
    signs mean under the NYC Health Code?" returned zero rows. Reusing
    plainto_tsquery's stemming/stopword handling but rewriting its AND (`&`)
    operators to OR (`|`) turns this into a proper ranked search — chunks
    matching more query terms rank higher via ts_rank_cd, but a chunk
    doesn't need *every* term to be a candidate at all.

    ``ts_rank_cd``'s normalization defaulted to 0 (no length adjustment),
    which let long, topically-broad chunks (e.g. a multi-page "Rodent
    Academy" training doc) outrank short, specific chunks that actually
    answer the question just because they contain more raw keyword hits.
    Normalization ``5`` (``1 | 4`` — log-length + unique-word-count) fixes
    this by rewarding term density over raw length. Verified empirically
    against the full gold set (``evals/gold/article151_qa_v1.jsonl``):
    recall@5 0.093→0.122, recall@6 0.115→0.144, recall@10 0.126→0.163 —
    a consistent ~30% relative improvement at every k tested, measured
    directly against Postgres (no LLM calls / Groq budget involved).
    """
    sql = """
        WITH q AS (
            SELECT replace(plainto_tsquery('english', $1)::text, ' & ', ' | ') AS tsq
        )
        SELECT
            chunk_id, document, citation, authority, section_path,
            content, content_with_prefix, parent_chunk_id,
            ts_rank_cd(content_tsv, to_tsquery('english', q.tsq), 5) AS score
        FROM app.health_code_chunks, q
        WHERE content_tsv @@ to_tsquery('english', q.tsq)
        ORDER BY score DESC
        LIMIT $2
    """
    try:
        rows = await conn.fetch(sql, query, top_k)
    except Exception as exc:  # noqa: BLE001
        log.warning("BM25 retrieval failed: %s", exc)
        return []
    return [
        RetrievedChunk(
            chunk_id=str(r["chunk_id"]),
            document=r["document"],
            citation=r["citation"],
            authority=r["authority"],
            section_path=list(r["section_path"]),
            content=r["content"],
            content_with_prefix=r["content_with_prefix"],
            score=float(r["score"]),
            retrieval_method="bm25",
            parent_chunk_id=str(r["parent_chunk_id"]) if r["parent_chunk_id"] else None,
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Step 4 — RRF fusion
# ---------------------------------------------------------------------------

def _rrf_fuse(
    lists: list[list[RetrievedChunk]],
    *,
    k: int = _RRF_K,
    top_k: int,
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion over multiple ranked lists.

    ``score(d) = Σ 1 / (k + rank_i(d))``
    """
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, RetrievedChunk] = {}

    for ranked_list in lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            cid = chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in chunks_by_id:
                chunks_by_id[cid] = chunk

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    result = []
    for cid, rrf_score in fused:
        chunk = chunks_by_id[cid]
        from dataclasses import replace  # noqa: PLC0415
        result.append(replace(chunk, score=rrf_score, retrieval_method="rrf"))
    return result


# ---------------------------------------------------------------------------
# Step 6 — Parent expansion
# ---------------------------------------------------------------------------

async def _expand_parents(
    chunks: list[RetrievedChunk],
    conn: asyncpg.Connection,
    *,
    # Groq free tier's 8000 TPM limit — verified empirically that the old
    # 4000-token budget here, stacked with generator.py's own context
    # budget, could exhaust it within a few calls. 1200 keeps a typical
    # /chat call comfortably under budget.
    token_budget: int = 1200,
) -> list[RetrievedChunk]:
    """Prepend parent chunk text to children when within token budget."""
    parent_ids = [c.parent_chunk_id for c in chunks if c.parent_chunk_id]
    if not parent_ids:
        return chunks

    sql = (
        "SELECT chunk_id, content_with_prefix FROM app.health_code_chunks "
        "WHERE chunk_id = ANY($1)"
    )
    try:
        rows = await conn.fetch(sql, parent_ids)
    except Exception:  # noqa: BLE001
        return chunks

    parent_map = {str(r["chunk_id"]): r["content_with_prefix"] for r in rows}

    result = []
    total_tokens = sum(len(c.content.split()) for c in chunks)  # rough estimate

    for chunk in chunks:
        if chunk.parent_chunk_id and chunk.parent_chunk_id in parent_map:
            parent_text = parent_map[chunk.parent_chunk_id]
            extra_tokens = len(parent_text.split())
            if total_tokens + extra_tokens <= token_budget:
                expanded = f"{parent_text}\n\n---\n\n{chunk.content_with_prefix}"
                from dataclasses import replace  # noqa: PLC0415
                chunk = replace(chunk, content_with_prefix=expanded)
                total_tokens += extra_tokens
        result.append(chunk)

    return result


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------

async def retrieve(
    query: str,
    conn: asyncpg.Connection,
    *,
    top_k_dense: int = 30,
    top_k_bm25: int = 30,
    top_k_after_rrf: int = 40,
    # The eval suite's recall_at_k already slices to its own top-k (5), so
    # returning a 6th chunk here doesn't help that metric — it only adds
    # generation-context tokens, which is the thing squeezing the Groq
    # free-tier TPM budget.
    top_k_final: int = 5,
    expand_parents: bool = True,
) -> list[RetrievedChunk]:
    """Full hybrid retrieval pipeline.

    Args:
        query:           User query string.
        conn:            Active asyncpg connection.
        top_k_dense:     Candidates from vector search.
        top_k_bm25:      Candidates from BM25.
        top_k_after_rrf: Candidates after RRF fusion (before rerank).
        top_k_final:     Final chunks returned after rerank.
        expand_parents:  Whether to prepend parent chunks for context.

    Returns:
        Up to *top_k_final* :class:`RetrievedChunk` objects.
    """
    settings = get_settings()

    with retriever_span("retrieve") as span:
        span.set_attribute("retrieval.top_k", top_k_final)
        span.set_attribute("retrieval.method", "hybrid_rrf_bge")

        # Step 1 — query rewriting
        rewritten = _rewrite_query(query, api_key=settings.groq_api_key)
        span.set_attribute("retrieval.rewritten_query", rewritten)

        # Step 2 — dense retrieval (BGE-M3 local; skipped when DISABLE_VECTOR_SEARCH=true)
        if not settings.disable_vector_search:
            query_vec = embed_query(rewritten)
            dense_chunks = await _dense_retrieve(query_vec, conn, top_k=top_k_dense)
        else:
            log.debug("Vector search disabled; using BM25-only retrieval.")
            dense_chunks = []

        # Step 3 — BM25 retrieval
        bm25_chunks = await _bm25_retrieve(rewritten, conn, top_k=top_k_bm25)

        # Step 4 — RRF fusion (or just BM25 order when dense is empty)
        fused = _rrf_fuse([dense_chunks, bm25_chunks], top_k=top_k_after_rrf)

        # Step 5 — rerank
        with reranker_span("bge_rerank") as rspan:
            reranker = get_reranker()
            if reranker is not None:
                final = reranker.rerank(rewritten, fused, top_k=top_k_final)
                rspan.set_attribute("reranker.model", reranker.model_name)
            elif settings.cohere_api_key:
                final = _cohere_rerank(
                    rewritten, fused, api_key=settings.cohere_api_key, top_k=top_k_final
                )
                rspan.set_attribute("reranker.model", "cohere-rerank-3.5")
            else:
                final = fused[:top_k_final]
                rspan.set_attribute("reranker.model", "none")

        # Step 6 — parent expansion
        if expand_parents:
            final = await _expand_parents(final, conn)

        # Span attributes for observability
        scores = [c.score for c in final]
        if scores:
            import statistics  # noqa: PLC0415
            span.set_attribute("retrieval.score_distribution.min", min(scores))
            span.set_attribute("retrieval.score_distribution.p50", statistics.median(scores))
            span.set_attribute("retrieval.score_distribution.max", max(scores))
        for i, c in enumerate(final):
            span.set_attribute(f"retrieval.documents.{i}.document.id", c.chunk_id)
            span.set_attribute(f"retrieval.documents.{i}.document.score", c.score)
            span.set_attribute(f"retrieval.documents.{i}.document.metadata.citation", c.citation)
            span.set_attribute(f"retrieval.documents.{i}.document.metadata.authority", c.authority)

    return final


# ---------------------------------------------------------------------------
# Cohere rerank (ablation path)
# ---------------------------------------------------------------------------

def _cohere_rerank(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    api_key: str,
    top_k: int,
) -> list[RetrievedChunk]:
    if not chunks:
        return []
    try:
        from dataclasses import replace  # noqa: PLC0415

        import cohere  # noqa: PLC0415
        client = cohere.Client(api_key=api_key)
        docs = [c.content for c in chunks]
        resp = client.rerank(model="rerank-english-v3.0", query=query, documents=docs, top_n=top_k)
        result = []
        for hit in resp.results:
            chunk = chunks[hit.index]
            result.append(
                replace(chunk, score=hit.relevance_score, retrieval_method="cohere_rerank")
            )
        return result
    except Exception as exc:  # noqa: BLE001
        log.warning("Cohere rerank failed (%s); falling back to RRF order.", exc)
        return chunks[:top_k]
