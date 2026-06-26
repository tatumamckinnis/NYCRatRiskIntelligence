# ADR-0007 — Embedding strategy: voyage-3 in dev, BM25-only in prod

**Status**: Accepted
**Date**: 2026-06-10
**Deciders**: project owner

---

## Context

SPEC §2 specified:
- Primary embeddings: `voyage-3-large` (1024-dim, Voyage AI hosted)
- Ablation: BGE-M3 self-hosted (stored in `embedding_bge` column)
- Reranker: BGE Reranker v2-M3 self-hosted in the API container

Three issues surfaced during Phase 4 execution:

1. **`voyage-3-large` vs `voyage-3`**: Voyage AI deprecated `voyage-3-large` as a standalone model name during the development period. The current production model at equivalent quality is simply `voyage-3` (1024-dim output unchanged). The spec's `voyage-3-large` string returns an API error; `voyage-3` is the correct identifier.

2. **BGE-M3 OOM on Render free tier**: `BAAI/bge-m3` requires ~800 MB RAM to load via `sentence-transformers`. Render free tier provides 512 MB. Loading BGE-M3 at API startup causes an immediate OOM kill. BGE Reranker v2-M3 has the same problem (~570 MB). Both are disabled in production via `DISABLE_VECTOR_SEARCH=true` and `DISABLE_RERANKER=true` env vars.

3. **Voyage API key not required in prod**: with vector search disabled in prod, the `voyageai` client is never called. `VOYAGE_API_KEY` is therefore a dev-only dependency.

---

## Decision

**Development (local / Docker Compose):**
- Dense embeddings: `voyage-3` via Voyage AI API (corrected from `voyage-3-large`)
- Reranker: BGE Reranker v2-M3 self-hosted (loaded at API startup, requires `DISABLE_RERANKER` unset)
- Retrieval: BM25 + dense + RRF + BGE Reranker (full hybrid pipeline)

**Production (Render free tier):**
- `DISABLE_VECTOR_SEARCH=true` — skips dense embedding at query time
- `DISABLE_RERANKER=true` — skips BGE Reranker
- Effective retrieval: BM25 only (`tsvector plainto_tsquery`, top-30, returned directly)
- `VOYAGE_API_KEY` not required

---

## Rationale

| Option | Pros | Cons |
|---|---|---|
| **BM25-only in prod (chosen)** | Fits in 512 MB; zero API cost; answers are still cited | Estimated Recall@5 drops from ~0.82 to ~0.68 on complex queries |
| Load BGE-M3 lazily (on first query) | Full hybrid in prod | First request OOMs; no improvement if resident in memory |
| Use smaller reranker (`bge-reranker-base`, ~280 MB) | Fits in 512 MB | Still requires BGE-M3 for dense embed (~800 MB); net OOM |
| Cohere Rerank 3.5 (API) | No local RAM cost | Requires `COHERE_API_KEY`; adds API dependency for free-tier path |
| Upgrade to Render Starter ($7/mo) | Full hybrid, no OOM | Monthly cost; out of scope for $0 budget |

BM25 retrieval is acceptable for well-formed legal queries (statute numbers, key terms). Quality degrades on colloquial or paraphrased queries where dense recall would help — documented in Known Limitations in the README.

**Why:** 512 MB RAM constraint is hard; $0 budget is hard; BM25 recall is good enough for the demo use case.
**How to apply:** Check `DISABLE_VECTOR_SEARCH` in `api/src/rat_api/rag/retrieve.py` before embedding the query. The full hybrid path is exercised in local dev and integration tests.

---

## Consequences

- `VOYAGE_API_KEY` is listed as legacy / optional in `.env.example`. Local dev needs it only if `DISABLE_VECTOR_SEARCH` is unset.
- The `embedding` column in `app.health_code_chunks` is populated with voyage-3 vectors (ingested locally before deployment). The `embedding_bge` column is populated with BGE-M3 vectors (same ingest run). Both columns persist in the schema for future use.
- The architecture diagram shows the dev path (full hybrid). The README Known Limitations section and the runbook document the prod degraded path.
- If Render is upgraded to a paid tier with ≥2 GB RAM, re-enable with: unset `DISABLE_VECTOR_SEARCH` and `DISABLE_RERANKER` in Render env vars — no code change required.
- This decision is related to ADR 0001 (single-cloud / free tier) which also notes the BGE-M3 + Reranker RAM constraint.
