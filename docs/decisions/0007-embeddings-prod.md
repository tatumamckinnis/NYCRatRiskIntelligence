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
| **BM25-only in prod (chosen)** | Fits in 512 MB; zero API cost; answers are still cited | Measured Recall@5 ≈ 0.20–0.31 on the 45-item gold set with expected citations (see 2026-08-30 update below — this was never measured against real data when this ADR was first written; the ~0.82→0.68 figure below was a pre-launch estimate, not a measurement) |
| Load BGE-M3 lazily (on first query) | Full hybrid in prod | First request OOMs; no improvement if resident in memory |
| Use smaller reranker (`bge-reranker-base`, ~280 MB) | Fits in 512 MB | Still requires BGE-M3 for dense embed (~800 MB); net OOM |
| Cohere Rerank 3.5 (API) | No local RAM cost | Requires `COHERE_API_KEY`; adds API dependency for free-tier path |
| ~~Upgrade to Render Starter ($7/mo)~~ | ~~Full hybrid, no OOM~~ | **Corrected 2026-08-30**: Starter is still 512 MB RAM, same as free — it only removes spin-down/CPU throttling, it does NOT fit BGE-M3+Reranker. The actual RAM-adequate tier is **Render Standard, $25/mo, 2 GB RAM**. Out of scope for $0 budget. |

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

---

## Update 2026-08-30 — measured the real cost of this decision, ruled out cheap fixes, reaffirmed it

The nightly eval (`evals/gold/article151_qa_v1.jsonl`, 45 items with expected citations) has been running clean (no rate-limit contamination) since the pacing fix in commit `587b52c`, giving real numbers for the first time: `recall_at_k_mean` has ranged **0.14–0.31** and `citation_accuracy_mean` **0.20–0.29** across several clean runs — both well below the SPEC thresholds (0.70 / 0.60) this ADR's original estimate (~0.68/~0.82) assumed would be close enough. Two BM25-only improvement attempts were made and both ruled out:

1. **Reindex `content_tsv` onto `content_with_prefix`** (so BM25 gets credit for document-name context like "ECB Penalty Schedule") — tested in isolation against the actual LLM-rewritten queries `/chat` uses in production (not raw questions): **no measurable effect** (recall@5 0.2000 → 0.2000, identical).
2. **`ts_rank_cd` normalization tuning** (penalizing long generic chunks that outrank short specific ones) — looked like a ~30% relative improvement in an offline SQL test against raw question text, but shipped to production and **measured as a real regression** end-to-end (`recall_at_k` 0.31→0.21, `faithfulness` 0.72→0.58, dropping a previously-passing gate) — reverted same day (commits `2b572c9` → `98e0c39`, DB migration applied and downgraded). Re-tested against the real rewritten-query text afterward and confirmed the regression reproduces there too (0.20→0.13) — not a fluke.

**Root cause, confirmed by sweeping the retrieval window from k=5 to k=100**: recall climbs slowly and plateaus at **~0.38 even at k=100** (out of 1280 chunks) — meaning roughly **62% of expected citations are never matched by BM25 at all**, regardless of how wide the candidate window is. This rules out any ranking/cutoff tuning as a path forward — it isn't that the right chunk is ranked too low, it's that lexical keyword search structurally cannot find a majority of the answers a paraphrased or cross-referencing legal question needs. Widening `top_k_final` in `retriever.py` also wouldn't help the `recall_at_k` metric even if it could raise real recall — the eval measures a fixed top-5 regardless of how many chunks `/chat` returns.

**Conclusion**: BM25-only retrieval in prod is capped well below SPEC's targets by design, not by a fixable bug, and no further free/code-only lever was found after a genuine attempt. The only real fix is dense retrieval, which needs the Standard tier's 2 GB RAM (see corrected rationale table above). Project owner's decision as of 2026-08-30: **accept the current BM25-only state and keep this documented rather than pay for the upgrade** — this ADR's original decision stands, now with real measurements instead of a pre-launch estimate. Revisit if/when the project owner decides the $25/mo is worth it; at that point, unsetting the two env vars per the Consequences section above is still the entire re-enable path — no code change needed.
