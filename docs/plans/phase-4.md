# Phase 4 — RAG + Observability

**Goal**: cited-answer `/chat` endpoint over the NYC Health Code, fully instrumented with OpenInference tracing, with a passing eval suite (Recall@5 ≥ 0.70).

**Acceptance check**: `curl -N localhost:8000/chat` with a known-answerable question streams a response with at least one correct citation; eval suite Recall@5 ≥ 0.70 on first run.

---

## Status: what's already done

From the Phase 3 commits:
- `api/src/rat_api/rag/retrieve.py` — vector search (pgvector HNSW cosine) + Cohere Rerank 3.5 (optional) ✅
- `ml/src/rat_ml/rag/corpus.py` — static web-scrape corpus builder (not PDF-based) ✅
- `ml/src/rat_ml/rag/embed.py` — Voyage AI embedding client ✅
- `ml/src/rat_ml/rag/store.py` — upsert to `app.health_code_chunks` ✅
- `ml/scripts/build_rag_corpus.py` — CLI entry point ✅
- `api/src/rat_api/routes/narrative.py` — per-NTA narrative via Groq (not the chat endpoint) ✅
- `app.health_code_chunks` table with HNSW + tsvector indexes ✅
- `app.chat_sessions` and `app.chat_messages` tables ✅
- Phoenix in `docker-compose.yml` ✅

**Still to build**: PDF ingest scripts, BGE-M3 ablation embeddings, BM25 + RRF + BGE Reranker retrieval pipeline, query rewriting, parent-chunk expansion, OpenInference tracing, `/chat` SSE endpoint, eval gold set, eval runners.

---

## Task List

Complexity: **S** (< 1 hr), **M** (1–3 hr), **L** (3–6 hr)

---

### T-37 — PDF corpus ingest scripts
**Complexity**: L
**Dependencies**: Phase 3 complete
**Files created**:
- `ml/scripts/rag/ingest_health_code_151.py`
- `ml/scripts/rag/ingest_hmc_27_2017.py`
- `ml/scripts/rag/ingest_rcny_81_23.py`
- `ml/scripts/rag/ingest_ecb_penalties.py`
- `ml/scripts/rag/ingest_rodent_academy.py`
- `ml/src/rat_ml/rag/pdf_parser.py` *(shared PDF parsing utilities)*

**Description**:

**`pdf_parser.py`** — shared utilities:
- `parse_pdf(path) -> list[str]`: `pdfplumber` primary, `pypdf` fallback; OCR only for image-based PDFs.
- `parse_legal_hierarchy(text) -> list[LegalSection]`: regex for `§\d+\.\d+`, subsection markers `(a)`, `(1)`, `(i)`. Returns tree of `LegalSection(citation, title, content, depth)`.
- `chunk_section(section, max_tokens=600, overlap_pct=0.12) -> list[Chunk]`: leaf subsection if ≤600 tokens; sliding-window split otherwise. Parents at 1500–2500 tokens.
- `build_contextual_prefix(authority, document, citation, content) -> str`: `"From <authority> <document> <citation>: <content>"`.
- `extract_defined_terms(text) -> dict[str, str]`: extract `"term" means ...` and `"term" shall mean ...` patterns.
- `extract_cross_refs(text) -> list[str]`: regex for `§\d+[\.\-]\d+`, `Article \d+`, `Title \d+`.
- `version_hash(content_with_prefix) -> str`: SHA-256 hex.

Each ingest script:
1. Downloads or reads the source PDF.
2. Calls `parse_pdf` → `parse_legal_hierarchy` → `chunk_section`.
3. For each chunk: builds contextual prefix, extracts defined terms and cross-refs, computes version hash.
4. Calls `store.upsert_chunks()` — skips if `version_hash` matches existing row (idempotent).

**Sources** (per SPEC §7.1):
- `ingest_health_code_151.py` — NYC Health Code Title 24 Article 151; authority = `"DOHMH"`, document = `"Health Code Title 24 Article 151"`.
- `ingest_hmc_27_2017.py` — HMC §§27-2017 through 27-2018.1; authority = `"HPD"`, document = `"Housing Maintenance Code"`.
- `ingest_rcny_81_23.py` — 24 RCNY §81.23 IPM for food establishments; authority = `"DOHMH"`, document = `"24 RCNY §81.23"`.
- `ingest_ecb_penalties.py` — ECB/OATH penalty schedule; authority = `"ECB"`, document = `"ECB Penalty Schedule"`.
- `ingest_rodent_academy.py` — DOHMH Rodent Academy PDFs; authority = `"DOHMH"`, document = `"Rodent Academy"`.

---

### T-38 — BGE-M3 ablation embeddings
**Complexity**: S
**Dependencies**: T-37
**Files modified**:
- `ml/src/rat_ml/rag/embed.py`
- `ml/scripts/build_rag_corpus.py`

**Description**:
- Add `BgeMThreeEmbedder` class using `sentence-transformers` with `BAAI/bge-m3`. Loads model at init; `encode(texts) -> list[list[float]]`.
- Update `build_rag_corpus.py` to call both Voyage and BGE-M3 embedders in sequence. Store Voyage embeddings in `embedding` column, BGE-M3 in `embedding_bge` column.
- Batch size 128 for both, exponential backoff (max 5 retries) on Voyage 429s.
- `embedding_bge` column already exists in `0001_initial_schemas.py`.

---

### T-39 — Hybrid retrieval: BM25 + RRF + BGE Reranker + parent expansion
**Complexity**: L
**Dependencies**: T-37, T-38
**Files created/modified**:
- `api/src/rat_api/rag/retriever.py` *(rewrite)*
- `api/src/rat_api/rag/reranker.py` *(new — BGE Reranker v2-M3)*

**Description**:

**`reranker.py`** — BGE Reranker v2-M3:
- `BgeReranker`: loads `BAAI/bge-reranker-v2-m3` via `sentence-transformers.CrossEncoder` at module import (loaded once at API lifespan startup).
- `rerank(query, chunks, top_k) -> list[RetrievedChunk]`: scores each (query, chunk.content) pair, returns top_k sorted by score.
- If `COHERE_API_KEY` is present in env, uses Cohere Rerank 3.5 instead (ablation path only).

**`retriever.py`** — full hybrid pipeline, public function:
```python
async def retrieve(
    query: str,
    conn: asyncpg.Connection,
    *,
    top_k_dense: int = 30,
    top_k_bm25: int = 30,
    top_k_after_rrf: int = 40,
    top_k_final: int = 6,
    expand_parents: bool = True,
) -> list[RetrievedChunk]
```

Steps:
1. **Query rewriting**: Groq `llama-3.1-8b-instant` with a statutory vocabulary expansion prompt (≤200 tokens). Log rewrite as a span attribute. (Originally planned as Claude Haiku — see ADR 0006.)
2. **Dense retrieval**: embed rewritten query via Voyage `input_type="query"`, pgvector HNSW cosine top-30.
3. **BM25 retrieval**: `plainto_tsquery` with `ts_rank_cd` on `tsvector` index, top-30.
4. **RRF fusion**: `score(d) = Σ 1/(k + rank_i(d))` with `k=60`; take top-40 unique chunks.
5. **Rerank**: `BgeReranker.rerank(query, top-40, top_k=6)`.
6. **Parent expansion**: for each final chunk where `parent_chunk_id IS NOT NULL`, fetch parent and prepend to context if within token budget (total context ≤ 4000 tokens). Do not duplicate content.

`RetrievedChunk` dataclass: `chunk_id, document, citation, authority, section_path, content, content_with_prefix, score, retrieval_method`.

---

### T-40 — OpenInference observability
**Complexity**: M
**Dependencies**: none (can be done in parallel with T-39)
**Files created**:
- `api/src/rat_api/obs/__init__.py`
- `api/src/rat_api/obs/tracing.py`

**Description**:

**`tracing.py`**:
- `setup_tracing(service_name: str)`: configures a `TracerProvider` with two `BatchSpanProcessor`s:
  1. `OTLPSpanExporter(endpoint=settings.otel_endpoint)` — Phoenix OTLP HTTP. No-op if `OTEL_EXPORTER_OTLP_ENDPOINT` is unset.
  2. `JsonlSpanExporter(path=settings.obs_jsonl_path)` — custom exporter that writes one flattened JSON object per span to a JSONL file at `OBS_JSONL_PATH`. Flattens all span attributes.
- `get_tracer(name) -> opentelemetry.trace.Tracer`: module-level helper.
- Span kind helpers: `chain_span(name)`, `llm_span(name)`, `retriever_span(name)`, `reranker_span(name)` — context managers that set `openinference.span.kind` attribute.

Required attributes per SPEC §9.1:
- Retriever spans: `retrieval.documents.N.document.{id,content,score,metadata.citation,metadata.authority}`, `retrieval.method`, `retrieval.top_k`, `retrieval.score_distribution.{min,p50,max}`.
- LLM spans: `llm.model_name`, `llm.provider`, `llm.input_messages`, `llm.output_messages`, `llm.token_count.{prompt,completion,total}`, `llm.usd_cost`.

Add `OTEL_EXPORTER_OTLP_ENDPOINT` and `OBS_JSONL_PATH` fields to `api/src/rat_api/config.py`. Call `setup_tracing()` in the FastAPI lifespan.

---

### T-41 — `/chat` SSE endpoint
**Complexity**: M
**Dependencies**: T-39, T-40
**Files created**:
- `api/src/rat_api/rag/generator.py`
- `api/src/rat_api/rag/prompts.py`
- `api/src/rat_api/routes/chat.py`
**Files modified**:
- `api/src/rat_api/main.py` *(register chat router)*

**Description**:

**`prompts.py`** — system prompt constants:
```python
CHAT_SYSTEM_PROMPT = """You are a legal assistant answering questions about NYC rodent regulations. You
MUST cite every factual claim using the §<citation> format provided in the
retrieved chunks. If the answer is not supported by the retrieved chunks, say
so explicitly; do not speculate. Format: one short answer paragraph, followed
by a "Sources:" list of citations with brief quotes."""
```

**`generator.py`**:
- `async def generate_stream(query, chunks, *, session_id) -> AsyncIterator[str]`:
  - Builds messages list from `CHAT_SYSTEM_PROMPT` + retrieved chunks formatted as context + user query.
  - Calls `litellm.acompletion()` with `model="groq/llama-3.3-70b-versatile"` and streaming. (Originally planned as Claude Haiku — see ADR 0006.)
  - Wraps in an `llm_span` that records `llm.token_count.*` and `llm.usd_cost` (logged as 0.0 on Groq free tier).
  - Persists user message and assistant response to `app.chat_messages` when stream completes.

**`routes/chat.py`** — `POST /chat`:
- Request body: `ChatRequest(question: str, session_id: UUID | None)`. Creates session if `session_id` is None.
- Wraps the full request in a root `chain_span`.
- Calls `retrieve(question, conn)` (in a `retriever_span`).
- Calls `generate_stream(question, chunks, session_id=session_id)`.
- Returns `StreamingResponse(media_type="text/event-stream")` that yields `data: <token>\n\n` SSE frames. Final frame: `data: [DONE]\n\n`.
- `X-Session-Id` response header set to the session UUID.

---

### T-42 — Eval gold set + runners
**Complexity**: M
**Dependencies**: T-41
**Files created**:
- `evals/gold/article151_qa_v1.jsonl` *(50 hand-authored items)*
- `evals/src/__init__.py`
- `evals/src/runners.py`
- `evals/src/metrics.py`
- `evals/src/judge.py`

**Description**:

**`article151_qa_v1.jsonl`** — 50 items, ~8–9 per failure mode:
- Failure modes: cross-reference following, defined-term collision, section-boundary fragmentation, citation accuracy, multi-hop penalty lookup, vocabulary gap.
- Each item:
  ```json
  {
    "id": "article151-q-001",
    "question": "...",
    "expected_citations": ["§27-2018", "§27-2115"],
    "must_cite_at_least_one_of": [["§27-2018"], ["AH4D"]],
    "must_not_say": ["I don't know", "not specified"],
    "reference_answer": "...",
    "failure_mode": "multi_hop_penalty_lookup"
  }
  ```

**`metrics.py`**:
- `recall_at_k(expected_citations, retrieved_chunks, k) -> float`: fraction of `expected_citations` present in `retrieved_chunks[:k].citation`.
- `citation_accuracy(expected_citations, generated_answer) -> float`: regex match for each citation in generated text.
- `refusal_calibration(items, responses) -> float`: fraction of unanswerable items that correctly produced a "not in the documents" response.

**`judge.py`** — faithfulness judge:
- `async def judge_faithfulness(question, answer, chunks) -> int`: binary 0/1. Sends chunks + answer via `litellm` with `model="groq/llama-3.3-70b-versatile"` and a rubric prompt; parses `FAITHFUL: YES/NO`. (Originally planned as Claude Sonnet 4.5 — see ADR 0006.)

**`runners.py`**:
- `async def run_eval_suite(base_url, gold_path) -> dict`: loads JSONL, calls `POST /chat` for each question, collects retrieved chunks from JSONL trace sink (or via a `?debug=1` query param that returns chunks in response), computes all metrics.
- Writes results to `evals/results/<date>.json`.
- CLI: `uv run python -m evals.src.runners --base-url http://localhost:8000`.

---

### T-43 — RAG + API tests
**Complexity**: S
**Dependencies**: T-41
**Files created**:
- `ml/tests/test_rag_chunking.py`
- `api/src/tests/test_retriever.py`
- `api/src/tests/test_chat_route.py`

**Description**:
- `test_rag_chunking.py`: assert all chunks from `pdf_parser` are 50–600 tokens; assert `version_hash` is a 64-char hex; assert citations match `§\d+[\.\-]\d+` regex; assert `content_with_prefix` starts with `"From "`.
- `test_retriever.py`: smoke test with mocked DB that `retrieve()` returns `RetrievedChunk` list; assert query embedding calls use `input_type="query"` (not `"document"`).
- `test_chat_route.py`: mock `retrieve()` and `generate_stream()`, assert `/chat` returns `text/event-stream` content type and final `data: [DONE]` frame; assert `X-Session-Id` header is set.

---

## Task Dependencies (DAG)

```
T-37 (PDF ingest)
  └── T-38 (BGE-M3 embeddings)
        └── T-39 (hybrid retriever + RRF + BGE Reranker)
T-40 (OpenInference tracing) ← independent, can run in parallel with T-38/T-39
              │
        T-39 + T-40
              └── T-41 (/chat SSE endpoint)
                    ├── T-42 (eval gold set + runners)
                    └── T-43 (RAG + API tests)
```

---

## Files to Create (Complete List)

```
ml/src/rat_ml/rag/pdf_parser.py
ml/scripts/rag/__init__.py
ml/scripts/rag/ingest_health_code_151.py
ml/scripts/rag/ingest_hmc_27_2017.py
ml/scripts/rag/ingest_rcny_81_23.py
ml/scripts/rag/ingest_ecb_penalties.py
ml/scripts/rag/ingest_rodent_academy.py
api/src/rat_api/rag/reranker.py
api/src/rat_api/rag/generator.py
api/src/rat_api/rag/prompts.py
api/src/rat_api/routes/chat.py
api/src/rat_api/obs/__init__.py
api/src/rat_api/obs/tracing.py
evals/__init__.py
evals/gold/article151_qa_v1.jsonl
evals/src/__init__.py
evals/src/runners.py
evals/src/metrics.py
evals/src/judge.py
evals/results/.gitkeep
ml/tests/test_rag_chunking.py
api/src/tests/test_retriever.py
api/src/tests/test_chat_route.py
```

**Modified**:
- `ml/src/rat_ml/rag/embed.py` — add `BgeMThreeEmbedder`
- `ml/scripts/build_rag_corpus.py` — call both embedders
- `api/src/rat_api/rag/retriever.py` — full rewrite with BM25 + RRF + BGE Reranker + parent expansion
- `api/src/rat_api/main.py` — register chat router, load BGE Reranker at lifespan startup, call `setup_tracing()`
- `api/src/rat_api/config.py` — add `otel_endpoint`, `obs_jsonl_path`, `PRICE_TABLE`

---

## Open Questions

1. **PDF download strategy**: The five source PDFs may be behind redirects or require accepting terms. For each script, prefer a direct URL to the official PDF; fall back to a local `data/pdfs/<filename>.pdf` if the URL is unstable. Document the download URL in a comment at the top of each script.

2. **BGE Reranker memory**: `BAAI/bge-reranker-v2-m3` is ~570MB. Loading it at API startup on Render's free tier (512MB RAM) will OOM. **Resolved**: disable both BGE-M3 and the Reranker in prod via `DISABLE_VECTOR_SEARCH=true` and `DISABLE_RERANKER=true` env vars; BM25-only retrieval in production. See ADR 0007 (not the originally planned 0005-reranker-prod.md — that reference was superseded).

3. **JSONL trace sink path**: `OBS_JSONL_PATH` defaults to `obs/traces.jsonl` relative to the working directory. The eval runner reads from this file to get retrieved chunks per query. Ensure the path is writable in the Docker container (add a volume mount in `docker-compose.yml`).

4. **Gold set authoring**: The 50 eval items require reading the actual PDFs to write accurate questions, expected citations, and reference answers. Author these after T-37 is complete and the PDFs are parsed.

5. **`must_not_say` refusal calibration**: include ~5 questions that are genuinely unanswerable from the corpus (e.g. questions about city policy not covered in Health Code text) to measure refusal calibration.
