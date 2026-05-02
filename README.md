# NYC Rat Risk Intelligence

> Multi-modal rat risk prediction and RAG platform over NYC public data.

**Status**: Phase 1 — Data Foundation (in progress)

---

## What this is

NTA-level (Neighborhood Tabulation Area) rat risk choropleth, factor attribution, 12-week forecast,
and cited legal Q&A chat over the NYC Health Code.

Supervised label: DOHMH rodent inspection `RESULT == 'Active Rat Signs'` — not 311 call volume.

---

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Python 3.12 + Pydantic v2 |
| Frontend | Next.js 16 + TypeScript + Tailwind 4 + MapLibre GL JS |
| Database | Supabase Free (Postgres 15 + PostGIS + pgvector) |
| ML | CatBoost / LightGBM / TFT + Chronos-2 ensemble |
| RAG | Hybrid BM25 + dense (pgvector HNSW) + BGE Reranker v2-M3 |
| LLM | Claude Haiku 4.5 (generation); Claude Sonnet 4.5 (eval judge) |
| Hosting | Render free tier (API) + Vercel Hobby (web) |

---

## Quickstart

```bash
# Python environment (requires uv)
uv sync --all-packages

# Run tests
uv run pytest

# Start API (dev)
uv run --package rat-api uvicorn rat_api.main:app --reload

# JS environment (requires pnpm + Node 22)
pnpm install
pnpm --filter web dev
```

---

## Architecture

See [`SPEC.md`](SPEC.md) for full build specification and [`docs/decisions/`](docs/decisions/) for ADRs.

---

## License

MIT
