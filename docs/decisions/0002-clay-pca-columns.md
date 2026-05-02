# ADR-0002: Use VECTOR(32) for Clay PCA embeddings

**Status**: Accepted
**Date**: 2026-05-01
**Deciders**: project owner

---

## Context

The spec calls for a 32-dimensional PCA projection of Clay v1.5 satellite embeddings to be stored
in `features.nta_week_panel`. Two storage options were considered:

1. **32 separate `NUMERIC` columns** (`clay_pca_0` through `clay_pca_31`) — matches the spec's
   default DDL sketch.
2. **A single `VECTOR(32)` column** — uses the pgvector extension already enabled in Supabase.

The `features.nta_week_panel` table also stores BM25/dense retrieval vectors elsewhere in the
pipeline, so pgvector is already a hard dependency.

## Decision

Use a single `VECTOR(32)` column named `clay_pca_embedding` in `features.nta_week_panel`.

## Rationale

- **Cleaner DDL**: one column instead of 32; schema remains readable as new PCA dimensions are
  explored.
- **Native pgvector operations**: cosine similarity queries and ANN index support work directly
  on the column without unpacking.
- **Consistency**: the RAG pipeline already uses `VECTOR` columns in `app.health_code_chunks`;
  using the same type in `features` avoids a two-type impedance mismatch.
- **No performance downside** at our data volume (< 200 NTAs × 3 years of weeks ≈ ~30k rows).

## Consequences

- The Alembic migration (T-02) creates `clay_pca_embedding VECTOR(32)` instead of 32 numeric
  columns.
- The Clay embedding script (Phase 3) writes a numpy array as a pgvector-compatible list.
- Querying individual PCA dimensions requires `clay_pca_embedding[n]` slice syntax (pgvector
  1-indexed), which is slightly less obvious than a named column — acceptable given the cleaner
  schema.
