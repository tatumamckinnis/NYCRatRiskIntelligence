# ADR 0001 — Single-cloud deployment (Render + Vercel + Supabase)

**Status:** Accepted  
**Date:** 2026-05-01  
**Author:** Tatum McKinnis

---

## Context

The original SPEC called for a multi-cloud story: FastAPI on Fly.io, a Terraform config deploying the same Docker image to AWS ECS as a sidecar, and a Phoenix OTLP trace exporter running locally while a JSONL sink ran in prod.

At the time of the Phase 6 planning meeting (2026-05-01), the project was at $0 recurring infrastructure cost and the AWS ECS + Terraform path would have introduced:

- Non-trivial IAM/VPC complexity with no observable benefit for a solo project at demo traffic
- A second runtime to keep in sync with Render (env vars, image tags, healthchecks)
- At least $10–30/month in ECS Fargate costs even on minimal task sizes
- A Fly.io account alongside a Render account (duplicate configuration)

The sole stated reason for the multi-cloud sidecar in the spec was "portability note / multi-cloud story for the portfolio narrative." That benefit is real but not worth the operational overhead at this scale.

---

## Decision

Deploy to **one cloud per tier** using free-tier managed services:

| Tier | Provider | Rationale |
|---|---|---|
| API | Render free | Zero-config, auto-deploy from GitHub, 512 MB RAM sufficient for BM25-only RAG |
| Frontend | Vercel Hobby | Native Next.js support, global CDN, preview deployments free |
| Database | Supabase Free | PostGIS + pgvector + pg_trgm in one managed Postgres; 500 MB fits current corpus |

No AWS ECS. No Terraform. No Fly.io.

The JSONL trace sink (`docs/traces/`) remains as the sole observability backend in production. Phoenix OTLP is used in local development only.

---

## Portability note

The API is a standard ASGI app (`uvicorn rat_api.main:app`). Moving to any container host (Fly.io, Railway, AWS ECS, GCP Cloud Run) requires only:

1. Building the Docker image from `api/Dockerfile`
2. Passing the same five env vars (`DATABASE_URL`, `DIRECT_DATABASE_URL`, `GROQ_API_KEY`, `DISABLE_RERANKER`, `ML_ARTIFACTS_DIR`)
3. Updating `NEXT_PUBLIC_API_URL` in Vercel

The Supabase connection string is the only provider-specific dependency; a Neon or self-hosted Postgres with PostGIS + pgvector is a drop-in replacement.

---

## Consequences

**Good:**
- Infrastructure stays at $0/month indefinitely at demo traffic
- Zero operational overhead — no Terraform state, no IAM policies, no ECS task definitions
- GitHub push → Render/Vercel auto-deploy is the complete CI/CD story

**Acceptable tradeoffs:**
- Render free tier cold starts (~30s after 15 min idle); acceptable for demo traffic
- No horizontal scaling without a paid Render plan; acceptable for current load
- BGE-M3 and BGE Reranker disabled on free tier (512 MB RAM limit); BM25 retrieval only in prod

**Cuts documented as ADRs:**
- AWS ECS Terraform sidecar: this ADR (0001)
- Clay v1.5 embeddings: ADR 0005
