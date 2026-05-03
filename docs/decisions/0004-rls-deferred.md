# ADR-0004 — Row-Level Security deferred to Phase 5

**Status**: Accepted
**Date**: 2026-05-02
**Deciders**: project owner

---

## Context

Supabase enables Row-Level Security (RLS) on a per-table basis. Enabling RLS
without at least one permissive policy causes every query to return zero rows,
so the decision must be deliberate.

This project is a **public read-only dashboard**. The backend (FastAPI on
Render) is the sole writer; the Supabase PostgREST endpoint is not exposed
directly to browsers. All reads from the frontend go through the FastAPI API,
which authenticates to Supabase with the service-role key server-side.

---

## Decision

RLS is **not enabled** in the initial migration (`0001_initial_schemas.py`) on
any table.

The one table that may need RLS in the future is `app.chat_sessions` /
`app.chat_messages`, if a future iteration exposes per-user chat history via
the Supabase JS client (which would require user-level JWT auth). That is
outside Phase 1–4 scope.

---

## Rationale

| Option | Pros | Cons |
|---|---|---|
| Enable RLS now (default-deny) | Defence-in-depth | Breaks all queries until every policy is written; adds migration complexity with no near-term benefit |
| Enable RLS now (default-allow) | Nominal compliance | False sense of security; indistinguishable from no RLS |
| **Defer to Phase 5** | Zero risk of silent query breakage; revisited when auth story is concrete | Supabase PostgREST endpoint must remain blocked via Supabase dashboard network rules |

---

## Consequences

- The Supabase **PostgREST** endpoint (`<project>.supabase.co/rest/v1/`) must
  be restricted to service-role key only. Confirm in the Supabase dashboard
  under **Project Settings → API → Exposed schemas** that only the
  `public` schema is exposed and the anon key cannot write.
- All data mutations flow through `api/` (FastAPI + SQLAlchemy), never through
  PostgREST directly from the browser.
- When RLS is revisited (Phase 5 hardening), the migration will:
  1. `ALTER TABLE app.chat_sessions ENABLE ROW LEVEL SECURITY;`
  2. `ALTER TABLE app.chat_messages ENABLE ROW LEVEL SECURITY;`
  3. Add policies scoped to `auth.uid()` via Supabase Auth JWT.
