# ADR-0003: ACS median income via Census API, not pre-downloaded CSV

**Status**: Accepted
**Date**: 2026-05-01
**Deciders**: project owner

---

## Context

The reporting-bias notebook (T-12) and the NTA-week panel need ACS 5-year median household income
at the census tract level (Table B19013), then aggregated to NTAs using the T-03 crosswalk.

Two approaches were evaluated:

1. **Pre-downloaded CSV** from NYC Planning or Census Bureau website — zero API dependency, but
   requires a manual refresh step and a committed or tracked binary artifact.
2. **Census Bureau API** (`api.census.gov`) — pulls ACS 5-year estimates on demand; requires a
   free `CENSUS_API_KEY`.

## Decision

Use the Census Bureau API (`api.census.gov`) with the `census` Python library, pulling
ACS 5-year Table B19013 at the census tract level for New York State (FIPS 36).

## Rationale

- **Reproducibility**: any contributor can re-run the notebook against the latest ACS release
  without manually downloading and placing a file.
- **No binary artifacts**: avoids committing a CSV to the repo or managing it via DVC.
- **Free API key**: `CENSUS_API_KEY` is a free, instant signup; already added to `.env.example`.
- **Simple query**: `B19013_001E` (median household income estimate) is a single-variable pull;
  the `census` library wraps it in ~5 lines.

## Consequences

- `CENSUS_API_KEY` is a required env var for T-12 and any script that assembles income features.
- The `census` package is a required dependency of `rat-ml` (already added in T-01).
- ACS data vintage is fixed to the 5-year release available at time of pull; re-runs on different
  dates may produce slightly different estimates if a new release drops between runs. Acceptable
  for a portfolio project; document the vintage used in the notebook output.
- Tract-to-NTA aggregation uses the area-weighted crosswalk from T-03.
