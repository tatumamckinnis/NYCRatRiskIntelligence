# ADR-0005 — Clay v1.5 Sentinel-2 track cut

**Status**: Accepted
**Date**: 2026-06-07
**Deciders**: project owner

---

## Context

The Phase 3 plan included building Clay v1.5 satellite embeddings (T-31): download Sentinel-2
quarterly composites per NTA, run them through the frozen Clay v1.5 encoder, reduce to 32-dim
PCA, and write `clay_pca_0..31` columns into `features.nta_week_panel`.

Three blockers surfaced during Phase 3 execution:

1. **Clay cannot be loaded via `AutoModel`**: The HuggingFace repo `made-with-clay/Clay` contains
   only `config.json` and `v1.5/clay-v1.5.ckpt` (a PyTorch Lightning checkpoint). No
   `modeling_*.py` files exist; `trust_remote_code=True` does not help. The `geovit+DOFA`
   architecture is not registered in `transformers` 4.57.6 (latest).

2. **Checkpoint is 5.16 GB** and requires installing `claymodel` from GitHub plus additional
   dependencies (`timm`, `einops`, `vit-pytorch`), with non-trivial risk of dependency conflicts
   with the existing TFT/Chronos stack.

3. **Band mismatch**: Clay v1.5 expects 10 Sentinel-2 bands for the `sentinel-2-l2a` platform
   (blue, green, red, rededge1–3, nir, nir08, swir16, swir22). Our composites contain only 7
   (B02, B03, B04, B08, B8A, B11, B12) — the three rededge bands are absent. Running Clay on
   mismatched bands without the full training setup would produce unreliable embeddings.

Additionally, the Sentinel-2 ingest (started 2026-06-05) hit repeated Microsoft Planetary
Computer SAS token expiry after ~44/262 NTAs were downloaded, limiting coverage to 17% of the
panel even if Clay had loaded correctly.

The spec (Section 16 — Cut line) lists the Clay v1.5 Sentinel-2 track as item #4 to drop when
schedule compresses, with the explicit note: "TFT + CatBoost still counts as multi-modal when
combined with restaurant-inspection channel."

The fusion meta-learner already achieves **PR-AUC = 0.7975** without any Clay features
(using zero-filled Clay columns as a fallback). The marginal gain from Clay is expected to be
small.

---

## Decision

**Cut the Clay v1.5 Sentinel-2 track.** The `clay_pca_0..31` columns remain in the schema
(no migration required) and will stay zero-filled. The Sentinel-2 ingest is stopped.

---

## Rationale

| Option | Pros | Cons |
|---|---|---|
| **Cut Clay (chosen)** | Unblocks Phase 4 immediately; no 5 GB download; no dependency risk | Loses spectral features; 17% NTA coverage anyway |
| Download checkpoint + fix band mismatch | Theoretically correct embeddings | 5.16 GB download; band mismatch requires hacks; 2–3 days of engineering |
| Switch to DINOv2 RGB proxy | Works with standard transformers | Only 3 channels (RGB); loses NIR/SWIR spectral signal; still only 44 NTAs |

---

## Consequences

- `clay_pca_0..31` columns remain in `features.nta_week_panel` with value `NULL` / `0`.
  CatBoost and the fusion model treat missing/zero satellite features gracefully.
- The Sentinel-2 quarterly composites already on disk (`data/sentinel2/`, 604 files) are kept
  for potential future use but are not gittracked.
- If Clay is revisited, the correct approach is: clone `github.com/Clay-foundation/model`,
  use `ClayMAE` with `clay_mae_large()`, download the `.ckpt`, pass a 7-band datacube with
  custom wavelengths via `DynamicEmbedding`.
- Phase 4 (RAG + Observability) proceeds immediately.
