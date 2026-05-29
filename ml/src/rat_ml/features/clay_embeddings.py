"""Frozen Clay v1.5 embeddings → 32-dim PCA → features.nta_week_panel (T-31).

For each NTA, reads the most recent available quarterly Sentinel-2 composite
(from data/sentinel2/<nta_id>/<quarter>/composite.tif), runs it through the
frozen Clay v1.5 encoder, applies PCA to reduce to 32 dimensions, and writes
clay_pca_0 … clay_pca_31 into features.nta_week_panel (all weeks for that NTA
receive the same value, since the raster is quarterly).

The PCA model is fit on all NTA embeddings, then saved to
ml/artifacts/clay_pca.joblib for reproducibility.

Usage (from repo root)::

    uv run --package rat-ml --extra vision python ml/scripts/build_clay_embeddings.py

Requires:
    uv run --package rat-ml --extra vision  (rasterio, stackstac)
    uv run --package rat-ml --extra temporal  (transformers, torch for Clay)

Clay model:
    HuggingFace: made-with-clay/Clay (Clay v1.5)
    Architecture: masked autoencoder (MAE) trained on multi-modal satellite data
    Input: 7-band Sentinel-2 patch at 10m resolution, 256×256 pixels
    Output: 768-dim CLS token embedding per patch

PCA:
    Fit on all available NTA embeddings; transform to 32 dims.
    Saved to ml/artifacts/clay_pca.joblib.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "data" / "sentinel2"
ARTIFACTS_DIR = REPO_ROOT / "ml" / "artifacts"
CLAY_PCA_PATH = ARTIFACTS_DIR / "clay_pca.joblib"

# Clay v1.5 HuggingFace model ID
CLAY_MODEL_ID = "made-with-clay/Clay"

# Input patch size expected by Clay (pixels at 10m resolution)
PATCH_SIZE = 256

# Number of PCA dimensions to retain
N_PCA_COMPONENTS = 32

# Band order matching ingest_sentinel2.py BANDS list
BANDS = ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]

# Mean/std normalization values for Clay v1.5 (from model card)
# These are per-band values in surface reflectance units (0–1 scaled, ÷10000)
CLAY_BAND_MEAN = [
    485.0,   # B02
    559.0,   # B03
    660.0,   # B04
    825.0,   # B08
    864.0,   # B8A
    1610.0,  # B11
    2200.0,  # B12
]
CLAY_BAND_STD = [
    277.0,   # B02
    234.0,   # B03
    228.0,   # B04
    356.0,   # B08
    299.0,   # B8A
    476.0,   # B11
    580.0,   # B12
]


# ---------------------------------------------------------------------------
# Clay model loading + inference
# ---------------------------------------------------------------------------

def _load_clay_model() -> Any:
    """Load the frozen Clay v1.5 encoder from HuggingFace."""
    import torch  # noqa: PLC0415
    from transformers import AutoModel  # noqa: PLC0415

    log.info("Loading Clay v1.5 from HuggingFace (%s) …", CLAY_MODEL_ID)
    model = AutoModel.from_pretrained(CLAY_MODEL_ID, trust_remote_code=True)
    model.eval()
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    log.info("Clay model loaded and frozen.")
    return model


def _load_raster(tif_path: Path) -> np.ndarray | None:
    """Load composite.tif and return a (7, H, W) float32 array, or None if unreadable."""
    import rasterio  # noqa: PLC0415

    try:
        with rasterio.open(tif_path) as src:
            data = src.read().astype("float32")  # (bands, H, W)
            nodata = src.nodata
    except Exception as exc:  # noqa: BLE001
        log.warning("Cannot read %s: %s", tif_path, exc)
        return None

    if nodata is not None:
        data[data == nodata] = float("nan")

    if data.shape[0] != len(BANDS):
        log.warning(
            "Expected %d bands in %s, got %d — skipping",
            len(BANDS), tif_path, data.shape[0],
        )
        return None

    return data


def _centre_crop(arr: np.ndarray, size: int) -> np.ndarray:
    """Centre-crop (C, H, W) to (C, size, size). Pad with NaN if smaller."""
    c, h, w = arr.shape
    if h < size or w < size:
        padded = np.full((c, size, size), float("nan"), dtype="float32")
        oh = min(h, size)
        ow = min(w, size)
        padded[:, :oh, :ow] = arr[:, :oh, :ow]
        return padded
    row_off = (h - size) // 2
    col_off = (w - size) // 2
    return arr[:, row_off : row_off + size, col_off : col_off + size]


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Per-band z-score normalisation using Clay's published statistics."""
    mean = np.array(CLAY_BAND_MEAN, dtype="float32").reshape(-1, 1, 1)
    std = np.array(CLAY_BAND_STD, dtype="float32").reshape(-1, 1, 1)
    arr = (arr - mean) / (std + 1e-6)
    # Replace NaN (nodata) with 0 after normalisation
    arr = np.nan_to_num(arr, nan=0.0)
    return arr


def _embed_patch(model: Any, patch: np.ndarray) -> np.ndarray:
    """Run one (7, 256, 256) patch through Clay; return 768-dim CLS embedding."""
    import torch  # noqa: PLC0415

    tensor = torch.from_numpy(patch).unsqueeze(0)  # (1, 7, 256, 256)

    with torch.no_grad():
        outputs = model(pixel_values=tensor)

    # Clay returns last_hidden_state: (batch, seq_len, hidden); CLS token is index 0
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).numpy()
    return cls_embedding.astype("float32")


def compute_embeddings(
    nta_ids: list[str],
    clay_model: Any,
) -> dict[str, np.ndarray]:
    """Return {nta_id: 768-dim embedding} for all NTAs with available rasters."""
    embeddings: dict[str, np.ndarray] = {}

    for nta_id in nta_ids:
        nta_dir = DATA_DIR / nta_id
        if not nta_dir.exists():
            log.debug("No raster directory for NTA %s — skipping", nta_id)
            continue

        # Use the most recent quarter with a valid composite
        quarters = sorted(nta_dir.iterdir(), reverse=True)
        raster: np.ndarray | None = None
        for q_dir in quarters:
            tif = q_dir / "composite.tif"
            if tif.exists():
                raster = _load_raster(tif)
                if raster is not None:
                    log.debug("Using %s for NTA %s", q_dir.name, nta_id)
                    break

        if raster is None:
            log.warning("No valid raster for NTA %s — skipping", nta_id)
            continue

        patch = _centre_crop(raster, PATCH_SIZE)
        patch = _normalise(patch)
        emb = _embed_patch(clay_model, patch)
        embeddings[nta_id] = emb
        log.info("Embedded NTA %s → shape %s", nta_id, emb.shape)

    return embeddings


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def fit_pca(
    embeddings: dict[str, np.ndarray],
) -> tuple["Any", dict[str, np.ndarray]]:
    """Fit PCA on all embeddings and return (pca_model, {nta_id: 32-dim array})."""
    from sklearn.decomposition import PCA  # noqa: PLC0415

    nta_ids = list(embeddings.keys())
    X = np.stack([embeddings[n] for n in nta_ids])  # (N, 768)

    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
    X_pca = pca.fit_transform(X)  # (N, 32)

    explained = pca.explained_variance_ratio_.sum()
    log.info(
        "PCA fit: %d NTAs, %d → %d dims, explained variance=%.1f%%",
        len(nta_ids), X.shape[1], N_PCA_COMPONENTS, explained * 100,
    )

    pca_dict = {nta_id: X_pca[i] for i, nta_id in enumerate(nta_ids)}
    return pca, pca_dict


def save_pca(pca: Any) -> None:
    """Persist the fitted PCA model to disk."""
    import joblib  # noqa: PLC0415

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, CLAY_PCA_PATH)
    log.info("PCA saved to %s", CLAY_PCA_PATH)


# ---------------------------------------------------------------------------
# DB update
# ---------------------------------------------------------------------------

async def _update_panel(
    pca_dict: dict[str, np.ndarray],
    db_url: str,
) -> int:
    """Write clay_pca_0..31 into features.nta_week_panel for all affected NTAs.

    All rows for a given NTA receive the same values (Clay embeddings are
    quarterly and don't vary week-to-week).

    Returns the number of rows updated.
    """
    import asyncpg  # noqa: PLC0415

    if not pca_dict:
        return 0

    set_clauses = ", ".join(f"clay_pca_{i} = ${ i + 2}" for i in range(N_PCA_COMPONENTS))
    sql = f"""
        UPDATE features.nta_week_panel
        SET {set_clauses}
        WHERE nta_id = $1
    """

    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute("SET statement_timeout = 0")
        total = 0
        for nta_id, pca_vec in pca_dict.items():
            params: list[Any] = [nta_id] + [float(v) for v in pca_vec]
            result = await conn.execute(sql, *params)
            parts = result.split()
            total += int(parts[-1]) if parts else 0
        return total
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run(db_url: str) -> dict[str, Any]:
    """Run the full Clay embedding + PCA pipeline.

    Returns summary dict with keys: n_embedded, n_pca, rows_updated.
    """
    import asyncpg  # noqa: PLC0415

    # Fetch NTA IDs from the panel
    conn = await asyncpg.connect(db_url)
    try:
        rows = await conn.fetch("SELECT DISTINCT nta_id FROM features.nta_week_panel ORDER BY nta_id")
    finally:
        await conn.close()

    nta_ids = [r["nta_id"] for r in rows]
    log.info("Panel has %d NTAs", len(nta_ids))

    clay_model = _load_clay_model()
    embeddings = compute_embeddings(nta_ids, clay_model)
    log.info("Computed embeddings for %d / %d NTAs", len(embeddings), len(nta_ids))

    if not embeddings:
        log.warning("No embeddings produced — is data/sentinel2/ populated?")
        return {"n_embedded": 0, "n_pca": 0, "rows_updated": 0}

    pca, pca_dict = fit_pca(embeddings)
    save_pca(pca)

    rows_updated = await _update_panel(pca_dict, db_url)
    log.info("Updated %d panel rows with Clay PCA features", rows_updated)

    return {
        "n_embedded": len(embeddings),
        "n_pca": len(pca_dict),
        "rows_updated": rows_updated,
    }
