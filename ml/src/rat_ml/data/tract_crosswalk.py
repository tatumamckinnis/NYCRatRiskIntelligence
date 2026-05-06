"""2010 → 2020 NTA crosswalk and area-weighted value allocation.

The NYC DCP crosswalk CSV maps each 2010 NTA to the 2020 NTAs it overlaps,
with a weight column expressing what fraction of the 2010 NTA's area falls
within each 2020 NTA.  Weights are normalized at load time so they sum to
exactly 1.0 per source NTA, handling any floating-point imprecision in the
source file.

Typical CSV columns (names are normalized to lowercase at load):
    nta2010      – source 2010 NTA code (e.g. 'BK0101')
    nta2020      – target 2020 NTA code (e.g. 'BK2201')
    overlap_pct  – fraction of the 2010 NTA's area covered by this 2020 NTA

Usage::

    from rat_ml.data.tract_crosswalk import load_crosswalk, allocate

    load_crosswalk(Path("data/nta_2010_2020_crosswalk.csv"))
    shares = allocate(1234.5, source_nta_2010="MN2501")
    # → {'MN2501': 980.2, 'MN2502': 254.3, ...}
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Column name variants seen in NYC DCP releases.  The first match wins.
_NTA2010_CANDIDATES = ["nta2010", "ntacode10", "nta10", "nta_2010"]
_NTA2020_CANDIDATES = ["nta2020", "ntacode20", "nta20", "nta_2020"]
_WEIGHT_CANDIDATES = ["overlap_pct", "pct_overlap", "percentage", "pct", "weight"]

_crosswalk: pd.DataFrame | None = None


def _pick_col(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(
        f"Cannot find {label} column in crosswalk CSV. "
        f"Tried: {candidates}. Got: {list(df.columns)}"
    )


def load_crosswalk(path: Path, weight_col: str | None = None) -> None:
    """Load and normalize the 2010→2020 NTA crosswalk CSV.

    Args:
        path: Path to the crosswalk CSV file.
        weight_col: Override the auto-detected weight column name.
    """
    global _crosswalk

    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()

    col_2010 = _pick_col(df, _NTA2010_CANDIDATES, "nta2010")
    col_2020 = _pick_col(df, _NTA2020_CANDIDATES, "nta2020")
    col_w = weight_col or _pick_col(df, _WEIGHT_CANDIDATES, "weight")

    df = df.rename(columns={col_2010: "nta2010", col_2020: "nta2020", col_w: "weight"})
    df = df[["nta2010", "nta2020", "weight"]].copy()

    # Normalize weights so they sum to exactly 1.0 per source NTA.
    df["weight"] = df["weight"].astype(float)
    totals = df.groupby("nta2010")["weight"].transform("sum")
    df["weight"] = df["weight"] / totals

    _crosswalk = df.reset_index(drop=True)


def _require_loaded() -> pd.DataFrame:
    if _crosswalk is None:
        raise RuntimeError("Crosswalk not loaded. Call load_crosswalk() first.")
    return _crosswalk


def weights_for(source_nta_2010: str) -> dict[str, float]:
    """Return normalized area weights mapping 2020 NTA codes → weight for a given 2010 NTA.

    Weights sum to 1.0.  Returns an empty dict if the source NTA is not in the crosswalk.
    """
    df = _require_loaded()
    rows = df[df["nta2010"] == source_nta_2010]
    return dict(zip(rows["nta2020"], rows["weight"]))


def allocate(value: float, source_nta_2010: str) -> dict[str, float]:
    """Distribute *value* from a 2010 NTA across 2020 NTAs using area-weighted allocation.

    The sum of allocated values equals *value* (subject to float precision).

    Returns:
        Mapping of {nta2020_code: allocated_value}.
        Empty dict if *source_nta_2010* is not in the crosswalk.
    """
    weights = weights_for(source_nta_2010)
    return {nta: value * w for nta, w in weights.items()}


def all_nta2010_codes() -> set[str]:
    """Return all 2010 NTA codes present in the loaded crosswalk."""
    return set(_require_loaded()["nta2010"].unique())


def all_nta2020_codes() -> set[str]:
    """Return all 2020 NTA codes present in the loaded crosswalk."""
    return set(_require_loaded()["nta2020"].unique())
