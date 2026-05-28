"""Feature matrix assembly for tabular model training (T-15).

Loads features.nta_week_panel from the database into a pandas DataFrame,
derives the borough column, defines the canonical feature and label column
lists, and provides a chronological train/test split.

Usage::

    from rat_ml.features.feature_matrix import (
        load_feature_matrix,
        FEATURE_COLS,
        LABEL_COL,
        train_test_split,
    )

    df = asyncio.run(load_feature_matrix(db_url))
    train_df, test_df = train_test_split(df)
"""

from __future__ import annotations

import asyncpg
import pandas as pd

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

LABEL_COL = "active_rat_signs_ind"

# Columns that are identifiers or derived labels — never used as features.
_NON_FEATURE_COLS = frozenset(
    {
        "nta_id",
        "week_start",
        # Labels
        "active_rat_signs_ind",
        "active_rat_signs_count",
        "active_rat_signs_rate",
        "inspections_count",
    }
)

# All panel columns that are valid predictors (no Clay PCA — added dynamically).
# Ordered roughly: temporal lags → cross-domain signals → weather → static → spatial → regime.
FEATURE_COLS: list[str] = [
    # Derived
    "borough",
    # 311 lags
    "complaints_count",
    "complaints_lag_1w",
    "complaints_lag_4w",
    "complaints_lag_12w",
    # Cross-domain
    "rest_pest_violations_count",
    "permits_active_count",
    "demolitions_count",
    # Weather
    "weather_tavg_c",
    "weather_prcp_mm",
    "weather_hdd",
    "weather_cdd",
    # Static (PLUTO)
    "units_total",
    "year_built_median",
    "landuse_residential_pct",
    "landuse_commercial_pct",
    # Spatial lags
    "neighbor_active_rat_signs_rate_lag_1w",
    "neighbor_complaints_count_lag_4w",
    # Regime indicators
    "regime_covid",
    "regime_8pm_setout",
    "regime_commercial_containerization",
    "regime_residential_containerization",
    "regime_rmz_active",
]

# Categorical columns (passed to CatBoost/LightGBM as cat_features).
CAT_FEATURE_COLS: list[str] = ["borough"]

# Borough prefix → integer mapping for LR (which can't accept string categoricals).
BOROUGH_MAP: dict[str, int] = {"MN": 0, "BX": 1, "BK": 2, "QN": 3, "SI": 4}


# ---------------------------------------------------------------------------
# Database load
# ---------------------------------------------------------------------------

async def load_feature_matrix(db_url: str) -> pd.DataFrame:
    """Pull features.nta_week_panel from the database and return a DataFrame.

    Adds a ``borough`` column derived from the NTA prefix (first two chars of
    ``nta_id``).  Clay PCA columns (``clay_pca_0`` … ``clay_pca_31``) are
    included if present in the table.

    Args:
        db_url: asyncpg-compatible connection string.

    Returns:
        DataFrame sorted by ``week_start`` then ``nta_id``, with
        ``week_start`` as ``datetime64[ns]`` dtype.
    """
    conn = await asyncpg.connect(db_url)
    try:
        rows = await conn.fetch("SELECT * FROM features.nta_week_panel")
    finally:
        await conn.close()

    df = pd.DataFrame([dict(r) for r in rows])
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["borough"] = df["nta_id"].str[:2]

    # asyncpg returns NUMERIC columns as Decimal objects and all-NULL columns
    # as object dtype; cast every non-string column to float so LightGBM /
    # scikit-learn don't reject them.
    _STRING_COLS = frozenset({"nta_id", "borough"})
    for col in df.columns:
        if df[col].dtype == object and col not in _STRING_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Cast boolean regime columns to int so tree models handle them correctly.
    bool_cols = [c for c in df.columns if c.startswith("regime_")]
    for col in bool_cols:
        df[col] = df[col].astype("Int8")

    df = df.sort_values(["week_start", "nta_id"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Clay PCA feature detection
# ---------------------------------------------------------------------------

def clay_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return Clay PCA column names present in *df*, or an empty list."""
    return [c for c in df.columns if c.startswith("clay_pca_")]


def effective_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return FEATURE_COLS + any Clay PCA columns present in *df*."""
    present = set(df.columns)
    base = [c for c in FEATURE_COLS if c in present]
    clay = clay_feature_cols(df)
    return base + clay


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def train_test_split(
    df: pd.DataFrame,
    *,
    date_col: str = "week_start",
    holdout_weeks: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: last *holdout_weeks* ISO weeks become the test set.

    Args:
        df:             Full feature matrix (output of :func:`load_feature_matrix`).
        date_col:       Date column name.
        holdout_weeks:  Number of trailing weeks reserved for the test set.

    Returns:
        ``(train_df, test_df)`` both sorted by *date_col*.
    """
    from rat_ml.eval.timeseries_cv import holdout_split  # noqa: PLC0415

    return holdout_split(df, date_col=date_col, holdout_weeks=holdout_weeks)
