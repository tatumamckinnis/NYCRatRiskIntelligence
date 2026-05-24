"""Convert the NTA-week panel into Darts TimeSeries objects for TFT training (T-23).

Design
------
- One ``TimeSeries`` per NTA, weekly frequency.
- **Target**: ``active_rat_signs_ind`` (binary 0/1).
- **Past covariates** (observed, not known in advance):
    complaints_count, complaints_lag_1w, complaints_lag_4w, complaints_lag_12w,
    rest_pest_violations_count, permits_active_count, demolitions_count,
    weather_tavg_c, weather_prcp_mm, weather_hdd, weather_cdd,
    neighbor_active_rat_signs_rate_lag_1w, neighbor_complaints_count_lag_4w
- **Future covariates** (known in advance — policy/calendar):
    regime_covid, regime_8pm_setout, regime_commercial_containerization,
    regime_residential_containerization, regime_rmz_active
- **Static covariates** (constant per NTA):
    borough (int-encoded), units_total, year_built_median,
    landuse_residential_pct, landuse_commercial_pct

Minimum series length is enforced so that TFT has enough context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from darts import TimeSeries

# Minimum number of weekly observations required to include a series.
MIN_SERIES_LEN: int = 52  # one year

PAST_COV_COLS: list[str] = [
    "complaints_count",
    "complaints_lag_1w",
    "complaints_lag_4w",
    "complaints_lag_12w",
    "rest_pest_violations_count",
    "permits_active_count",
    "demolitions_count",
    "weather_tavg_c",
    "weather_prcp_mm",
    "weather_hdd",
    "weather_cdd",
    "neighbor_active_rat_signs_rate_lag_1w",
    "neighbor_complaints_count_lag_4w",
]

FUTURE_COV_COLS: list[str] = [
    "regime_covid",
    "regime_8pm_setout",
    "regime_commercial_containerization",
    "regime_residential_containerization",
    "regime_rmz_active",
]

STATIC_COV_COLS: list[str] = [
    "units_total",
    "year_built_median",
    "landuse_residential_pct",
    "landuse_commercial_pct",
]

BOROUGH_MAP: dict[str, int] = {"MN": 0, "BX": 1, "BK": 2, "QN": 3, "SI": 4}


@dataclass
class NtaSeriesBundle:
    """Container for one NTA's Darts series objects."""

    nta_id: str
    target: "TimeSeries"
    past_covariates: "TimeSeries"
    future_covariates: "TimeSeries"
    static_covariates: "pd.DataFrame"  # 1-row DataFrame of floats


@dataclass
class TFTDataset:
    """Collection of per-NTA series bundles, ready for Darts TFTModel.fit()."""

    bundles: list[NtaSeriesBundle] = field(default_factory=list)

    # ── convenience accessors ──────────────────────────────────────────────

    @property
    def targets(self) -> "list[TimeSeries]":
        return [b.target for b in self.bundles]

    @property
    def past_covariates(self) -> "list[TimeSeries]":
        return [b.past_covariates for b in self.bundles]

    @property
    def future_covariates(self) -> "list[TimeSeries]":
        return [b.future_covariates for b in self.bundles]

    def nta_ids(self) -> list[str]:
        return [b.nta_id for b in self.bundles]


def _fill_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Forward-fill then back-fill *cols*, then replace remaining NaN with 0."""
    df = df.copy()
    df[cols] = df[cols].ffill().bfill().fillna(0.0)
    return df


def _to_timeseries(
    df_nta: pd.DataFrame,
    value_cols: list[str],
    freq: str = "W-MON",
) -> "TimeSeries":
    """Build a Darts TimeSeries from a per-NTA slice."""
    from darts import TimeSeries  # noqa: PLC0415

    return TimeSeries.from_dataframe(
        df_nta.set_index("week_start")[value_cols].astype("float32"),
        freq=freq,
        fill_missing_dates=True,
        fillna_value=0.0,
    )


def build_tft_dataset(
    df: pd.DataFrame,
    *,
    min_len: int = MIN_SERIES_LEN,
) -> TFTDataset:
    """Convert a full panel DataFrame into a :class:`TFTDataset`.

    Args:
        df:      Full NTA-week panel (from :func:`~rat_ml.features.feature_matrix.load_feature_matrix`).
        min_len: Discard NTAs with fewer than *min_len* weekly observations.

    Returns:
        :class:`TFTDataset` with one bundle per qualifying NTA.
    """
    import pandas as pd  # noqa: PLC0415 — already imported, but for TYPE_CHECKING guard

    df = df.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["borough_int"] = df["nta_id"].str[:2].map(BOROUGH_MAP).fillna(0).astype("float32")

    # Cast regime cols to float32
    for col in FUTURE_COV_COLS:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    # Fill numeric past covariates
    all_num_cols = PAST_COV_COLS + FUTURE_COV_COLS + STATIC_COV_COLS + ["active_rat_signs_ind"]
    present = [c for c in all_num_cols if c in df.columns]
    df = _fill_numeric(df, present)

    bundles: list[NtaSeriesBundle] = []

    for nta_id, grp in df.groupby("nta_id"):
        grp = grp.sort_values("week_start").reset_index(drop=True)

        if len(grp) < min_len:
            continue

        # Missing covariate columns → zero series
        for col in PAST_COV_COLS + FUTURE_COV_COLS + STATIC_COV_COLS:
            if col not in grp.columns:
                grp[col] = 0.0

        target = _to_timeseries(grp, ["active_rat_signs_ind"])
        past_cov = _to_timeseries(grp, PAST_COV_COLS)
        future_cov = _to_timeseries(grp, FUTURE_COV_COLS)

        # Static covariates: take the last (most recent) row's values
        static_vals = {
            "borough_int": float(grp["borough_int"].iloc[-1]),
            **{col: float(grp[col].iloc[-1]) for col in STATIC_COV_COLS},
        }
        static_df = pd.DataFrame([static_vals])

        bundles.append(
            NtaSeriesBundle(
                nta_id=str(nta_id),
                target=target,
                past_covariates=past_cov,
                future_covariates=future_cov,
                static_covariates=static_df,
            )
        )

    return TFTDataset(bundles=bundles)


def split_tft_dataset(
    dataset: TFTDataset,
    *,
    holdout_weeks: int = 12,
) -> tuple[TFTDataset, TFTDataset]:
    """Split each NTA series into train and test portions.

    Args:
        dataset:       Full :class:`TFTDataset`.
        holdout_weeks: Number of trailing weeks to reserve for test.

    Returns:
        ``(train_dataset, test_dataset)``
    """
    train_bundles: list[NtaSeriesBundle] = []
    test_bundles: list[NtaSeriesBundle] = []

    for b in dataset.bundles:
        n = len(b.target)
        if n <= holdout_weeks:
            # Too short to split — skip
            continue
        cut = n - holdout_weeks

        train_bundles.append(
            NtaSeriesBundle(
                nta_id=b.nta_id,
                target=b.target[:cut],
                past_covariates=b.past_covariates[:cut],
                future_covariates=b.future_covariates[:cut],
                static_covariates=b.static_covariates,
            )
        )
        test_bundles.append(
            NtaSeriesBundle(
                nta_id=b.nta_id,
                target=b.target[cut:],
                past_covariates=b.past_covariates[cut:],
                future_covariates=b.future_covariates[cut:],
                static_covariates=b.static_covariates,
            )
        )

    return TFTDataset(bundles=train_bundles), TFTDataset(bundles=test_bundles)
