"""Unit tests for TFT dataset preparation (T-29).

Skipped automatically when ``darts`` is not installed
(it lives in the optional ``[temporal]`` extras group).
"""

from __future__ import annotations

import pandas as pd
import pytest

darts = pytest.importorskip("darts", reason="darts[torch] not installed — skipping TFT tests")

from rat_ml.features.tft_dataset import (
    MIN_SERIES_LEN,
    PAST_COV_COLS,
    FUTURE_COV_COLS,
    build_tft_dataset,
    split_tft_dataset,
)


# ---------------------------------------------------------------------------
# Synthetic panel fixture
# ---------------------------------------------------------------------------

def _make_panel(n_weeks: int = 80, n_ntas: int = 3) -> pd.DataFrame:
    """Create a minimal synthetic NTA-week panel."""
    nta_ids = [f"MN{i:04d}" for i in range(n_ntas)]
    weeks = pd.date_range("2020-01-06", periods=n_weeks, freq="W-MON")

    rows = []
    for nta_id in nta_ids:
        for week in weeks:
            row: dict = {"nta_id": nta_id, "week_start": week, "active_rat_signs_ind": 0}
            for col in PAST_COV_COLS + FUTURE_COV_COLS:
                row[col] = 1.0
            rows.append(row)

    df = pd.DataFrame(rows)
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df


# ---------------------------------------------------------------------------
# build_tft_dataset
# ---------------------------------------------------------------------------


def test_build_returns_one_bundle_per_nta():
    df = _make_panel(n_weeks=80, n_ntas=3)
    dataset = build_tft_dataset(df)
    assert len(dataset.bundles) == 3


def test_build_filters_short_series():
    # 3 NTAs: two have 80 weeks (≥52), one has only 10
    df_long = _make_panel(n_weeks=80, n_ntas=2)
    df_short = _make_panel(n_weeks=10, n_ntas=1)
    df_short["nta_id"] = "XX9999"
    df = pd.concat([df_long, df_short], ignore_index=True)

    dataset = build_tft_dataset(df, min_len=MIN_SERIES_LEN)
    assert all(b.nta_id != "XX9999" for b in dataset.bundles)
    assert len(dataset.bundles) == 2


def test_build_target_series_length_matches_panel():
    n_weeks = 80
    df = _make_panel(n_weeks=n_weeks, n_ntas=1)
    dataset = build_tft_dataset(df)
    assert len(dataset.bundles[0].target) == n_weeks


def test_past_cov_series_has_correct_components():
    df = _make_panel(n_weeks=80, n_ntas=1)
    dataset = build_tft_dataset(df)
    bundle = dataset.bundles[0]
    assert bundle.past_covariates.n_components == len(PAST_COV_COLS)


def test_future_cov_series_has_correct_components():
    df = _make_panel(n_weeks=80, n_ntas=1)
    dataset = build_tft_dataset(df)
    bundle = dataset.bundles[0]
    assert bundle.future_covariates.n_components == len(FUTURE_COV_COLS)


def test_nta_ids_accessible():
    df = _make_panel(n_weeks=80, n_ntas=3)
    dataset = build_tft_dataset(df)
    ids = dataset.nta_ids()
    assert len(ids) == 3
    assert all(isinstance(i, str) for i in ids)


def test_targets_list_length():
    df = _make_panel(n_weeks=80, n_ntas=3)
    dataset = build_tft_dataset(df)
    assert len(dataset.targets) == len(dataset.bundles)


# ---------------------------------------------------------------------------
# split_tft_dataset
# ---------------------------------------------------------------------------


def test_split_train_shorter_than_full():
    df = _make_panel(n_weeks=80, n_ntas=2)
    dataset = build_tft_dataset(df)
    train, test = split_tft_dataset(dataset, holdout_weeks=12)
    for train_b, test_b in zip(train.bundles, test.bundles):
        assert len(train_b.target) == 80 - 12
        assert len(test_b.target) == 12


def test_split_no_temporal_overlap():
    df = _make_panel(n_weeks=80, n_ntas=1)
    dataset = build_tft_dataset(df)
    train, test = split_tft_dataset(dataset, holdout_weeks=12)
    train_end = train.bundles[0].target.end_time()
    test_start = test.bundles[0].target.start_time()
    assert test_start > train_end


def test_split_skips_too_short_series():
    # 30 weeks total, holdout=12 → only 18 train weeks; that's fine.
    # But if total ≤ holdout the bundle should be dropped.
    df = _make_panel(n_weeks=10, n_ntas=1)
    dataset = build_tft_dataset(df, min_len=1)  # bypass min_len for this test
    train, test = split_tft_dataset(dataset, holdout_weeks=12)
    # 10 ≤ 12 → bundle dropped
    assert len(train.bundles) == 0
    assert len(test.bundles) == 0
