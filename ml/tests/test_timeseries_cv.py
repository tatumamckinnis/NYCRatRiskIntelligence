"""Tests for time-series CV utilities (T-14)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from rat_ml.eval.timeseries_cv import CVSplit, expanding_window_splits, holdout_split
from rat_ml.eval.metrics import (
    brier_score,
    metric_bundle,
    pr_auc,
    roc_auc,
    top_decile_lift,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_panel(n_weeks: int, n_ntas: int = 5) -> pd.DataFrame:
    """Synthetic NTA-week panel with a week_start column."""
    weeks = [date(2021, 1, 4) + timedelta(weeks=i) for i in range(n_weeks)]
    rows = [
        {"nta_id": f"MN{nta:04d}", "week_start": w, "label": int((nta + i) % 3 == 0)}
        for i, w in enumerate(weeks)
        for nta in range(n_ntas)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# holdout_split
# ---------------------------------------------------------------------------

def test_holdout_split_sizes() -> None:
    df = _make_panel(52)
    train, test = holdout_split(df, holdout_weeks=12)
    assert len(test) > 0
    assert len(train) > 0
    assert len(train) + len(test) == len(df)


def test_holdout_split_no_overlap() -> None:
    df = _make_panel(52)
    train, test = holdout_split(df, holdout_weeks=12)
    assert train["week_start"].max() < test["week_start"].min()


def test_holdout_split_test_is_last_12_weeks() -> None:
    df = _make_panel(52)
    train, test = holdout_split(df, holdout_weeks=12)
    cutoff = pd.Timestamp(df["week_start"].max()) - timedelta(weeks=12)
    assert (test["week_start"] > cutoff).all()
    assert (train["week_start"] <= cutoff).all()


# ---------------------------------------------------------------------------
# expanding_window_splits
# ---------------------------------------------------------------------------

def test_cv_yields_correct_fold_count() -> None:
    df = _make_panel(104)
    train_df, _ = holdout_split(df, holdout_weeks=12)
    splits = list(expanding_window_splits(train_df, n_folds=5, gap_days=28))
    assert len(splits) == 5


def test_cv_gap_enforced() -> None:
    df = _make_panel(104)
    train_df, _ = holdout_split(df, holdout_weeks=12)
    for split in expanding_window_splits(train_df, n_folds=5, gap_days=28):
        assert split.gap_days >= 28, (
            f"Fold {split.fold}: gap {split.gap_days}d < 28d"
        )


def test_cv_no_date_overlap() -> None:
    """Train and validation windows must never share a date."""
    df = _make_panel(104)
    train_df, _ = holdout_split(df, holdout_weeks=12)
    for split in expanding_window_splits(train_df, n_folds=5, gap_days=28):
        train_dates = set(train_df.loc[split.train_idx, "week_start"])
        val_dates = set(train_df.loc[split.val_idx, "week_start"])
        overlap = train_dates & val_dates
        assert not overlap, (
            f"Fold {split.fold}: {len(overlap)} dates overlap between train and val"
        )


def test_cv_train_always_before_val() -> None:
    df = _make_panel(104)
    train_df, _ = holdout_split(df, holdout_weeks=12)
    for split in expanding_window_splits(train_df, n_folds=5, gap_days=28):
        max_train = train_df.loc[split.train_idx, "week_start"].max()
        min_val = train_df.loc[split.val_idx, "week_start"].min()
        assert max_train < min_val, (
            f"Fold {split.fold}: train end {max_train} >= val start {min_val}"
        )


def test_cv_training_window_expands() -> None:
    """Each successive fold must have a strictly larger training set."""
    df = _make_panel(104)
    train_df, _ = holdout_split(df, holdout_weeks=12)
    splits = list(expanding_window_splits(train_df, n_folds=5, gap_days=28))
    sizes = [len(s.train_idx) for s in splits]
    assert sizes == sorted(sizes), f"Training window did not expand: {sizes}"


def test_cv_raises_on_too_few_weeks() -> None:
    df = _make_panel(10)  # far too small
    with pytest.raises(ValueError, match="Not enough weeks"):
        list(expanding_window_splits(df, n_folds=5, gap_days=28))


def test_cv_raises_on_negative_gap() -> None:
    df = _make_panel(104)
    with pytest.raises(ValueError, match="gap_days"):
        list(expanding_window_splits(df, n_folds=3, gap_days=0))


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def _binary_arrays(n: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    y_prob = np.clip(y_true * 0.6 + rng.uniform(0, 0.4, size=n), 0, 1)
    return y_true, y_prob


def test_pr_auc_range() -> None:
    y_true, y_prob = _binary_arrays()
    score = pr_auc(y_true, y_prob)
    assert 0.0 <= score <= 1.0


def test_roc_auc_range() -> None:
    y_true, y_prob = _binary_arrays()
    score = roc_auc(y_true, y_prob)
    assert 0.0 <= score <= 1.0


def test_brier_score_range() -> None:
    y_true, y_prob = _binary_arrays()
    score = brier_score(y_true, y_prob)
    assert 0.0 <= score <= 1.0


def test_top_decile_lift_random_model() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=1000)
    y_prob = rng.uniform(0, 1, size=1000)
    lift = top_decile_lift(y_true, y_prob)
    # Random model should give lift ≈ 1.0 (±0.5 with 1000 samples)
    assert 0.4 < lift < 2.0


def test_top_decile_lift_perfect_model() -> None:
    y_true = np.array([1] * 100 + [0] * 900)
    y_prob = np.array([0.9] * 100 + [0.1] * 900)
    lift = top_decile_lift(y_true, y_prob)
    assert lift == pytest.approx(10.0, abs=0.01)


def test_metric_bundle_keys() -> None:
    y_true, y_prob = _binary_arrays()
    bundle = metric_bundle(y_true, y_prob)
    assert set(bundle.keys()) == {"pr_auc", "roc_auc", "brier", "top_decile_lift"}


def test_metric_bundle_better_than_random() -> None:
    """A correlated y_prob should outscore a random one on PR-AUC."""
    y_true, y_prob_good = _binary_arrays()
    rng = np.random.default_rng(99)
    y_prob_random = rng.uniform(0, 1, size=len(y_true))
    assert pr_auc(y_true, y_prob_good) > pr_auc(y_true, y_prob_random)
