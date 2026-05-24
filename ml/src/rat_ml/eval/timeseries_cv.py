"""Expanding-window time-series cross-validation (T-14).

Produces train/validation index splits for time-ordered DataFrames with a
mandatory gap between the end of the training window and the start of the
validation window, preventing leakage through the reporting lag inherent in
DOHMH inspection data.

Usage::

    from rat_ml.eval.timeseries_cv import expanding_window_splits, holdout_split

    train_df, test_df = holdout_split(df, holdout_weeks=12)
    for train_idx, val_idx in expanding_window_splits(train_df, n_folds=5, gap_days=28):
        X_tr, y_tr = train_df.iloc[train_idx][FEATURE_COLS], train_df.iloc[train_idx][LABEL_COL]
        X_val, y_val = train_df.iloc[val_idx][FEATURE_COLS], train_df.iloc[val_idx][LABEL_COL]
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Iterator

import pandas as pd


@dataclass(frozen=True)
class CVSplit:
    """One train/validation split from expanding-window CV."""

    fold: int
    train_idx: pd.Index
    val_idx: pd.Index
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    gap_days: int


def holdout_split(
    df: pd.DataFrame,
    *,
    date_col: str = "week_start",
    holdout_weeks: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* chronologically: most recent *holdout_weeks* become the test set.

    Args:
        df:             DataFrame with a date column, not necessarily sorted.
        date_col:       Name of the column containing ISO week start dates.
        holdout_weeks:  Number of trailing weeks to reserve as the final test set.

    Returns:
        ``(train_df, test_df)`` — non-overlapping, sorted by *date_col*.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    cutoff = df[date_col].max() - timedelta(weeks=holdout_weeks)
    train = df[df[date_col] <= cutoff].copy()
    test = df[df[date_col] > cutoff].copy()
    return train, test


def expanding_window_splits(
    df: pd.DataFrame,
    *,
    date_col: str = "week_start",
    n_folds: int = 5,
    gap_days: int = 28,
    val_weeks: int = 4,
) -> Iterator[CVSplit]:
    """Yield expanding-window CV splits with a mandatory gap.

    The training window grows from left to right; each validation window
    immediately follows the gap.  The DataFrame must already have the test
    holdout removed (call :func:`holdout_split` first).

    Args:
        df:         Train DataFrame (holdout already removed), needs *date_col*.
        date_col:   Name of the ISO week-start column.
        n_folds:    Number of folds to yield.
        gap_days:   Minimum days between the last training date and the first
                    validation date.  Must be ≥ 1.  Default 28 (4 weeks) to
                    avoid inspection-reporting-lag leakage.
        val_weeks:  Width of each validation window in weeks.

    Yields:
        :class:`CVSplit` instances in ascending order of fold index.

    Raises:
        ValueError: if the DataFrame has too few distinct weeks to form *n_folds*
                    non-overlapping splits with the requested gap.
    """
    if gap_days < 1:
        raise ValueError(f"gap_days must be ≥ 1, got {gap_days}")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    weeks = sorted(df[date_col].unique())
    n_weeks = len(weeks)

    # Each fold needs at least val_weeks of validation data after the gap.
    # We divide the timeline into n_folds + 1 segments; the first segment is
    # the minimum initial training window, the rest are fold increments.
    gap = timedelta(days=gap_days)
    val_delta = timedelta(weeks=val_weeks)

    # Determine the earliest possible val_start for each fold by working
    # backwards from the end of the available weeks.
    total_val_span = val_weeks * n_folds
    if n_weeks <= total_val_span + gap_days // 7 + 4:
        raise ValueError(
            f"Not enough weeks ({n_weeks}) to form {n_folds} folds with "
            f"gap={gap_days}d and val_weeks={val_weeks}."
        )

    # Step size: how many weeks each fold advances the training cutoff.
    # We spread the validation windows evenly across the available timeline.
    weeks_ts = [pd.Timestamp(w) for w in weeks]
    timeline_start = weeks_ts[0]
    timeline_end = weeks_ts[-1]

    # The last fold's val window ends at timeline_end.
    # Work backwards to find each fold's val_start.
    fold_val_ends = [
        timeline_end - timedelta(weeks=val_weeks * (n_folds - 1 - i))
        for i in range(n_folds)
    ]

    for fold_idx, val_end in enumerate(fold_val_ends):
        val_start = val_end - val_delta + timedelta(weeks=1)
        train_end = val_start - gap

        if train_end <= timeline_start:
            continue  # not enough training data for this fold

        train_mask = df[date_col] <= train_end
        val_mask = (df[date_col] >= val_start) & (df[date_col] <= val_end)

        train_idx = df.index[train_mask]
        val_idx = df.index[val_mask]

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        yield CVSplit(
            fold=fold_idx,
            train_idx=train_idx,
            val_idx=val_idx,
            train_end=pd.Timestamp(train_end),
            val_start=pd.Timestamp(val_start),
            gap_days=int((val_start - train_end).days),
        )
