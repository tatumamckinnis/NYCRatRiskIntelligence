"""Train tabular risk models and write versioned artifacts (T-16).

Trains CatBoost (primary), LightGBM (ablation), and Logistic Regression
(baseline) using expanding-window time-series CV on features.nta_week_panel.
Writes calibrated model artifacts, SHAP importances, and a report.md to
ml/artifacts/tabular/<model_name>/<timestamp>/.

Usage (from repo root)::

    uv run --package rat-ml --extra ml python ml/scripts/train_tabular.py

Optional env vars:
    MODEL_ARTIFACTS_DIR  override artifact output directory (default: ml/artifacts)
    SKIP_LGB             set to "1" to skip LightGBM (faster iteration)
    SKIP_LR              set to "1" to skip Logistic Regression
"""

from __future__ import annotations

import asyncio
import os
import sys
import textwrap
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

from rat_ml.eval.metrics import metric_bundle
from rat_ml.features.feature_matrix import (
    LABEL_COL,
    effective_feature_cols,
    load_feature_matrix,
    train_test_split,
)
from rat_ml.models.registry import ModelRegistry
from rat_ml.models.tabular import CatBoostTrainer, LightGBMTrainer, LRTrainer, TrainResult


ARTIFACTS_DIR = Path(os.environ.get("MODEL_ARTIFACTS_DIR", "ml/artifacts"))


def _report_md(result: TrainResult) -> str:
    m = result.test_metrics
    lines = [
        f"# {result.model_name} — Training Report",
        "",
        "## CV Results",
        "",
        f"| Fold | PR-AUC | ROC-AUC | Brier | Top-Decile Lift |",
        f"|---|---:|---:|---:|---:|",
    ]
    for i, fold in enumerate(result.fold_metrics):
        lines.append(
            f"| {i+1} | {fold['pr_auc']:.4f} | {fold['roc_auc']:.4f}"
            f" | {fold['brier']:.4f} | {fold['top_decile_lift']:.2f} |"
        )
    lines += [
        "",
        f"**CV PR-AUC**: {result.cv_pr_auc_mean:.4f} ± {result.cv_pr_auc_std:.4f}",
        "",
        "## Test Set Results",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| PR-AUC | {m['pr_auc']:.4f} |",
        f"| ROC-AUC | {m['roc_auc']:.4f} |",
        f"| Brier Score | {m['brier']:.4f} |",
        f"| Top-Decile Lift | {m['top_decile_lift']:.2f} |",
        "",
    ]
    if result.top_shap_features:
        lines += [
            "## Top SHAP Features",
            "",
            "| Feature | Mean |SHAP| |",
            "|---|---:|",
        ]
        for feat, val in list(result.top_shap_features.items())[:20]:
            lines.append(f"| `{feat}` | {val:.6f} |")
        lines.append("")
    return "\n".join(lines)


def _ablation_row(result: TrainResult) -> str:
    m = result.test_metrics
    return (
        f"| {result.model_name} "
        f"| {result.cv_pr_auc_mean:.4f} ± {result.cv_pr_auc_std:.4f} "
        f"| {m['pr_auc']:.4f} "
        f"| {m['brier']:.4f} "
        f"| {m['top_decile_lift']:.2f} |"
    )


async def main() -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DATABASE_URL is not set.")

    print("Loading feature matrix from DB …")
    df = await load_feature_matrix(db_url)
    print(f"  {len(df):,} rows, {df['nta_id'].nunique()} NTAs, "
          f"{df['week_start'].min().date()} – {df['week_start'].max().date()}")

    train_df, test_df = train_test_split(df)
    print(f"  Train: {len(train_df):,} rows | Test (holdout): {len(test_df):,} rows")

    feature_cols = effective_feature_cols(df)
    # Drop rows where label is null
    train_df = train_df.dropna(subset=[LABEL_COL]).reset_index(drop=True)
    test_df = test_df.dropna(subset=[LABEL_COL]).reset_index(drop=True)

    print(f"  Features: {len(feature_cols)} columns")
    print(f"  Label prevalence (train): "
          f"{train_df[LABEL_COL].mean():.1%}")

    registry = ModelRegistry(ARTIFACTS_DIR)
    results: list[TrainResult] = []

    # ------------------------------------------------------------------
    # CatBoost (primary)
    # ------------------------------------------------------------------
    print("\n[1/3] Training CatBoost …")
    cb_result = CatBoostTrainer().fit(train_df, test_df, feature_cols)
    path = registry.save(
        "catboost",
        cb_result.model,
        metadata={
            "feature_cols": feature_cols,
            "cv_pr_auc_mean": cb_result.cv_pr_auc_mean,
            "cv_pr_auc_std": cb_result.cv_pr_auc_std,
            "test_metrics": cb_result.test_metrics,
            "top_shap_features": cb_result.top_shap_features,
        },
    )
    (path / "report.md").write_text(_report_md(cb_result))
    print(f"  Test PR-AUC: {cb_result.test_metrics['pr_auc']:.4f}  "
          f"Top-decile lift: {cb_result.test_metrics['top_decile_lift']:.2f}x  "
          f"→ {path}")
    results.append(cb_result)

    # ------------------------------------------------------------------
    # LightGBM (ablation)
    # ------------------------------------------------------------------
    if os.environ.get("SKIP_LGB") != "1":
        print("\n[2/3] Training LightGBM …")
        lgb_result = LightGBMTrainer().fit(train_df, test_df, feature_cols)
        path = registry.save(
            "lightgbm",
            lgb_result.model,
            metadata={
                "feature_cols": feature_cols,
                "cv_pr_auc_mean": lgb_result.cv_pr_auc_mean,
                "test_metrics": lgb_result.test_metrics,
            },
        )
        (path / "report.md").write_text(_report_md(lgb_result))
        print(f"  Test PR-AUC: {lgb_result.test_metrics['pr_auc']:.4f}  → {path}")
        results.append(lgb_result)
    else:
        print("\n[2/3] LightGBM skipped (SKIP_LGB=1)")

    # ------------------------------------------------------------------
    # Logistic Regression (baseline)
    # ------------------------------------------------------------------
    if os.environ.get("SKIP_LR") != "1":
        print("\n[3/3] Training Logistic Regression …")
        lr_result = LRTrainer().fit(train_df, test_df, feature_cols)
        path = registry.save(
            "logistic_regression",
            lr_result.model,
            metadata={
                "feature_cols": feature_cols,
                "cv_pr_auc_mean": lr_result.cv_pr_auc_mean,
                "test_metrics": lr_result.test_metrics,
            },
        )
        (path / "report.md").write_text(_report_md(lr_result))
        print(f"  Test PR-AUC: {lr_result.test_metrics['pr_auc']:.4f}  → {path}")
        results.append(lr_result)
    else:
        print("\n[3/3] Logistic Regression skipped (SKIP_LR=1)")

    # ------------------------------------------------------------------
    # Ablation table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    header = (
        "| Model | CV PR-AUC (mean ± std) | Test PR-AUC | Brier | Top-Decile Lift |"
    )
    sep = "|---|---|---:|---:|---:|"
    print(header)
    print(sep)
    for r in results:
        print(_ablation_row(r))
    print("=" * 70)

    # Write ablation table to artifacts root
    ablation_md = "\n".join(
        ["# Phase 2 Ablation Table", "", header, sep]
        + [_ablation_row(r) for r in results]
        + [""]
    )
    ablation_path = ARTIFACTS_DIR / "tabular" / "ablation.md"
    ablation_path.parent.mkdir(parents=True, exist_ok=True)
    ablation_path.write_text(ablation_md)
    print(f"\nAblation table written to {ablation_path}")


if __name__ == "__main__":
    asyncio.run(main())
