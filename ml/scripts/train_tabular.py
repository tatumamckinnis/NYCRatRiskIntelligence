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
    SKIP_TABPFN          set to "1" to skip TabPFN (per-borough, faster iteration)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

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
from rat_ml.models.tabular import (
    CatBoostTrainer,
    LightGBMTrainer,
    LRTrainer,
    TabPFNTrainer,
    TrainResult,
    _encode_categoricals,
)


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
    print("\n[1/4] Training CatBoost …")
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
    # Save OOF predictions for fusion meta-learner (final model on full panel)
    all_panel = pd.concat([train_df, test_df], ignore_index=True)
    y_prob_all = cb_result.model.predict_proba(all_panel[feature_cols])[:, 1]
    oof_dict = {
        f"{row['nta_id']}|{str(row['week_start'])[:10]}": float(prob)
        for (_, row), prob in zip(all_panel.iterrows(), y_prob_all)
    }
    (path / "oof_predictions.json").write_text(json.dumps(oof_dict))
    print(f"  Test PR-AUC: {cb_result.test_metrics['pr_auc']:.4f}  "
          f"Top-decile lift: {cb_result.test_metrics['top_decile_lift']:.2f}x  "
          f"→ {path}")
    results.append(cb_result)

    # ------------------------------------------------------------------
    # LightGBM (ablation)
    # ------------------------------------------------------------------
    if os.environ.get("SKIP_LGB") != "1":
        print("\n[2/4] Training LightGBM …")
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
        print("\n[2/4] LightGBM skipped (SKIP_LGB=1)")

    # ------------------------------------------------------------------
    # Logistic Regression (baseline)
    # ------------------------------------------------------------------
    if os.environ.get("SKIP_LR") != "1":
        print("\n[3/4] Training Logistic Regression …")
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
        print("\n[3/4] Logistic Regression skipped (SKIP_LR=1)")

    # ------------------------------------------------------------------
    # TabPFN v2 (per-borough specialist; feeds the CatBoost + TabPFN row)
    # ------------------------------------------------------------------
    if os.environ.get("SKIP_TABPFN") != "1":
        print("\n[4/4] Training TabPFN (per-borough) …")
        cat_cols = [c for c in ["borough"] if c in feature_cols]
        test_labels: list[np.ndarray] = []
        test_probs: list[np.ndarray] = []
        cv_means: list[float] = []

        for borough in sorted(train_df["borough"].unique()):
            bo_train = train_df[train_df["borough"] == borough].reset_index(drop=True)
            bo_test = test_df[test_df["borough"] == borough].reset_index(drop=True)
            if bo_train.empty or bo_test.empty:
                continue
            if len(bo_train) > TabPFNTrainer.MAX_ROWS:
                bo_train = bo_train.sample(
                    TabPFNTrainer.MAX_ROWS, random_state=42
                ).reset_index(drop=True)

            bo_result = TabPFNTrainer().fit(bo_train, bo_test, feature_cols)
            path = registry.save(
                f"tabpfn_{borough.lower()}",
                bo_result.model,
                metadata={
                    "borough": borough,
                    "feature_cols": feature_cols,
                    "cv_pr_auc_mean": bo_result.cv_pr_auc_mean,
                    "cv_pr_auc_std": bo_result.cv_pr_auc_std,
                    "test_metrics": bo_result.test_metrics,
                },
            )
            (path / "report.md").write_text(_report_md(bo_result))
            print(f"  [{borough}] Test PR-AUC: {bo_result.test_metrics['pr_auc']:.4f}  → {path}")

            X_bo_test = bo_test[feature_cols].copy()
            X_bo_test_enc, _, _ = _encode_categoricals(
                X_bo_test, X_bo_test.iloc[:1].copy(), cat_cols
            )
            y_bo_prob = bo_result.model.predict_proba(X_bo_test_enc)[:, 1]

            test_labels.append(bo_test[LABEL_COL].astype(int).values)
            test_probs.append(y_bo_prob)
            cv_means.append(bo_result.cv_pr_auc_mean)

        if test_labels:
            agg_metrics = metric_bundle(
                np.concatenate(test_labels), np.concatenate(test_probs)
            )
            tabpfn_result = TrainResult(
                model_name="tabpfn",
                model=None,
                label_encoder=None,
                fold_metrics=[],
                cv_pr_auc_mean=float(np.mean(cv_means)),
                cv_pr_auc_std=float(np.std(cv_means)),
                test_metrics=agg_metrics,
                top_shap_features={},
                feature_cols=feature_cols,
            )
            print(f"  [aggregate] Test PR-AUC: {agg_metrics['pr_auc']:.4f}  "
                  f"Top-decile lift: {agg_metrics['top_decile_lift']:.2f}x")
            results.append(tabpfn_result)
    else:
        print("\n[4/4] TabPFN skipped (SKIP_TABPFN=1)")

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
