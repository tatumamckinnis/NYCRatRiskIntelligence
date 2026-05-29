"""Train the fusion stacked meta-learner (T-34).

Assembles OOF predictions from CatBoost, TFT, and Chronos-2 plus Clay PCA
features, trains a calibrated logistic regression meta-learner (primary) and
a shallow MLP (ablation row), and writes both to the model registry.

Prints an extended ablation table comparing:
  CatBoost alone / +TFT / +Clay / +Chronos-2 / full ensemble (LR) / full ensemble (MLP)

Usage::

    uv run --package rat-ml python ml/scripts/train_fusion.py

Requires environment variables (or .env file):
    DIRECT_DATABASE_URL — direct Supabase connection (bypasses PgBouncer)

Prerequisites:
    1. train_tabular.py already run  (CatBoost OOF in artifacts)
    2. train_tft.py already run      (TFT OOF in artifacts)
    3. train_chronos.py already run  (Chronos OOF in artifacts)
    4. build_clay_embeddings.py run  (clay_pca_* columns in panel)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("train_fusion")


async def _load_panel(db_url: str):  # type: ignore[return]
    from rat_ml.features.feature_matrix import load_feature_matrix  # noqa: PLC0415
    return await load_feature_matrix(db_url)


def main() -> int:
    db_url = os.environ.get("DIRECT_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DIRECT_DATABASE_URL is not set.")

    artifacts_dir = "ml/artifacts"

    log.info("Loading panel from database …")
    panel_df = asyncio.run(_load_panel(db_url))
    log.info("Panel: %d rows, %d NTAs", len(panel_df), panel_df["nta_id"].nunique())

    from rat_ml.models.fusion import build_meta_features, train_fusion  # noqa: PLC0415
    from rat_ml.models.registry import ModelRegistry  # noqa: PLC0415

    log.info("Assembling meta-feature matrix …")
    meta_df = build_meta_features(panel_df, artifacts_dir=artifacts_dir)
    log.info("Meta-feature matrix: %d rows", len(meta_df))

    if len(meta_df) == 0:
        sys.exit("Meta-feature matrix is empty. Check OOF prediction files exist.")

    registry = ModelRegistry(artifacts_dir)

    ablation_rows = []

    # ── Primary: Logistic Regression fusion ──────────────────────────────
    log.info("Training LR fusion model …")
    lr_model, lr_metrics = train_fusion(meta_df, model_type="logistic_regression")
    lr_path = registry.save("fusion_lr", lr_model, metadata={
        "model_type": "logistic_regression",
        "features": lr_model.feature_cols,
        **lr_metrics,
    })
    ablation_rows.append({
        "model": "Full Ensemble (LR meta)",
        "pr_auc": lr_metrics["pr_auc"],
        "brier": lr_metrics["brier"],
        "top_decile_lift": lr_metrics["top_decile_lift"],
    })
    log.info("LR fusion saved → %s", lr_path)

    # ── Ablation: MLP fusion ──────────────────────────────────────────────
    log.info("Training MLP fusion model (ablation) …")
    mlp_model, mlp_metrics = train_fusion(meta_df, model_type="mlp")
    mlp_path = registry.save("fusion_mlp", mlp_model, metadata={
        "model_type": "mlp",
        "features": mlp_model.feature_cols,
        **mlp_metrics,
    })
    ablation_rows.append({
        "model": "Full Ensemble (MLP meta)",
        "pr_auc": mlp_metrics["pr_auc"],
        "brier": mlp_metrics["brier"],
        "top_decile_lift": mlp_metrics["top_decile_lift"],
    })
    log.info("MLP fusion saved → %s", mlp_path)

    # ── Load existing tabular ablation for comparison ─────────────────────
    ablation_path = Path(artifacts_dir) / "tabular" / "ablation.md"
    existing_rows = _parse_ablation_md(ablation_path) if ablation_path.exists() else []

    all_rows = existing_rows + ablation_rows

    # ── Print combined ablation table ─────────────────────────────────────
    _print_ablation(all_rows)

    # ── Append fusion rows to ablation.md ────────────────────────────────
    _append_to_ablation(ablation_path, ablation_rows)

    return 0


def _parse_ablation_md(path: Path) -> list[dict]:
    """Parse existing ablation.md table rows into dicts."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("|") and not line.startswith("| Model") and not line.startswith("|---"):
                parts = [p.strip() for p in line.strip("|").split("|")]
                if len(parts) >= 4:
                    try:
                        rows.append({
                            "model": parts[0],
                            "pr_auc": float(parts[2]),
                            "brier": float(parts[3]),
                            "top_decile_lift": float(parts[4]) if len(parts) > 4 else 0.0,
                        })
                    except (ValueError, IndexError):
                        pass
    return rows


def _print_ablation(rows: list[dict]) -> None:
    header = f"\n{'='*80}\n| {'Model':<40} | {'PR-AUC':>8} | {'Brier':>7} | {'Top-D Lift':>10} |\n|{'---'*26}|"
    print(header)
    for r in rows:
        print(f"| {r['model']:<40} | {r['pr_auc']:>8.4f} | {r['brier']:>7.4f} | {r['top_decile_lift']:>10.2f} |")
    print("=" * 80)


def _append_to_ablation(path: Path, rows: list[dict]) -> None:
    """Append fusion rows to the existing ablation.md."""
    if not path.exists():
        return
    lines = [
        f"| {r['model']} | — | {r['pr_auc']:.4f} | {r['brier']:.4f} | {r['top_decile_lift']:.2f} |"
        for r in rows
    ]
    with open(path, "a") as f:
        f.write("\n## Fusion meta-learner rows\n\n")
        f.write("| Model | CV PR-AUC | Test PR-AUC | Brier | Top-Decile Lift |\n")
        f.write("|---|---|---|---|---|\n")
        f.write("\n".join(lines) + "\n")
    log.info("Fusion rows appended to %s", path)


if __name__ == "__main__":
    sys.exit(main())
