#!/usr/bin/env python
"""Train the TFT 12-week forecast model (T-24).

Usage::

    uv run --package rat-ml python ml/scripts/train_tft.py \\
        --db-url "postgresql://user:pass@host/db" \\
        --artifacts-dir ml/artifacts \\
        --epochs 50 \\
        --accelerator auto

The script:
1. Loads the NTA-week panel from the database.
2. Builds per-NTA Darts TimeSeries bundles.
3. Splits into train / validation sets.
4. Trains a TFTModel with early stopping.
5. Saves the model to the registry under the name "tft".
6. Prints a summary with val_loss and series count.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("train_tft")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TFT forecast model")
    p.add_argument("--db-url", required=True, help="asyncpg database URL")
    p.add_argument(
        "--artifacts-dir",
        default="ml/artifacts",
        help="Root artifacts directory (default: ml/artifacts)",
    )
    p.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    p.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    p.add_argument("--hidden-size", type=int, default=64, help="TFT hidden dimension")
    p.add_argument(
        "--input-chunk",
        type=int,
        default=52,
        help="Encoder input chunk length (weeks)",
    )
    p.add_argument(
        "--output-chunk",
        type=int,
        default=12,
        help="Forecast horizon (weeks)",
    )
    p.add_argument(
        "--holdout-weeks",
        type=int,
        default=12,
        help="Weeks held out for validation split",
    )
    p.add_argument(
        "--accelerator",
        default="auto",
        choices=["auto", "cpu", "gpu", "mps"],
        help="PyTorch Lightning accelerator",
    )
    return p.parse_args(argv)


async def _load_panel(db_url: str):  # type: ignore[return]
    from rat_ml.features.feature_matrix import load_feature_matrix  # noqa: PLC0415

    log.info("Loading panel from database …")
    return await load_feature_matrix(db_url)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ── Load panel ────────────────────────────────────────────────────────
    df = asyncio.run(_load_panel(args.db_url))
    log.info("Panel loaded: %d rows, %d NTAs", len(df), df["nta_id"].nunique())

    # ── Build TFT dataset ─────────────────────────────────────────────────
    from rat_ml.features.tft_dataset import build_tft_dataset, split_tft_dataset  # noqa: PLC0415

    full_dataset = build_tft_dataset(df)
    log.info("TFT dataset: %d qualifying series", len(full_dataset.bundles))

    if len(full_dataset.bundles) == 0:
        log.error("No qualifying series — panel may be empty. Aborting.")
        return 1

    train_dataset, val_dataset = split_tft_dataset(
        full_dataset, holdout_weeks=args.holdout_weeks
    )
    log.info(
        "Train series: %d  |  Val series: %d",
        len(train_dataset.bundles),
        len(val_dataset.bundles),
    )

    # ── Train ─────────────────────────────────────────────────────────────
    from rat_ml.models.tft_trainer import train_tft  # noqa: PLC0415

    result = train_tft(
        train_dataset,
        val_dataset,
        artifacts_dir=args.artifacts_dir,
        input_chunk_length=args.input_chunk,
        output_chunk_length=args.output_chunk,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        accelerator=args.accelerator,
    )

    log.info("Training complete.")
    log.info("  Model saved: %s", result.model_path)
    log.info("  Val loss:    %.6f", result.val_loss)
    log.info("  Series:      %d", result.n_series)

    print(
        f"\nTFT training summary\n"
        f"  model_path : {result.model_path}\n"
        f"  val_loss   : {result.val_loss:.6f}\n"
        f"  n_series   : {result.n_series}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
