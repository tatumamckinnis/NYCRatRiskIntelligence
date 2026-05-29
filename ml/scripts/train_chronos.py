"""Fine-tune Chronos-2 on the NTA-week panel (T-32).

Usage::

    uv run --package rat-ml --extra temporal python ml/scripts/train_chronos.py \\
        --accelerator auto \\
        --epochs 10

Requires environment variables (or .env file):
    DIRECT_DATABASE_URL — direct Supabase connection (bypasses PgBouncer)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("train_chronos")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Chronos-2 on NTA-week panel")
    p.add_argument("--artifacts-dir", default="ml/artifacts")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument(
        "--accelerator",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
    )
    return p.parse_args()


async def _load_panel(db_url: str):  # type: ignore[return]
    from rat_ml.features.feature_matrix import load_feature_matrix  # noqa: PLC0415
    return await load_feature_matrix(db_url)


def main() -> int:
    args = parse_args()

    db_url = os.environ.get("DIRECT_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DIRECT_DATABASE_URL is not set.")

    log.info("Loading panel …")
    df = asyncio.run(_load_panel(db_url))
    log.info("Panel: %d rows, %d NTAs", len(df), df["nta_id"].nunique())

    from rat_ml.models.chronos_trainer import train_chronos  # noqa: PLC0415

    result = train_chronos(
        df,
        artifacts_dir=args.artifacts_dir,
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        accelerator=args.accelerator,
    )

    print(
        f"\nChronos fine-tune summary\n"
        f"  model_path   : {result.model_path}\n"
        f"  val_loss_mae : {result.val_loss:.4f}\n"
        f"  n_series     : {result.n_series}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
