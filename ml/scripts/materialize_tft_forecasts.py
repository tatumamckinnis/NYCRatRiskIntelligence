#!/usr/bin/env python
"""Materialise TFT 12-week forecasts into app.tft_forecasts (T-24).

Run weekly (e.g. Monday morning) after ingest is complete::

    uv run --package rat-ml python ml/scripts/materialize_tft_forecasts.py \\
        --db-url "postgresql://user:pass@host/db" \\
        --artifacts-dir ml/artifacts

The script:
1. Loads the trained TFT model from the registry.
2. Fetches the latest panel data.
3. Builds per-NTA TFT series bundles.
4. For each NTA, generates a 12-week probabilistic forecast (p10/p50/p90).
5. Upserts results into app.tft_forecasts.
6. Prints a summary row count.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta

import asyncpg
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("materialize_tft")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Materialize TFT forecasts")
    p.add_argument("--db-url", required=True, help="asyncpg database URL")
    p.add_argument(
        "--artifacts-dir",
        default="ml/artifacts",
        help="Root artifacts directory",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Monte Carlo samples for quantile estimation",
    )
    return p.parse_args(argv)


def _current_week() -> "pd.Timestamp":
    """Return the Monday of the current ISO week."""
    today = datetime.utcnow().date()
    return pd.Timestamp(today - timedelta(days=today.weekday()))


async def _upsert_forecasts(
    conn: asyncpg.Connection,
    rows: list[dict],
    model_version: str,
) -> int:
    """Upsert forecast rows into app.tft_forecasts. Returns number of rows written."""
    sql = """
        INSERT INTO app.tft_forecasts
            (nta_id, as_of_week, forecast_week, p10, p50, p90, model_version)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (nta_id, as_of_week, forecast_week, model_version)
        DO UPDATE SET
            p10 = EXCLUDED.p10,
            p50 = EXCLUDED.p50,
            p90 = EXCLUDED.p90,
            created_at = NOW()
    """
    count = 0
    async with conn.transaction():
        for row in rows:
            await conn.execute(
                sql,
                row["nta_id"],
                row["as_of_week"],
                row["forecast_week"],
                float(row["p10"]),
                float(row["p50"]),
                float(row["p90"]),
                model_version,
            )
            count += 1
    return count


async def run(args: argparse.Namespace) -> int:
    # ── Load model ────────────────────────────────────────────────────────
    from rat_ml.models.tft_trainer import forecast_nta, load_tft  # noqa: PLC0415
    from rat_ml.models.registry import ModelRegistry  # noqa: PLC0415

    log.info("Loading TFT model from checkpoint in %s …", args.artifacts_dir)
    model = load_tft(args.artifacts_dir)

    registry = ModelRegistry(args.artifacts_dir)
    model_version = registry.list_models().get("tft", "unknown")

    # ── Load panel ────────────────────────────────────────────────────────
    from rat_ml.features.feature_matrix import load_feature_matrix  # noqa: PLC0415
    from rat_ml.features.tft_dataset import build_tft_dataset  # noqa: PLC0415
    from rat_ml.models.tft_trainer import INPUT_CHUNK_LENGTH  # noqa: PLC0415

    log.info("Loading panel …")
    df = await load_feature_matrix(args.db_url)

    log.info("Panel: %d rows", len(df))
    dataset = build_tft_dataset(df)
    log.info("TFT bundles: %d NTAs", len(dataset.bundles))

    as_of_week = _current_week().date()

    # ── Generate forecasts ────────────────────────────────────────────────
    from darts import TimeSeries as _TS  # noqa: PLC0415
    HORIZON = 12  # forecast weeks

    def _extend_future_cov(fc: "_TS", n: int) -> "_TS":
        """Forward-fill the last row of future covariates by n weeks.

        Regime indicators are policy-level variables that are known in advance;
        we assume they remain at their last observed value for the forecast horizon.
        """
        df = fc.to_dataframe()
        last_val = df.iloc[-1]
        freq = df.index.freq or pd.tseries.frequencies.to_offset("7D")
        new_idx = pd.date_range(df.index[-1] + freq, periods=n, freq=freq)
        new_rows = pd.DataFrame(
            [last_val.values] * n, index=new_idx, columns=df.columns
        )
        return _TS.from_dataframe(pd.concat([df, new_rows]))

    all_rows: list[dict] = []
    failed = 0

    for bundle in dataset.bundles:
        try:
            future_cov_ext = _extend_future_cov(bundle.future_covariates, HORIZON)
            forecast_df = forecast_nta(
                model,
                bundle.target,
                bundle.past_covariates,
                future_cov_ext,
                num_samples=args.num_samples,
            )
            for _, frow in forecast_df.iterrows():
                all_rows.append(
                    {
                        "nta_id": bundle.nta_id,
                        "as_of_week": as_of_week,
                        "forecast_week": frow["week"],
                        "p10": frow["p10"],
                        "p50": frow["p50"],
                        "p90": frow["p90"],
                    }
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to forecast NTA %s: %s", bundle.nta_id, exc)
            failed += 1

    log.info(
        "Generated %d forecast rows (%d NTAs, %d failed)",
        len(all_rows),
        len(dataset.bundles) - failed,
        failed,
    )

    if not all_rows:
        log.error("No forecasts generated — aborting upsert.")
        return 1

    # ── Upsert ────────────────────────────────────────────────────────────
    conn = await asyncpg.connect(args.db_url)
    try:
        n_written = await _upsert_forecasts(conn, all_rows, model_version)
    finally:
        await conn.close()

    log.info("Upserted %d rows into app.tft_forecasts.", n_written)
    print(
        f"\nMaterialization summary\n"
        f"  as_of_week    : {as_of_week}\n"
        f"  model_version : {model_version}\n"
        f"  rows written  : {n_written}\n"
        f"  NTAs failed   : {failed}\n"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
