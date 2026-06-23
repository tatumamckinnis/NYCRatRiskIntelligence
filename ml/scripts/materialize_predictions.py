#!/usr/bin/env python3
"""Pre-compute and cache CatBoost predictions for all NTA × week combinations.

Reads features.nta_week_panel, runs the registered CatBoost model for every
NTA × week row, and upserts results into app.risk_predictions so the API can
serve /risk/map instantly without live inference.

Usage (from repo root):
    export $(grep -v '^#' .env | xargs)
    uv run --package rat-api python ml/scripts/materialize_predictions.py

Optional flags:
    --weeks 4       Only materialise the most recent N distinct weeks (default: all)
    --model catboost Model name to use from registry (default: catboost)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

import asyncpg
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "api" / "src"))

from rat_api.ml.loader import load_models
from rat_api.ml.predict import predict_risk


def compute_decile_thresholds(scores: list[float]) -> list[float]:
    if not scores:
        return [i / 10 for i in range(1, 11)]
    arr = np.array(scores)
    return [float(np.percentile(arr, p)) for p in range(10, 101, 10)]


async def main(model_name: str, weeks_limit: int | None) -> None:
    db_url = os.environ.get("DIRECT_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DIRECT_DATABASE_URL is not set.")

    print(f"Loading model '{model_name}' from registry …")
    bundle = load_models("ml/artifacts", model_name)
    model = bundle["model"]
    feature_cols = bundle["metadata"]["feature_cols"]
    model_version = bundle["version"]
    print(f"  Model version: {model_version}")

    conn = await asyncpg.connect(db_url)
    try:
        # Get all distinct weeks in the panel, most recent first
        all_weeks = await conn.fetch(
            "SELECT DISTINCT week_start FROM features.nta_week_panel ORDER BY week_start DESC"
        )
        weeks = [r["week_start"] for r in all_weeks]
        if weeks_limit:
            weeks = weeks[:weeks_limit]
        print(f"  Materialising {len(weeks)} weeks …")

        # Run all predictions first to compute global decile thresholds
        all_scores: list[float] = []
        results: list[tuple] = []

        for i, week in enumerate(weeks):
            rows = await conn.fetch(
                "SELECT * FROM features.nta_week_panel WHERE week_start = $1", week
            )
            for row in rows:
                feature_row = dict(row)
                try:
                    r = predict_risk(
                        model=model,
                        feature_row=feature_row,
                        feature_cols=feature_cols,
                        decile_thresholds=[i / 10 for i in range(1, 11)],  # placeholder
                        model_version=model_version,
                    )
                    all_scores.append(r.risk_score)
                    results.append((feature_row["nta_id"], week, r))
                except Exception as exc:
                    print(f"  WARN: skipping {feature_row.get('nta_id')} {week}: {exc}")

            if (i + 1) % 10 == 0:
                print(f"  scored {i + 1}/{len(weeks)} weeks …")

        print(f"  Total predictions: {len(results)}")

        # Compute proper decile thresholds from global score distribution
        thresholds = compute_decile_thresholds(all_scores)
        print(f"  Decile thresholds: {[round(t, 3) for t in thresholds]}")

        # Recompute deciles with proper thresholds and upsert
        print("  Upserting into app.risk_predictions …")
        upserted = 0
        async with conn.transaction():
            for nta_id, week, r in results:
                # Recompute decile with global thresholds
                decile = next(
                    (i + 1 for i, t in enumerate(thresholds) if r.risk_score <= t),
                    10,
                )
                top_factors_json = json.dumps(
                    [f.model_dump() for f in r.top_factors]
                )
                await conn.execute(
                    """
                    INSERT INTO app.risk_predictions
                        (nta_id, predicted_for_week, risk_score, risk_decile, top_factors, model_version)
                    VALUES ($1, $2, $3, $4, $5::jsonb, $6)
                    ON CONFLICT (nta_id, predicted_for_week, model_version)
                    DO UPDATE SET
                        risk_score  = EXCLUDED.risk_score,
                        risk_decile = EXCLUDED.risk_decile,
                        top_factors = EXCLUDED.top_factors,
                        created_at  = NOW()
                    """,
                    nta_id, week, float(r.risk_score), decile,
                    top_factors_json, model_version,
                )
                upserted += 1

        print(f"  Done — upserted {upserted} rows.")
    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="catboost")
    parser.add_argument("--weeks", type=int, default=None,
                        help="Limit to the most recent N weeks (default: all)")
    args = parser.parse_args()
    asyncio.run(main(args.model, args.weeks))
