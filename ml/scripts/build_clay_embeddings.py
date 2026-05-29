"""CLI entry point for Clay v1.5 embedding + PCA pipeline (T-31).

Reads Sentinel-2 composites from data/sentinel2/, runs them through the
frozen Clay v1.5 encoder, fits a 32-dim PCA, and writes clay_pca_0..31
columns into features.nta_week_panel.

Usage (from repo root)::

    uv run --package rat-ml --extra vision --extra temporal \\
        python ml/scripts/build_clay_embeddings.py

Requires environment variables (or .env file):
    DIRECT_DATABASE_URL — direct Supabase connection (bypasses PgBouncer)

Run ingest_sentinel2.py first to populate data/sentinel2/.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

from rat_ml.features.clay_embeddings import run


async def main() -> None:
    db_url = os.environ.get("DIRECT_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DIRECT_DATABASE_URL is not set. Add it to .env or export it.")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    print("Building Clay v1.5 embeddings + PCA …")
    stats = await run(db_url)
    print(
        f"Done.\n"
        f"  NTAs embedded : {stats['n_embedded']}\n"
        f"  NTAs in PCA   : {stats['n_pca']}\n"
        f"  Panel rows updated: {stats['rows_updated']}"
    )


if __name__ == "__main__":
    asyncio.run(main())
