"""CLI entry point for Sentinel-2 quarterly mosaic ingest (T-30).

Usage (from repo root)::

    uv run --package rat-ml --extra vision python ml/scripts/ingest_sentinel2.py

Requires environment variables (or .env file):
    DIRECT_DATABASE_URL — direct Supabase connection (bypasses PgBouncer)

Optional:
    PLANETARY_COMPUTER_KEY — raises anonymous rate limits
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

from rat_ml.data.ingest_sentinel2 import main

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(main())
