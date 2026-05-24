"""Generate the Phase 1 data-quality report (T-11).

Queries the live database and writes a dated markdown file to
ml/artifacts/data_quality/<YYYY-MM-DD>.md.

Logic lives in rat_ml.reporting.data_quality; this script is the CLI
entry point.

Usage (from repo root)::

    uv run --package rat-ml python ml/scripts/data_quality_report.py
"""

from __future__ import annotations

import asyncio
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

from rat_ml.reporting.data_quality import run


async def main() -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DATABASE_URL is not set.")

    print("Generating data quality report …")
    path = await run(db_url)
    print(f"Report written to {path}")


if __name__ == "__main__":
    asyncio.run(main())
