"""Populate regime-indicator columns in features.nta_week_panel (T-10).

Sets five boolean columns that flag NYC policy regimes:

  - regime_covid
  - regime_8pm_setout
  - regime_commercial_containerization
  - regime_residential_containerization
  - regime_rmz_active (always FALSE in Phase 1)

SQL logic lives in rat_ml.features.regime_indicators; this script is the CLI
entry point.

Usage (from repo root)::

    uv run --package rat-ml python ml/scripts/build_regime_indicators.py
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

from rat_ml.features.regime_indicators import run


async def main() -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DATABASE_URL is not set.")

    print("Populating regime indicator columns …")
    n = await run(db_url)
    print(f"Done. rows updated={n}")


if __name__ == "__main__":
    asyncio.run(main())
