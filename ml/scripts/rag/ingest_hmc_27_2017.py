#!/usr/bin/env python
"""Ingest NYC Housing Maintenance Code §§27-2017 – 27-2018.1 into the RAG corpus.

Source: NYC HPD / NYC Admin Code
  Primary URL : https://www.nyc.gov/assets/buildings/pdf/HousingMaintenanceCode.pdf
  Fallback    : data/pdfs/hmc_27_2017.pdf

Authority : HPD
Document  : Housing Maintenance Code

Usage::

    uv run --package rat-ml python ml/scripts/rag/ingest_hmc_27_2017.py \\
        --db-url "$DIRECT_DATABASE_URL" \\
        --voyage-api-key "$VOYAGEAI_API_KEY"
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))

from _ingest_common import run_ingest  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_ingest(
        pdf_url="https://www.nyc.gov/assets/buildings/pdf/HousingMaintenanceCode.pdf",
        pdf_fallback=REPO_ROOT / "data" / "pdfs" / "hmc_27_2017.pdf",
        authority="HPD",
        document="Housing Maintenance Code",
    ))
