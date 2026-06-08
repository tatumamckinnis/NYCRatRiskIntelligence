#!/usr/bin/env python
"""Ingest ECB/OATH Rodent-Related Penalty Schedule into the RAG corpus.

Source: NYC OATH
  Primary URL : https://www.nyc.gov/assets/oath/downloads/pdf/ecb/civil-penalty-schedules/dohmh-ecb-penalty-schedule.pdf
  Fallback    : data/pdfs/ecb_penalties.pdf

Authority : ECB
Document  : ECB Penalty Schedule

Usage::

    uv run --package rat-ml python ml/scripts/rag/ingest_ecb_penalties.py \\
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
        pdf_url="https://www.nyc.gov/assets/oath/downloads/pdf/ecb/civil-penalty-schedules/dohmh-ecb-penalty-schedule.pdf",
        pdf_fallback=REPO_ROOT / "data" / "pdfs" / "ecb_penalties.pdf",
        authority="ECB",
        document="ECB Penalty Schedule",
    ))
