#!/usr/bin/env python
"""Ingest ECB/OATH Rodent-Related Penalty Schedule into the RAG corpus.

Source: NYC OATH
  Primary URL : https://www.nyc.gov/html/ecb/downloads/pdf/HealthCodeandMiscellaneousFoodVendorViolationsPenaltySchedule.pdf
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

from rat_ml.rag.pdf_parser import parse_penalty_table  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_ingest(
        pdf_url="https://www.nyc.gov/html/ecb/downloads/pdf/HealthCodeandMiscellaneousFoodVendorViolationsPenaltySchedule.pdf",
        pdf_fallback=REPO_ROOT / "data" / "pdfs" / "ecb_penalties.pdf",
        authority="ECB",
        document="ECB Penalty Schedule",
        # This PDF is a borderless fine-schedule table, not numbered legal
        # text — the default legal-hierarchy parser produced 5 giant chunks
        # that interleaved rodent penalty rows with unrelated violations
        # (smoking, menu labeling, permit transfers), starving BM25 of any
        # chunk actually about rodent penalties. parse_penalty_table splits
        # per-row on the table's AH-code violation ids instead.
        chunker=parse_penalty_table,
    ))
