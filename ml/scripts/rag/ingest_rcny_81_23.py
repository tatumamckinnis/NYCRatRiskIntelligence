#!/usr/bin/env python
"""Ingest 24 RCNY §81.23 (IPM for Food Service Establishments) into the RAG corpus.

Source: NYC DOHMH / NYC Rules
  Primary URL : https://www.nyc.gov/assets/doh/downloads/pdf/about/healthcode/health-code-article81.pdf
  Fallback    : data/pdfs/rcny_81_23.pdf

Authority : DOHMH
Document  : 24 RCNY §81.23

Usage::

    uv run --package rat-ml python ml/scripts/rag/ingest_rcny_81_23.py \\
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
        pdf_url="https://www.nyc.gov/assets/doh/downloads/pdf/about/healthcode/health-code-article81.pdf",
        pdf_fallback=REPO_ROOT / "data" / "pdfs" / "rcny_81_23.pdf",
        authority="DOHMH",
        document="24 RCNY §81.23",
    ))
