#!/usr/bin/env python
"""Ingest DOHMH Rodent Academy training materials into the RAG corpus.

Source: NYC DOHMH Rodent Academy
  Primary URL : https://www.nyc.gov/assets/doh/downloads/pdf/rodent/rodent-academy-manual.pdf
  Fallback    : data/pdfs/rodent_academy.pdf

Authority : DOHMH
Document  : Rodent Academy

Usage::

    uv run --package rat-ml python ml/scripts/rag/ingest_rodent_academy.py \\
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
        pdf_url="https://www.nyc.gov/assets/doh/downloads/pdf/rodent/rodent-academy-manual.pdf",
        pdf_fallback=REPO_ROOT / "data" / "pdfs" / "rodent_academy.pdf",
        authority="DOHMH",
        document="Rodent Academy",
    ))
