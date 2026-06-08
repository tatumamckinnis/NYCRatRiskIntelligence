#!/usr/bin/env python
"""Ingest NYC Health Code Title 24 Article 151 (Rodent Control) into the RAG corpus.

Source: NYC DOHMH
  Primary URL : https://www.nyc.gov/assets/doh/downloads/pdf/rodent/rodent-health-code-article151.pdf
  Fallback    : data/pdfs/health_code_article_151.pdf

Authority : DOHMH
Document  : Health Code Title 24 Article 151

Usage::

    uv run --package rat-ml python ml/scripts/rag/ingest_health_code_151.py \\
        --db-url "$DIRECT_DATABASE_URL" \\
        --voyage-api-key "$VOYAGEAI_API_KEY"
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "ml" / "scripts" / "rag"))

from _ingest_common import run_ingest  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_ingest(
        pdf_url="https://www.nyc.gov/assets/doh/downloads/pdf/rodent/rodent-health-code-article151.pdf",
        pdf_fallback=REPO_ROOT / "data" / "pdfs" / "health_code_article_151.pdf",
        authority="DOHMH",
        document="Health Code Title 24 Article 151",
    ))
