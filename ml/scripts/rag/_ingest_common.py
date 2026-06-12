"""Shared ingest logic for all RAG PDF corpus scripts (T-37).

Each per-document ingest script calls ``run_ingest()`` with its own
PDF URL, fallback path, authority, and document name.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Sequence

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

log = logging.getLogger(__name__)


def _chunk_id_from_hash(version_hash: str) -> str:
    """Derive a deterministic UUID hex from a SHA-256 version hash."""
    # Use UUID5 in the DNS namespace with the hash as the name
    return uuid.uuid5(uuid.NAMESPACE_DNS, version_hash).hex


def _make_corpus_chunks(
    pages: list[str],
    *,
    authority: str,
    document: str,
) -> list:
    """Convert parsed PDF pages to CorpusChunk objects."""
    from rat_ml.rag.corpus import CorpusChunk  # noqa: PLC0415
    from rat_ml.rag.pdf_parser import pages_to_chunks  # noqa: PLC0415

    raw_chunks = pages_to_chunks(pages, authority=authority, document=document)
    result = []
    for c in raw_chunks:
        chunk_id = _chunk_id_from_hash(c.version_hash)
        result.append(
            CorpusChunk(
                chunk_id=chunk_id,
                document=document,
                citation=c.citation,
                authority=authority,
                section_path=[c.citation],
                content=c.content,
                content_with_prefix=c.content_with_prefix,
                token_count=c.token_count,
                version_hash=c.version_hash,
                effective_date=None,
            )
        )
    return result


def _fetch_pages(pdf_url: str, pdf_fallback: Path) -> list[str]:
    from rat_ml.rag.pdf_parser import parse_pdf  # noqa: PLC0415

    if pdf_fallback.exists():
        log.info("Using local PDF: %s", pdf_fallback)
        return parse_pdf(pdf_fallback)

    try:
        import urllib.request  # noqa: PLC0415
        pdf_fallback.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading %s …", pdf_url)
        urllib.request.urlretrieve(pdf_url, pdf_fallback)  # noqa: S310
        log.info("Saved → %s", pdf_fallback)
        return parse_pdf(pdf_fallback)
    except Exception as exc:  # noqa: BLE001
        log.error(
            "Download failed: %s\n"
            "Place the PDF manually at %s and re-run.",
            exc,
            pdf_fallback,
        )
        return []


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--db-url", default="")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


async def _run(
    *,
    pdf_url: str,
    pdf_fallback: Path,
    authority: str,
    document: str,
    db_url: str,
    dry_run: bool,
) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    pages = _fetch_pages(pdf_url, pdf_fallback)
    if not pages:
        log.error("No pages parsed — aborting.")
        return 1

    chunks = _make_corpus_chunks(pages, authority=authority, document=document)
    log.info("Parsed %d chunks from %d pages", len(chunks), len(pages))

    if dry_run:
        for c in chunks[:5]:
            print(f"  {c.citation}: {c.content[:80]!r} ({c.token_count} tok)")
        print(f"  … ({len(chunks)} total)")
        return 0

    from rat_ml.rag.embed import embed_chunks_bge  # noqa: PLC0415
    from rat_ml.rag.store import upsert_chunks  # noqa: PLC0415

    log.info("Embedding %d chunks with BGE-M3 (local, free) …", len(chunks))
    embeddings = embed_chunks_bge(chunks)

    n = await upsert_chunks(chunks, embeddings, db_url=db_url)
    log.info("Upserted %d rows into app.health_code_chunks", n)
    return 0


def run_ingest(
    *,
    pdf_url: str,
    pdf_fallback: Path,
    authority: str,
    document: str,
    argv: list[str] | None = None,
) -> int:
    args = _parse_args(argv)
    db_url = args.db_url or os.environ.get("DIRECT_DATABASE_URL") or os.environ.get("DATABASE_URL", "")

    if not db_url and not args.dry_run:
        sys.exit("DIRECT_DATABASE_URL is not set.")

    return asyncio.run(
        _run(
            pdf_url=pdf_url,
            pdf_fallback=pdf_fallback,
            authority=authority,
            document=document,
            db_url=db_url,
            dry_run=args.dry_run,
        )
    )
