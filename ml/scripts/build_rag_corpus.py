#!/usr/bin/env python
"""Build and embed the RAG corpus, then upsert into app.health_code_chunks (T-25).

Usage::

    uv run --package rat-ml python ml/scripts/build_rag_corpus.py \\
        --db-url "postgresql://user:pass@host/db" \\
        --voyage-api-key "$VOYAGEAI_API_KEY"

Set ``--dry-run`` to print chunks without embedding or writing to the DB.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_rag_corpus")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build and upsert RAG corpus")
    p.add_argument("--db-url", required=True, help="asyncpg database URL")
    p.add_argument("--voyage-api-key", required=True, help="Voyage AI API key")
    p.add_argument(
        "--voyage-model",
        default="voyage-3",
        help="Voyage AI embedding model (default: voyage-3)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print chunks without embedding or writing to DB",
    )
    return p.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    from rat_ml.rag.corpus import build_corpus  # noqa: PLC0415
    from rat_ml.rag.embed import embed_chunks  # noqa: PLC0415
    from rat_ml.rag.store import upsert_chunks  # noqa: PLC0415

    # ── Build corpus ──────────────────────────────────────────────────────
    log.info("Building corpus …")
    chunks = build_corpus()
    log.info("Corpus: %d chunks", len(chunks))

    if args.dry_run:
        for c in chunks:
            print(f"[{c.document}] {c.citation} — {c.token_count} tokens")
            print(f"  {c.content[:120]}…\n")
        return 0

    # ── Embed ─────────────────────────────────────────────────────────────
    log.info("Embedding %d chunks with %s …", len(chunks), args.voyage_model)
    embeddings = embed_chunks(
        chunks,
        api_key=args.voyage_api_key,
        model=args.voyage_model,
    )

    # ── Upsert ────────────────────────────────────────────────────────────
    log.info("Upserting into app.health_code_chunks …")
    n = await upsert_chunks(chunks, embeddings, db_url=args.db_url)

    print(
        f"\nRAG corpus build summary\n"
        f"  chunks built   : {len(chunks)}\n"
        f"  rows upserted  : {n}\n"
        f"  embedding model: {args.voyage_model}\n"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
