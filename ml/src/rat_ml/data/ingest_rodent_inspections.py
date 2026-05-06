"""Ingest DOHMH rodent inspection records into raw.rodent_inspections.

Source:  NYC Open Data — Rodent Inspection (Socrata p937-wjvj; fallback jh4g-rp64)
Cap:     Last 3 years (applied at query time via $where)
Key:     inspection_id (upsert)
OTel:    One span per batch with batch.size and batch.offset attributes.

Usage (from repo root):
    uv run --package rat-ml python ml/src/rat_ml/data/ingest_rodent_inspections.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import date, timedelta

import asyncpg
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

from rat_ml.data._socrata import get_client, paginate
from rat_ml.data.bbl_join import emit_unmatched_report, normalize_bbl

PRIMARY_DATASET = "p937-wjvj"
FALLBACK_DATASET = "jh4g-rp64"
SOURCE_NAME = "rodent_inspections"


def _setup_otel() -> trace.Tracer:
    endpoint = os.environ.get("PHOENIX_OTLP_ENDPOINT")
    if endpoint:
        provider = TracerProvider(resource=Resource({"service.name": "rat-ml-ingest"}))
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
        )
        trace.set_tracer_provider(provider)
    return trace.get_tracer(__name__)


def _cutoff_date() -> str:
    return (date.today() - timedelta(days=3 * 365)).isoformat()


def _parse_row(row: dict) -> dict | None:
    """Normalise a Socrata row to the raw.rodent_inspections schema.

    Returns None for rows missing the primary key.
    """
    inspection_id = (
        row.get("inspectionid")
        or row.get("inspection_id")
        or row.get("unique_key")
    )
    if not inspection_id:
        return None

    raw_bbl = row.get("bbl")
    bbl = normalize_bbl(raw_bbl)

    lat = row.get("latitude") or row.get("y_coordinate")
    lon = row.get("longitude") or row.get("x_coordinate")
    geom_wkt = f"POINT({lon} {lat})" if lat and lon else None

    insp_date = row.get("inspectiondate") or row.get("inspection_date")

    try:
        borough = int(row["boro"]) if row.get("boro") else None
    except (ValueError, TypeError):
        borough = None

    def _int(val: object) -> int | None:
        try:
            return int(val) if val is not None else None
        except (ValueError, TypeError):
            return None

    return {
        "inspection_id": str(inspection_id),
        "inspection_date": insp_date,
        "bbl": bbl,
        "raw_bbl_missing": bool(raw_bbl and bbl is None),
        "bin": row.get("bin"),
        "borough": borough,
        "block": _int(row.get("block")),
        "lot": _int(row.get("lot")),
        "result": row.get("result") or "",
        "inspection_type": row.get("inspection_type") or row.get("inspectiontype"),
        "job_progress": _int(row.get("jobprogress") or row.get("job_progress")),
        "geom_wkt": geom_wkt,
    }


async def upsert_batch(conn: asyncpg.Connection, rows: list[dict]) -> tuple[int, int]:
    """Upsert a batch of parsed rows. Returns (total, unmatched_bbl)."""
    parsed = [_parse_row(r) for r in rows]
    valid = [p for p in parsed if p is not None]
    unmatched = sum(1 for p in valid if p["raw_bbl_missing"])

    async with conn.transaction():
        for p in valid:
            await conn.execute(
                """
                INSERT INTO raw.rodent_inspections (
                    inspection_id, inspection_date, bbl, bin, borough,
                    block, lot, result, inspection_type, job_progress, geom
                ) VALUES (
                    $1, $2::date, $3, $4, $5, $6, $7, $8, $9, $10,
                    CASE WHEN $11 IS NOT NULL
                         THEN ST_SetSRID(ST_GeomFromText($11), 4326)
                    END
                )
                ON CONFLICT (inspection_id) DO UPDATE SET
                    inspection_date  = EXCLUDED.inspection_date,
                    bbl              = EXCLUDED.bbl,
                    result           = EXCLUDED.result,
                    inspection_type  = EXCLUDED.inspection_type,
                    job_progress     = EXCLUDED.job_progress,
                    geom             = EXCLUDED.geom
                """,
                p["inspection_id"],
                p["inspection_date"],
                p["bbl"],
                p["bin"],
                p["borough"],
                p["block"],
                p["lot"],
                p["result"],
                p["inspection_type"],
                p["job_progress"],
                p["geom_wkt"],
            )

    return len(valid), unmatched


async def run(db_url: str) -> None:
    tracer = _setup_otel()
    client = get_client()
    conn = await asyncpg.connect(db_url)

    # Try primary dataset; fall back if Socrata returns an error.
    try:
        dataset_id = PRIMARY_DATASET
        total_rows = total_unmatched = 0
        cutoff = _cutoff_date()

        for batch, offset in paginate(
            client,
            dataset_id,
            where=f"inspectiondate >= '{cutoff}'",
            order=":id",
        ):
            with tracer.start_as_current_span("ingest.rodent_inspections.batch") as span:
                span.set_attribute("batch.size", len(batch))
                span.set_attribute("batch.offset", offset)
                span.set_attribute("dataset_id", dataset_id)

                n, u = await upsert_batch(conn, batch)
                total_rows += n
                total_unmatched += u

                print(f"  offset={offset:>7}  upserted={n}  unmatched_bbl={u}")

    except Exception:  # noqa: BLE001
        print(f"Primary dataset {PRIMARY_DATASET} failed, trying fallback {FALLBACK_DATASET}",
              file=sys.stderr)
        dataset_id = FALLBACK_DATASET
        total_rows = total_unmatched = 0
        cutoff = _cutoff_date()

        for batch, offset in paginate(
            client,
            dataset_id,
            where=f"inspectiondate >= '{cutoff}'",
            order=":id",
        ):
            with tracer.start_as_current_span("ingest.rodent_inspections.batch") as span:
                span.set_attribute("batch.size", len(batch))
                span.set_attribute("batch.offset", offset)
                span.set_attribute("dataset_id", dataset_id)

                n, u = await upsert_batch(conn, batch)
                total_rows += n
                total_unmatched += u

    emit_unmatched_report(SOURCE_NAME, total=total_rows, unmatched=total_unmatched)
    print(f"Done. total={total_rows}  unmatched_bbl={total_unmatched}")
    await conn.close()


async def main() -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DATABASE_URL is not set.")
    await run(db_url)


if __name__ == "__main__":
    asyncio.run(main())
