"""Add raw.nta_boundaries reference table.

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-03
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # raw.nta_boundaries
    # NTA 2020 boundaries from NYC DCP.
    # Populated by ml/scripts/load_nta_boundaries.py (T-03).
    # Used by:
    #   - T-06 ingest_311.py  — spatial join complaints → NTA
    #   - T-09 spatial_lags.py — queen-contiguity adjacency matrix
    # ------------------------------------------------------------------
    op.execute("""
        CREATE TABLE raw.nta_boundaries (
            nta_id      TEXT PRIMARY KEY,               -- NTA 2020 code, e.g. 'BK2201'
            nta_name    TEXT NOT NULL,
            borough     TEXT NOT NULL,
            geom        GEOMETRY(MultiPolygon, 4326),   -- WGS84; NY State Plane re-projected at ingest
            area_sq_m   NUMERIC
        )
    """)
    op.execute('CREATE INDEX ON raw.nta_boundaries USING GIST (geom)')


def downgrade() -> None:
    op.execute('DROP TABLE IF EXISTS raw.nta_boundaries CASCADE')
