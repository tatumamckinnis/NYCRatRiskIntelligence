"""Add raw.pluto reference table for PLUTO lot data.

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-06
"""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # raw.pluto
    # Source: NYC MapPLUTO (DCP), downloaded as CSV zip (~860k rows).
    # Loaded once per quarter by ml/src/rat_ml/data/ingest_pluto.py.
    # Key fields used by T-08 panel assembly for static NTA aggregates.
    # appbbl enables condo billing-BBL resolution (see bbl_join.py).
    # ------------------------------------------------------------------
    op.execute("""
        CREATE TABLE raw.pluto (
            bbl         TEXT PRIMARY KEY,      -- 10-char normalized
            unitsres    INTEGER,               -- residential unit count
            unitstotal  INTEGER,               -- total unit count
            yearbuilt   SMALLINT,
            landuse     TEXT,                  -- 2-digit landuse code
            bldgclass   TEXT,
            appbbl      TEXT,                  -- billing BBL for condos
            nta2020     TEXT,                  -- NTA 2020 code from DCP field
            ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX ON raw.pluto (nta2020)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS raw.pluto CASCADE")
