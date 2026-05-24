"""Add app.narrative_cache for LLM narrative caching.

Revision ID: 0005
Revises: 0004
Create Date: 2026-05-24
"""

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS app.narrative_cache (
            cache_key       TEXT PRIMARY KEY,   -- sha1(nta_id|week|model_version)
            narrative       TEXT NOT NULL,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS app.narrative_cache CASCADE")
