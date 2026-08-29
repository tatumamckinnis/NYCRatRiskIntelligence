"""Reindex health_code_chunks.content_tsv on content_with_prefix, not content.

BM25 recall was structurally capped because the generated tsvector only
covered `content`, not `content_with_prefix` (where document/authority
context like "ECB Penalty Schedule" or "NYC Health Code" lives) — a query
that names the source document by name got zero credit from that context
for every chunk in the corpus. Postgres generated columns can't have their
expression altered in place, so this drops and recreates both the column
and its GIN index.

Revision ID: 0006
Revises: 0005
Create Date: 2026-08-29
"""

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS app.health_code_chunks_content_tsv_idx")
    op.execute("ALTER TABLE app.health_code_chunks DROP COLUMN content_tsv")
    op.execute("""
        ALTER TABLE app.health_code_chunks
        ADD COLUMN content_tsv TSVECTOR GENERATED ALWAYS AS (
            to_tsvector('english', content_with_prefix)
        ) STORED
    """)
    op.execute("CREATE INDEX ON app.health_code_chunks USING GIN (content_tsv)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS app.health_code_chunks_content_tsv_idx")
    op.execute("ALTER TABLE app.health_code_chunks DROP COLUMN content_tsv")
    op.execute("""
        ALTER TABLE app.health_code_chunks
        ADD COLUMN content_tsv TSVECTOR GENERATED ALWAYS AS (
            to_tsvector('english', content)
        ) STORED
    """)
    op.execute("CREATE INDEX ON app.health_code_chunks USING GIN (content_tsv)")
