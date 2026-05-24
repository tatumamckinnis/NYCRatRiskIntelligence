"""Add app.tft_forecasts table for TFT 12-week probabilistic forecasts.

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-24
"""

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS app.tft_forecasts (
            nta_id              TEXT NOT NULL,
            as_of_week          DATE NOT NULL,   -- Monday of the week the forecast was generated
            forecast_week       DATE NOT NULL,   -- Monday of the week being forecasted
            p10                 NUMERIC NOT NULL,
            p50                 NUMERIC NOT NULL,
            p90                 NUMERIC NOT NULL,
            model_version       TEXT NOT NULL,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (nta_id, as_of_week, forecast_week, model_version)
        )
    """)
    op.execute("CREATE INDEX ON app.tft_forecasts (nta_id, as_of_week)")
    op.execute("CREATE INDEX ON app.tft_forecasts (forecast_week)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS app.tft_forecasts CASCADE")
