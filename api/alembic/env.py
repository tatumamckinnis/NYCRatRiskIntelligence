"""Alembic migration environment.

Migrations use DIRECT_DATABASE_URL pointed at the Supabase SESSION pooler
(same host as DATABASE_URL, port 5432 instead of 6543). The direct Postgres
host (db.<ref>.supabase.co) only has an AAAA (IPv6) record on this project,
which Python's asyncio does not fall back to reliably on macOS. The session
pooler resolves to IPv4 and fully supports DDL (CREATE TABLE, CREATE INDEX,
etc.) — only the transaction pooler (port 6543) blocks DDL.

Driver note: the app uses asyncpg exclusively (spec §4.2). Alembic's async
runner pattern (create_async_engine + connection.run_sync) lets Alembic drive
migrations over asyncpg without needing psycopg2 at all.  This has been
supported since Alembic 1.8 / SQLAlchemy 1.4.

Usage:
    uv run --package rat-api alembic -c api/alembic.ini upgrade head
"""

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# We use raw SQL in migrations (PostGIS, custom types) so target_metadata is None.
target_metadata = None


def _async_url() -> str:
    """Return DIRECT_DATABASE_URL rewritten to the postgresql+asyncpg scheme."""
    url = os.environ.get("DIRECT_DATABASE_URL")
    if url is None:
        raise RuntimeError(
            "DIRECT_DATABASE_URL is not set. "
            "Export it or add it to .env before running Alembic."
        )
    # Supabase and most hosted Postgres providers hand out postgresql:// or
    # postgres:// URLs.  SQLAlchemy requires the +asyncpg dialect suffix.
    url = url.replace("postgres://", "postgresql://", 1)
    if not url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


def run_migrations_offline() -> None:
    """Generate SQL without connecting — useful for review/dry-run."""
    context.configure(
        url=_async_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:  # type: ignore[no-untyped-def]
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations over an asyncpg connection via SQLAlchemy's async engine."""
    engine = create_async_engine(_async_url(), poolclass=NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
