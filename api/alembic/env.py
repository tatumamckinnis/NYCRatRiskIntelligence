"""Alembic migration environment.

Migrations use DIRECT_DATABASE_URL (a direct Supabase connection, bypassing
PgBouncer) because DDL statements (CREATE TABLE, ALTER TABLE, etc.) are not
compatible with PgBouncer's transaction-pooling mode.

Usage:
    uv run --package rat-api alembic -c api/alembic.ini upgrade head
"""

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Override the URL from alembic.ini with the direct (non-pooled) connection.
direct_url = os.environ.get("DIRECT_DATABASE_URL")
if direct_url is None:
    raise RuntimeError(
        "DIRECT_DATABASE_URL is not set. "
        "Export it or add it to .env before running Alembic."
    )
config.set_main_option("sqlalchemy.url", direct_url)

# We use raw SQL in migrations (PostGIS, custom types) so target_metadata is None.
target_metadata = None


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # NullPool: each migration gets a fresh connection.
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
