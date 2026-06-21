import os
from logging.config import fileConfig

from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool
from alembic import context

load_dotenv()  # must run before reading DATABASE_URL below

# Alembic Config object — gives access to values in alembic.ini
config = context.config

# Build the DB URL — remap CockroachDB hosts to the correct dialect
_db_url = os.environ.get(
    "DATABASE_URL",
    "postgresql://glucosense:changeme@localhost:5432/glucosense",
)
if "cockroachlabs.cloud" in _db_url and _db_url.startswith("postgresql://"):
    _db_url = _db_url.replace("postgresql://", "cockroachdb+psycopg2://", 1)

config.set_main_option("sqlalchemy.url", _db_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import all models so autogenerate can detect schema changes
from src.db.base import Base  # noqa: E402
import src.db.models  # noqa: E402, F401

target_metadata = Base.metadata


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
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
