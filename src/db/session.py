"""
SQLAlchemy sync engine + session factory.

Uses psycopg2 (sync) so both FastAPI routes (run in threadpool when defined
as `def`, not `async def`) and Celery workers share the same session pattern.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from src.db.base import Base  # noqa: F401 — imported so Alembic can see all models
import src.db.models  # noqa: F401 — register models on Base.metadata

DATABASE_URL: str = os.environ.get(
    "DATABASE_URL",
    "postgresql://glucosense:changeme@localhost:5432/glucosense",
)

# Strip async driver prefix if someone accidentally sets asyncpg URL
if DATABASE_URL.startswith("postgresql+asyncpg://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://", 1)

# CockroachDB requires its own SQLAlchemy dialect for correct version detection.
# Handle raw cockroachdb:// scheme (no driver specified)
if DATABASE_URL.startswith("cockroachdb://"):
    DATABASE_URL = DATABASE_URL.replace("cockroachdb://", "cockroachdb+psycopg2://", 1)
# Remap postgresql:// → cockroachdb+psycopg2:// when host is a CockroachDB cloud URL.
elif "cockroachlabs.cloud" in DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "cockroachdb+psycopg2://", 1)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI dependency — yields a DB session and always closes it."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_all_tables() -> None:
    """Create all tables (idempotent). Used in tests; prefer Alembic for production."""
    Base.metadata.create_all(bind=engine)
