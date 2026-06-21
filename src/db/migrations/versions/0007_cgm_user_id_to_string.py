"""alter cgm_readings.user_id from INT to STRING (UUID)

The table was originally created outside Alembic with user_id as INTEGER.
Users.id is a UUID string, so this column must be STRING to accept UUIDs.
We also need source_device_id to exist (added in 0005 as part of wearable work).

Revision ID: 0007
Revises: 0006
Create Date: 2026-06-06
"""

from typing import Sequence, Union

from alembic import op

revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from sqlalchemy import text

    bind = op.get_bind()

    # Check current user_id type via information_schema (avoids SQLAlchemy type-mapping quirks)
    result = bind.execute(text(
        "SELECT data_type FROM information_schema.columns "
        "WHERE table_name = 'cgm_readings' AND column_name = 'user_id'"
    ))
    row = result.fetchone()
    if row is None:
        return  # table or column missing — nothing to do
    if "INT" not in row[0].upper():
        return  # already STRING — idempotent exit

    # user_id is INT; drop and recreate with STRING user_id.
    # No production data — only stale integer-keyed rows that never match a UUID user.
    bind.execute(text("DROP TABLE IF EXISTS cgm_readings"))
    bind.execute(text("""
        CREATE TABLE cgm_readings (
            id                     STRING      NOT NULL PRIMARY KEY,
            user_id                STRING      NOT NULL,
            timestamp              TIMESTAMPTZ NOT NULL,
            glucose_mgdl           FLOAT8      NOT NULL,
            glucose_rate_of_change FLOAT8,
            transmitter_time       STRING,
            source_device_id       STRING,
            glucose_mmol           FLOAT8,
            source                 STRING,
            created_at             TIMESTAMPTZ,
            direction              STRING,
            INDEX ix_cgm_readings_user_id  (user_id),
            INDEX ix_cgm_readings_timestamp (timestamp)
        )
    """))


def downgrade() -> None:
    pass  # intentionally irreversible — no safe way to cast UUID strings back to INT
