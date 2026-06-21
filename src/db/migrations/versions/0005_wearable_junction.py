"""add junction wearable integration tables

Revision ID: 0005
Revises: 0004
Create Date: 2026-06-05

Notes:
  - users.junction_user_id already exists (added in a partial prior run); skip it
  - cgm_readings already exists with different column names; add missing columns only
  - wearable_activity is new
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── users.junction_user_id ────────────────────────────────────────────────
    # Already added in a partial prior run — guard with a check to stay idempotent
    from sqlalchemy import inspect, text
    conn = op.get_bind()
    insp = inspect(conn)
    existing_user_cols = [c["name"] for c in insp.get_columns("users")]
    if "junction_user_id" not in existing_user_cols:
        op.add_column("users", sa.Column("junction_user_id", sa.String(), nullable=True))
        op.create_index("ix_users_junction_user_id", "users", ["junction_user_id"])

    # ── cgm_readings: add missing columns ────────────────────────────────────
    existing_cgm_cols = [c["name"] for c in insp.get_columns("cgm_readings")]
    if "glucose_mmol" not in existing_cgm_cols:
        op.add_column("cgm_readings", sa.Column("glucose_mmol", sa.Float(), nullable=True))
    if "source" not in existing_cgm_cols:
        op.add_column("cgm_readings", sa.Column("source", sa.String(), nullable=True))
    if "created_at" not in existing_cgm_cols:
        op.add_column("cgm_readings", sa.Column("created_at", sa.DateTime(timezone=True), nullable=True))

    # ── wearable_activity ─────────────────────────────────────────────────────
    op.create_table(
        "wearable_activity",
        sa.Column("id",              sa.String(),                primary_key=True),
        sa.Column("user_id_fk",      sa.String(),                nullable=False),
        sa.Column("calendar_date",   sa.String(),                nullable=False),
        sa.Column("steps",           sa.Integer(),               nullable=True),
        sa.Column("calories_total",  sa.Float(),                 nullable=True),
        sa.Column("calories_active", sa.Float(),                 nullable=True),
        sa.Column("distance_m",      sa.Float(),                 nullable=True),
        sa.Column("hr_avg_bpm",      sa.Float(),                 nullable=True),
        sa.Column("hr_min_bpm",      sa.Float(),                 nullable=True),
        sa.Column("hr_max_bpm",      sa.Float(),                 nullable=True),
        sa.Column("hr_resting_bpm",  sa.Float(),                 nullable=True),
        sa.Column("provider",        sa.String(),                nullable=True),
        sa.Column("created_at",      sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id_fk"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_wearable_activity_user_id_fk",    "wearable_activity", ["user_id_fk"])
    op.create_index("ix_wearable_activity_calendar_date", "wearable_activity", ["calendar_date"])


def downgrade() -> None:
    op.drop_table("wearable_activity")
    # Don't drop junction_user_id or cgm_readings columns — too destructive for downgrade
