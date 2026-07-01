"""CGM sensor sessions + per-phase personalization tracking

  cgm_sessions   — one row per CGM sensor journey (~14 days); drives the
                   personalization phase state machine (when to train the
                   personal while_on_cgm / post_cgm models).
  retrain_jobs:
    phase        — "while_on_cgm" | "post_cgm" (which personal model this job trains)
    session_id   — the CgmSession the training data came from

Revision ID: 0016
Revises:     0015
"""
from __future__ import annotations

from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "0016"
down_revision: Union[str, None] = "0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "cgm_sessions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id_fk", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("first_reading_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_reading_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("n_readings", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("n_days", sa.Float(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(), nullable=False, server_default="active"),
        sa.Column("end_reason", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_cgm_sessions_user_id_fk", "cgm_sessions", ["user_id_fk"])
    op.create_index("ix_cgm_sessions_last_reading_at", "cgm_sessions", ["last_reading_at"])

    op.add_column("retrain_jobs", sa.Column("phase", sa.String(), nullable=True))
    op.add_column("retrain_jobs", sa.Column("session_id", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("retrain_jobs", "session_id")
    op.drop_column("retrain_jobs", "phase")
    op.drop_index("ix_cgm_sessions_last_reading_at", table_name="cgm_sessions")
    op.drop_index("ix_cgm_sessions_user_id_fk", table_name="cgm_sessions")
    op.drop_table("cgm_sessions")
