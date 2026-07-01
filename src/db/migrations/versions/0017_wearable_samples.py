"""Intraday (realtime) wearable samples from Health Connect

  wearable_samples — timestamped HR / SpO2 / interval steps-calories-distance, so the
                     model gets real intraday HR features instead of a flat daily average.

Revision ID: 0017
Revises:     0016
"""
from __future__ import annotations

from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "0017"
down_revision: Union[str, None] = "0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "wearable_samples",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id_fk", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("hr_bpm", sa.Float(), nullable=True),
        sa.Column("spo2_pct", sa.Float(), nullable=True),
        sa.Column("steps", sa.Integer(), nullable=True),
        sa.Column("calories_active", sa.Float(), nullable=True),
        sa.Column("distance_m", sa.Float(), nullable=True),
        sa.Column("provider", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_wearable_samples_user_id_fk", "wearable_samples", ["user_id_fk"])
    op.create_index("ix_wearable_samples_timestamp", "wearable_samples", ["timestamp"])
    op.create_index("ix_wearable_samples_user_ts", "wearable_samples", ["user_id_fk", "timestamp"])


def downgrade() -> None:
    op.drop_index("ix_wearable_samples_user_ts", table_name="wearable_samples")
    op.drop_index("ix_wearable_samples_timestamp", table_name="wearable_samples")
    op.drop_index("ix_wearable_samples_user_id_fk", table_name="wearable_samples")
    op.drop_table("wearable_samples")
