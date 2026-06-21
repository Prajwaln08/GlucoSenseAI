"""Add model_tier, cgm_mode, data_quality, readings_used to prediction_logs.

Revision ID: 0008
Revises:     0007
"""
from __future__ import annotations

from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: Union[str, None] = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # model_tier: which tier was served for this prediction
    op.add_column(
        "prediction_logs",
        sa.Column("model_tier", sa.String(), nullable=True),
    )
    # cgm_mode: whether the prediction was made with or without live glucose
    op.add_column(
        "prediction_logs",
        sa.Column("cgm_mode", sa.String(), nullable=True),
    )
    # data_quality: "full" (>=24 HR readings) or "warming_up" (8-23 readings)
    op.add_column(
        "prediction_logs",
        sa.Column("data_quality", sa.String(), nullable=True),
    )
    # readings_used: how many HR readings were in the request window
    op.add_column(
        "prediction_logs",
        sa.Column("readings_used", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("prediction_logs", "readings_used")
    op.drop_column("prediction_logs", "data_quality")
    op.drop_column("prediction_logs", "cgm_mode")
    op.drop_column("prediction_logs", "model_tier")
