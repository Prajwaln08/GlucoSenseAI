"""add chat attachments and food quantity/portion_size

Revision ID: 0004
Revises: 0003
Create Date: 2026-06-03
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── messages: attachment support ──────────────────────────────────────────
    op.add_column("messages", sa.Column("attachment_url",  sa.String(), nullable=True))
    op.add_column("messages", sa.Column("attachment_type", sa.String(), nullable=True))  # image|pdf
    op.add_column("messages", sa.Column("attachment_name", sa.String(), nullable=True))

    # ── food_logs: quantity + portion size ────────────────────────────────────
    op.add_column("food_logs", sa.Column("quantity",     sa.Float(),  nullable=True))
    op.add_column("food_logs", sa.Column("portion_size", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "attachment_name")
    op.drop_column("messages", "attachment_type")
    op.drop_column("messages", "attachment_url")
    op.drop_column("food_logs", "portion_size")
    op.drop_column("food_logs", "quantity")
