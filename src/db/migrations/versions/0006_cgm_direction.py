"""add direction column to cgm_readings

Revision ID: 0006
Revises: 0005
Create Date: 2026-06-06
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from sqlalchemy import inspect
    conn = op.get_bind()
    existing = [c["name"] for c in inspect(conn).get_columns("cgm_readings")]
    if "direction" not in existing:
        op.add_column("cgm_readings", sa.Column("direction", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("cgm_readings", "direction")
