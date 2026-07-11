"""Per-user learned clock offset for Junction provider streams.

Some providers (freestyle_libre observed) stamp readings in device-local /
skewed time labeled as UTC with no timezone metadata. The offset is learned
from physically-impossible future stamps and ratchets upward only — persisting
it keeps every ingest path (webhook, poll, manual sync) on ONE correction
basis, so dedup-by-timestamp stays stable across batches.

Revision ID: 0018
Revises: 0017
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0018"
down_revision: Union[str, None] = "0017"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("junction_clock_offset_min", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "junction_clock_offset_min")
