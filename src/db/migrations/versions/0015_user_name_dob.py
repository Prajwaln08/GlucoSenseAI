"""split user name + store date of birth (mobile onboarding)

  users:
    first_name     — given name (Home greeting, coach personalisation)
    last_name      — family name
    date_of_birth  — exact DOB (age is derived from this)

Revision ID: 0015
Revises:     0014
"""
from __future__ import annotations

from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "0015"
down_revision: Union[str, None] = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("first_name", sa.String(), nullable=True))
    op.add_column("users", sa.Column("last_name", sa.String(), nullable=True))
    op.add_column("users", sa.Column("date_of_birth", sa.Date(), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "date_of_birth")
    op.drop_column("users", "last_name")
    op.drop_column("users", "first_name")
