"""google fit identity + tokens on users (sole Huawei-watch source)

Google Fit access tokens expire ~1h, so a stored refresh token + expiry is required
for background watch sync.

  users:
    google_fit_user_id        — Google account / Fit user id (indexed)
    google_fit_refresh_token  — long-lived refresh token (encrypt at rest in Phase 5)
    google_fit_token_expiry   — current access-token expiry
    google_fit_scopes         — granted scopes (audit)
    google_fit_last_sync_at   — last successful activity sync

Revision ID: 0013
Revises:     0012
"""
from __future__ import annotations

from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "0013"
down_revision: Union[str, None] = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("google_fit_user_id", sa.String(), nullable=True))
    op.create_index("ix_users_google_fit_user_id", "users", ["google_fit_user_id"])
    op.add_column("users", sa.Column("google_fit_refresh_token", sa.Text(), nullable=True))
    op.add_column("users", sa.Column("google_fit_token_expiry", sa.DateTime(timezone=True), nullable=True))
    op.add_column("users", sa.Column("google_fit_scopes", sa.Text(), nullable=True))
    op.add_column("users", sa.Column("google_fit_last_sync_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "google_fit_last_sync_at")
    op.drop_column("users", "google_fit_scopes")
    op.drop_column("users", "google_fit_token_expiry")
    op.drop_column("users", "google_fit_refresh_token")
    op.drop_index("ix_users_google_fit_user_id", table_name="users")
    op.drop_column("users", "google_fit_user_id")
