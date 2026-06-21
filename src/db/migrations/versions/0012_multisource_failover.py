"""multi-source CGM failover bookkeeping + provenance + per-user xDRIP key

Supports Junction-PRIMARY → xDRIP-FALLBACK:
  users:
    cgm_active_source        — "junction" | "xdrip" (single source of truth for the UI)
    cgm_last_junction_ok_at  — heartbeat: last Junction reading / healthy check
    cgm_last_xdrip_at        — heartbeat: last xDRIP push
    cgm_api_key              — per-user secret embedded in the xDRIP push URL
  cgm_readings:
    device_type              — provenance ("cgm")
    ingested_via             — provenance ("webhook" | "poll" | "manual_sync" | "push")

Also backfills legacy NULL cgm_readings.source to "junction".

Revision ID: 0012
Revises:     0011
"""
from __future__ import annotations

from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "0012"
down_revision: Union[str, None] = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── users: failover state + per-user xDRIP key ────────────────────────────
    op.add_column("users", sa.Column("cgm_active_source", sa.String(), nullable=True))
    op.add_column("users", sa.Column("cgm_last_junction_ok_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("users", sa.Column("cgm_last_xdrip_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("users", sa.Column("cgm_api_key", sa.String(), nullable=True))
    op.create_index("ix_users_cgm_api_key", "users", ["cgm_api_key"])

    # ── cgm_readings: provenance ──────────────────────────────────────────────
    op.add_column("cgm_readings", sa.Column("device_type", sa.String(), nullable=True))
    op.add_column("cgm_readings", sa.Column("ingested_via", sa.String(), nullable=True))

    # ── backfill legacy NULL sources (default added back in 0005) ─────────────
    op.execute("UPDATE cgm_readings SET source='junction' WHERE source IS NULL")


def downgrade() -> None:
    op.drop_column("cgm_readings", "ingested_via")
    op.drop_column("cgm_readings", "device_type")
    op.drop_index("ix_users_cgm_api_key", table_name="users")
    op.drop_column("users", "cgm_api_key")
    op.drop_column("users", "cgm_last_xdrip_at")
    op.drop_column("users", "cgm_last_junction_ok_at")
    op.drop_column("users", "cgm_active_source")
