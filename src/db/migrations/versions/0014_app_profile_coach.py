"""app profile fields, vitals, and coach tables (mobile Phases 2–4)

  users:
    user_id / dataset       — relaxed to NULLable (real app users have no research id)
    medications             — free-text / JSON meds list
    bp_systolic/diastolic   — last-recorded blood pressure
    bp_recorded_at          — when BP was recorded
    onboarding_complete     — gate for the app onboarding flow

  vitals          — user-logged BP / weight / glucose / hba1c (Home CTA or chat)
  chat_messages   — coach conversation turns
  recommendations — coach-generated diet/activity suggestions for the dashboard

Revision ID: 0014
Revises:     0013
"""
from __future__ import annotations

from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "0014"
down_revision: Union[str, None] = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── users: relax research linkage + add app/profile fields ──
    op.alter_column("users", "user_id", existing_type=sa.String(), nullable=True)
    op.alter_column("users", "dataset", existing_type=sa.String(), nullable=True)
    op.add_column("users", sa.Column("medications", sa.Text(), nullable=True))
    op.add_column("users", sa.Column("bp_systolic", sa.Integer(), nullable=True))
    op.add_column("users", sa.Column("bp_diastolic", sa.Integer(), nullable=True))
    op.add_column("users", sa.Column("bp_recorded_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("users", sa.Column(
        "onboarding_complete", sa.Boolean(), nullable=False, server_default=sa.false()))

    # ── vitals ──
    op.create_table(
        "vitals",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id_fk", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("value", sa.Float(), nullable=True),
        sa.Column("bp_systolic", sa.Integer(), nullable=True),
        sa.Column("bp_diastolic", sa.Integer(), nullable=True),
        sa.Column("source", sa.String(), nullable=True),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_vitals_user_id_fk", "vitals", ["user_id_fk"])

    # ── chat_messages ──
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id_fk", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_chat_messages_user_id_fk", "chat_messages", ["user_id_fk"])

    # ── recommendations ──
    op.create_table(
        "recommendations",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id_fk", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_recommendations_user_id_fk", "recommendations", ["user_id_fk"])


def downgrade() -> None:
    op.drop_index("ix_recommendations_user_id_fk", table_name="recommendations")
    op.drop_table("recommendations")
    op.drop_index("ix_chat_messages_user_id_fk", table_name="chat_messages")
    op.drop_table("chat_messages")
    op.drop_index("ix_vitals_user_id_fk", table_name="vitals")
    op.drop_table("vitals")

    op.drop_column("users", "onboarding_complete")
    op.drop_column("users", "bp_recorded_at")
    op.drop_column("users", "bp_diastolic")
    op.drop_column("users", "bp_systolic")
    op.drop_column("users", "medications")
    op.alter_column("users", "dataset", existing_type=sa.String(), nullable=False)
    op.alter_column("users", "user_id", existing_type=sa.String(), nullable=False)
