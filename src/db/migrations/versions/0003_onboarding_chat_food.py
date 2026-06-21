"""add onboarding fields, food_logs, messages

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-03
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── New columns on users ───────────────────────────────────────────────────
    op.add_column("users", sa.Column("name",            sa.String(),  nullable=True))
    op.add_column("users", sa.Column("gender",          sa.String(),  nullable=True))
    op.add_column("users", sa.Column("height_cm",       sa.Float(),   nullable=True))
    op.add_column("users", sa.Column("weight_kg",       sa.Float(),   nullable=True))
    op.add_column("users", sa.Column("diabetes_type",   sa.String(),  nullable=True))
    op.add_column("users", sa.Column("medical_history", sa.Text(),    nullable=True))
    op.add_column("users", sa.Column("assigned_doctor_id", sa.String(), nullable=True))
    op.create_foreign_key(
        "fk_users_assigned_doctor",
        "users", "users",
        ["assigned_doctor_id"], ["id"],
    )

    # ── food_logs ──────────────────────────────────────────────────────────────
    op.create_table(
        "food_logs",
        sa.Column("id",          sa.String(),  primary_key=True),
        sa.Column("user_id_fk",  sa.String(),  sa.ForeignKey("users.id"), nullable=False),
        sa.Column("logged_at",   sa.DateTime(timezone=True), nullable=False),
        sa.Column("meal_type",   sa.String(),  nullable=False),
        sa.Column("description", sa.Text(),    nullable=True),
        sa.Column("calories",    sa.Float(),   nullable=True),
        sa.Column("carbs_g",     sa.Float(),   nullable=True),
        sa.Column("protein_g",   sa.Float(),   nullable=True),
        sa.Column("fat_g",       sa.Float(),   nullable=True),
        sa.Column("fiber_g",     sa.Float(),   nullable=True),
        sa.Column("sugar_g",     sa.Float(),   nullable=True),
        sa.Column("gi_proxy",    sa.Float(),   nullable=True),
        sa.Column("notes",       sa.Text(),    nullable=True),
    )
    op.create_index("ix_food_logs_user_id_fk", "food_logs", ["user_id_fk"])
    op.create_index("ix_food_logs_logged_at",  "food_logs", ["logged_at"])

    # ── messages ───────────────────────────────────────────────────────────────
    op.create_table(
        "messages",
        sa.Column("id",          sa.String(),  primary_key=True),
        sa.Column("sender_id",   sa.String(),  sa.ForeignKey("users.id"), nullable=False),
        sa.Column("receiver_id", sa.String(),  sa.ForeignKey("users.id"), nullable=False),
        sa.Column("body",        sa.Text(),    nullable=False),
        sa.Column("sent_at",     sa.DateTime(timezone=True), nullable=False),
        sa.Column("read_at",     sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_messages_sender_id",   "messages", ["sender_id"])
    op.create_index("ix_messages_receiver_id", "messages", ["receiver_id"])
    op.create_index("ix_messages_sent_at",     "messages", ["sent_at"])


def downgrade() -> None:
    op.drop_table("messages")
    op.drop_index("ix_food_logs_logged_at",  table_name="food_logs")
    op.drop_index("ix_food_logs_user_id_fk", table_name="food_logs")
    op.drop_table("food_logs")
    op.drop_constraint("fk_users_assigned_doctor", "users", type_="foreignkey")
    op.drop_column("users", "assigned_doctor_id")
    op.drop_column("users", "medical_history")
    op.drop_column("users", "diabetes_type")
    op.drop_column("users", "weight_kg")
    op.drop_column("users", "height_cm")
    op.drop_column("users", "gender")
    op.drop_column("users", "name")
