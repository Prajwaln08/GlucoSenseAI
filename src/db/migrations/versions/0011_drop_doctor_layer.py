"""drop the doctor layer (single end-user role)

Removes everything doctor/admin per the GlucoSense AI single-user direction:

  - DROP TABLE messages            (doctor–patient chat)
  - DROP users.assigned_doctor_id  (+ FK fk_users_assigned_doctor)
  - DROP users.is_doctor
  - DROP retrain_jobs.notes         (doctor's note; triggered_by is now
                                     "auto_drift" | "patient_request")

The `messages` table is dropped FIRST so the subsequent users-column drops have
no dependents. Mirrors the (proven) inverse ops from migrations 0002/0003/0004/0009.

⚠️ downgrade() recreates the empty structures but is DATA-LOSSY: any messages or
doctor assignments created after this upgrade cannot be restored.

Revision ID: 0011
Revises:     0010
"""
from __future__ import annotations

from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: Union[str, None] = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Doctor–patient chat table (FKs sender_id/receiver_id → users.id).
    op.drop_table("messages")

    # 2. Self-referential doctor-assignment FK + column.
    op.drop_constraint("fk_users_assigned_doctor", "users", type_="foreignkey")
    op.drop_column("users", "assigned_doctor_id")

    # 3. Doctor role flag.
    op.drop_column("users", "is_doctor")

    # 4. Doctor's retrain note.
    op.drop_column("retrain_jobs", "notes")


def downgrade() -> None:
    # Recreate in reverse. NOTE: data is NOT restored (lossy by nature).
    op.add_column("retrain_jobs", sa.Column("notes", sa.Text(), nullable=True))

    op.add_column(
        "users",
        sa.Column("is_doctor", sa.Boolean(), nullable=False, server_default=sa.false()),
    )

    op.add_column("users", sa.Column("assigned_doctor_id", sa.String(), nullable=True))
    op.create_foreign_key(
        "fk_users_assigned_doctor",
        "users", "users",
        ["assigned_doctor_id"], ["id"],
    )

    # messages at head-0010 shape: 0003 base columns + 0004 attachment columns.
    op.create_table(
        "messages",
        sa.Column("id",          sa.String(),  primary_key=True),
        sa.Column("sender_id",   sa.String(),  sa.ForeignKey("users.id"), nullable=False),
        sa.Column("receiver_id", sa.String(),  sa.ForeignKey("users.id"), nullable=False),
        sa.Column("body",        sa.Text(),    nullable=False),
        sa.Column("sent_at",     sa.DateTime(timezone=True), nullable=False),
        sa.Column("read_at",     sa.DateTime(timezone=True), nullable=True),
        sa.Column("attachment_url",  sa.String(), nullable=True),
        sa.Column("attachment_type", sa.String(), nullable=True),
        sa.Column("attachment_name", sa.String(), nullable=True),
    )
    op.create_index("ix_messages_sender_id",   "messages", ["sender_id"])
    op.create_index("ix_messages_receiver_id", "messages", ["receiver_id"])
    op.create_index("ix_messages_sent_at",     "messages", ["sent_at"])
