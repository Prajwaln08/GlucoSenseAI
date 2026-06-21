"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-05-31

Creates: users, prediction_logs, retrain_jobs
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id",              sa.String(),  nullable=False),
        sa.Column("email",           sa.String(),  nullable=False),
        sa.Column("hashed_password", sa.String(),  nullable=False),
        sa.Column("user_id",         sa.String(),  nullable=False),
        sa.Column("dataset",         sa.String(),  nullable=False),
        sa.Column("age",             sa.Float(),   nullable=True),
        sa.Column("bmi",             sa.Float(),   nullable=True),
        sa.Column("hba1c",           sa.Float(),   nullable=True),
        sa.Column("is_active",       sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at",      sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )
    op.create_index("ix_users_email",   "users", ["email"],   unique=True)
    op.create_index("ix_users_user_id", "users", ["user_id"], unique=False)

    op.create_table(
        "prediction_logs",
        sa.Column("id",              sa.String(),  nullable=False),
        sa.Column("user_id_fk",      sa.String(),  nullable=False),
        sa.Column("dataset",         sa.String(),  nullable=False),
        sa.Column("horizon",         sa.String(),  nullable=False),
        sa.Column("model_type",      sa.String(),  nullable=False),
        sa.Column("current_glucose", sa.Float(),   nullable=False),
        sa.Column("horizon_glucose", sa.Float(),   nullable=False),
        sa.Column("clarke_zone",     sa.String(),  nullable=False),
        sa.Column("is_hypo_risk",    sa.Boolean(), nullable=False),
        sa.Column("is_hyper_risk",   sa.Boolean(), nullable=False),
        sa.Column("n_readings",      sa.Integer(), nullable=False),
        sa.Column("predicted_at",    sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("actual_glucose",  sa.Float(),   nullable=True),
        sa.Column("absolute_error",  sa.Float(),   nullable=True),
        sa.ForeignKeyConstraint(["user_id_fk"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_prediction_logs_user_id_fk", "prediction_logs",
                    ["user_id_fk"], unique=False)

    op.create_table(
        "retrain_jobs",
        sa.Column("id",           sa.String(),  nullable=False),
        sa.Column("celery_id",    sa.String(),  nullable=True),
        sa.Column("user_id_fk",   sa.String(),  nullable=False),
        sa.Column("dataset",      sa.String(),  nullable=False),
        sa.Column("horizon",      sa.String(),  nullable=False),
        sa.Column("status",       sa.String(),  nullable=False, server_default="pending"),
        sa.Column("triggered_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("started_at",   sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("test_rmse",    sa.Float(),   nullable=True),
        sa.Column("prev_rmse",    sa.Float(),   nullable=True),
        sa.Column("improved",     sa.Boolean(), nullable=True),
        sa.Column("error_msg",    sa.Text(),    nullable=True),
        sa.ForeignKeyConstraint(["user_id_fk"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_retrain_jobs_user_id_fk", "retrain_jobs",
                    ["user_id_fk"], unique=False)
    op.create_index("ix_retrain_jobs_celery_id", "retrain_jobs",
                    ["celery_id"], unique=False)


def downgrade() -> None:
    op.drop_table("retrain_jobs")
    op.drop_table("prediction_logs")
    op.drop_table("users")
