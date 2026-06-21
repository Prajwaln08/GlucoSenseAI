"""Add artefact_dir, triggered_by, notes, version_tag to retrain_jobs.

artefact_dir  — path to trained model artifacts; stored so doctors can restore
                a previous version by re-pointing the registry at this dir.
triggered_by  — audit trail: "doctor" | "auto_drift" | "patient_request"
notes         — optional doctor note explaining why the retrain was triggered.
version_tag   — auto-incremented per (user, dataset, horizon): "v1", "v2", …

Revision ID: 0009
Revises:     0008
"""
from __future__ import annotations

from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: Union[str, None] = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "retrain_jobs",
        sa.Column("artefact_dir", sa.Text(), nullable=True),
    )
    op.add_column(
        "retrain_jobs",
        sa.Column("triggered_by", sa.String(), nullable=True),
    )
    op.add_column(
        "retrain_jobs",
        sa.Column("notes", sa.Text(), nullable=True),
    )
    op.add_column(
        "retrain_jobs",
        sa.Column("version_tag", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("retrain_jobs", "version_tag")
    op.drop_column("retrain_jobs", "notes")
    op.drop_column("retrain_jobs", "triggered_by")
    op.drop_column("retrain_jobs", "artefact_dir")
