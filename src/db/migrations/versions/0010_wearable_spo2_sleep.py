"""Add spo2_avg, sleep_hours, sleep_score to wearable_activity

Revision ID: 0010
Revises: 0009
Create Date: 2026-06-15
"""
from alembic import op
import sqlalchemy as sa

revision = '0010'
down_revision = '0009'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('wearable_activity', sa.Column('spo2_avg',    sa.Float(),   nullable=True))
    op.add_column('wearable_activity', sa.Column('sleep_hours', sa.Float(),   nullable=True))
    op.add_column('wearable_activity', sa.Column('sleep_score', sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column('wearable_activity', 'sleep_score')
    op.drop_column('wearable_activity', 'sleep_hours')
    op.drop_column('wearable_activity', 'spo2_avg')
