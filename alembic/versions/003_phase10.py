"""Phase 10: AI scene understanding + depth fusion columns.

Revision ID: 003
Revises: 002
Create Date: 2026-05-16

"""
from alembic import op
import sqlalchemy as sa

revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Extend pipelinestage enum with Phase 10 stages.
    op.execute("ALTER TYPE pipelinestage ADD VALUE IF NOT EXISTS 'scene_understanding'")
    op.execute("ALTER TYPE pipelinestage ADD VALUE IF NOT EXISTS 'depth_fusion'")

    # Add scene_analysis JSON column to projects table.
    op.add_column('projects', sa.Column('scene_analysis', sa.JSON, nullable=True))


def downgrade() -> None:
    op.drop_column('projects', 'scene_analysis')
    # Postgres does not support removing enum values; downgrade leaves the enum values in place.
