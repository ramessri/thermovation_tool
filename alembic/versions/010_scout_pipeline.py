"""Add scout_calibration and pipeline_mode columns for two-pass adaptive pipeline.

Revision ID: 010
Revises: 009
Create Date: 2026-05-30
"""

from alembic import op
import sqlalchemy as sa

revision = '010'
down_revision = '009'
branch_labels = None
depends_on = None


def upgrade():
    # Stores calibration JSON from scout run: target_frames, match_window, etc.
    op.add_column('projects', sa.Column('scout_calibration', sa.JSON(), nullable=True))
    # 'standard' | 'scout' | 'full' — tracks which pipeline mode this project used
    op.add_column('projects', sa.Column('pipeline_mode', sa.String(32), nullable=True))


def downgrade():
    op.drop_column('projects', 'pipeline_mode')
    op.drop_column('projects', 'scout_calibration')
