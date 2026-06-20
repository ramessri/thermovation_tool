"""Add missing project columns: suggestions, coverage_runs, splat_key, mesh_key, anchor_reminder, quality_issues.

Revision ID: 008
Revises: 007
Create Date: 2026-05-29
"""

from alembic import op
import sqlalchemy as sa

revision = '008'
down_revision = '007'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('projects', sa.Column('suggestions',    sa.JSON(), nullable=True))
    op.add_column('projects', sa.Column('coverage_runs',  sa.JSON(), nullable=True))
    op.add_column('projects', sa.Column('splat_key',      sa.Text(), nullable=True))
    op.add_column('projects', sa.Column('mesh_key',       sa.Text(), nullable=True))
    op.add_column('projects', sa.Column('anchor_reminder', sa.Text(), nullable=True))
    op.add_column('projects', sa.Column('quality_issues', sa.JSON(), nullable=True))


def downgrade():
    op.drop_column('projects', 'quality_issues')
    op.drop_column('projects', 'anchor_reminder')
    op.drop_column('projects', 'mesh_key')
    op.drop_column('projects', 'splat_key')
    op.drop_column('projects', 'coverage_runs')
    op.drop_column('projects', 'suggestions')
