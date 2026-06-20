"""Add clarifications column to projects.

Revision ID: 005
Revises: 004
"""
from alembic import op
import sqlalchemy as sa

revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('projects', sa.Column('clarifications', sa.JSON, nullable=True))


def downgrade():
    op.drop_column('projects', 'clarifications')
