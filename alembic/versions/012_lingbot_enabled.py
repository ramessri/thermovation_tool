"""Add per-project lingbot_enabled opt-in flag for the densification stage.

Revision ID: 012
Revises: 011
Create Date: 2026-06-30
"""

from alembic import op
import sqlalchemy as sa

revision = '012'
down_revision = '011'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('projects', sa.Column('lingbot_enabled', sa.Boolean(),
                                        nullable=False, server_default=sa.text('false')))


def downgrade():
    op.drop_column('projects', 'lingbot_enabled')
