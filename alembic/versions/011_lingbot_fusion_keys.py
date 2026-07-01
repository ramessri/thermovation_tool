"""Add lingbot_cloud_key and lingbot_mesh_key columns for the optional
LingBot depth-fusion densification artifact.

Revision ID: 011
Revises: 010
Create Date: 2026-06-30
"""

from alembic import op
import sqlalchemy as sa

revision = '011'
down_revision = '010'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('projects', sa.Column('lingbot_cloud_key', sa.Text(), nullable=True))
    op.add_column('projects', sa.Column('lingbot_mesh_key', sa.Text(), nullable=True))


def downgrade():
    op.drop_column('projects', 'lingbot_mesh_key')
    op.drop_column('projects', 'lingbot_cloud_key')
