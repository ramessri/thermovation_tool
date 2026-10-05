"""MetricAnything depth fusion: per-project opt-in flag + output artifact keys.

Revision ID: 015
Revises: 014
Create Date: 2026-10-05
"""

from alembic import op

revision = '015'
down_revision = '014'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS metricanything_enabled BOOLEAN "
               "NOT NULL DEFAULT false")
    op.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS metricanything_cloud_key TEXT")
    op.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS metricanything_mesh_key TEXT")


def downgrade():
    op.drop_column('projects', 'metricanything_mesh_key')
    op.drop_column('projects', 'metricanything_cloud_key')
    op.drop_column('projects', 'metricanything_enabled')
