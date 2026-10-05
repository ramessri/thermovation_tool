"""Add per-project marker_type: which scale fiducial the scan uses.

"aruco" (default) — printed ArUco markers (DICT_4X4_100)
"grid"            — custom 3×3 grid marker sheet (28.6 × 20.2 cm)

Revision ID: 013
Revises: 012
Create Date: 2026-10-05
"""

from alembic import op
import sqlalchemy as sa

revision = '013'
down_revision = '012'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('projects', sa.Column('marker_type', sa.String(32),
                                        nullable=False, server_default='aruco'))


def downgrade():
    op.drop_column('projects', 'marker_type')
