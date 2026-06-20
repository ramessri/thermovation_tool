"""Add calibration_data column for camera focal length from still photo.

Revision ID: 007
Revises: 006
Create Date: 2026-05-28
"""
from alembic import op
import sqlalchemy as sa

revision = '007'
down_revision = '006'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('projects',
        sa.Column('calibration_data', sa.JSON(), nullable=True,
                  comment='Focal length calibration from a still image: '
                          '{make, model, fl_mm, fl_35mm, sensor_width_mm, '
                          'fl_px_at_width, calibration_width, source}'))


def downgrade():
    op.drop_column('projects', 'calibration_data')
