"""HVAC wall-mount placement: per-project opt-in + results.

- hvac_mode          — per-project opt-in (also needs ENABLE_HVAC_PLACEMENT)
- hvac_placement     — {status, candidates[{rank, ..., corners_world_m}], overlay_image_key}
- hvac_segmentation  — {wall_candidates, hvac_fixtures, rucklauf_position, vorlauf_position}

Revision ID: 016
Revises: 015
Create Date: 2026-10-05
"""

from alembic import op

revision = '016'
down_revision = '015'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS hvac_mode BOOLEAN NOT NULL DEFAULT false")
    op.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS hvac_placement JSON")
    op.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS hvac_segmentation JSON")


def downgrade():
    op.drop_column('projects', 'hvac_segmentation')
    op.drop_column('projects', 'hvac_placement')
    op.drop_column('projects', 'hvac_mode')
