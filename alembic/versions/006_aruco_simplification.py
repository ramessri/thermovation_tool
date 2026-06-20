"""ArUco simplification: add scene_type, video_metadata, aruco columns; new pipeline stages.

Revision ID: 006
Revises: 005
Create Date: 2026-05-28
"""
from alembic import op
import sqlalchemy as sa

revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None


def upgrade():
    # ── New enums ─────────────────────────────────────────────────────────────
    # scene_type enum
    op.execute("CREATE TYPE scenetype AS ENUM ('indoor_room', 'outdoor', 'object')")

    # Add new PipelineStage values (PostgreSQL enums are append-only)
    for val in ('extract_metadata', 'detect_aruco', 'scale_from_aruco'):
        op.execute(f"ALTER TYPE pipelinestage ADD VALUE IF NOT EXISTS '{val}'")

    # ── New project columns ───────────────────────────────────────────────────
    op.add_column('projects',
        sa.Column('scene_type', sa.Enum('indoor_room', 'outdoor', 'object',
                                         name='scenetype'),
                  nullable=False, server_default='indoor_room'))

    op.add_column('projects',
        sa.Column('video_metadata', sa.JSON(), nullable=True))

    op.add_column('projects',
        sa.Column('aruco_markers', sa.JSON(), nullable=True))

    op.add_column('projects',
        sa.Column('gravity_up_world', sa.JSON(), nullable=True))

    # ── Remove server_default after adding (keep column nullable) ─────────────
    op.alter_column('projects', 'scene_type', server_default=None)


def downgrade():
    op.drop_column('projects', 'gravity_up_world')
    op.drop_column('projects', 'aruco_markers')
    op.drop_column('projects', 'video_metadata')
    op.drop_column('projects', 'scene_type')
    op.execute("DROP TYPE IF EXISTS scenetype")
