"""LiDAR .ply ingestion + real-world dimensions.

- projects.dimensions       — L x B x H computed at export time
- projects.scan_source      — 'video' (default) or 'lidar_ply'
- projects.pipeline_results — per-stage metrics written by emit_stage_complete
  (used by the model and UI but never added by an earlier migration; IF NOT
  EXISTS keeps this safe on databases that already have it)
- 'ingest_lidar_ply' added to the pipelinestage enum (chain-head stage for
  lidar_ply projects; job rows are inserted with stage cast ::pipelinestage)

Revision ID: 014
Revises: 013
Create Date: 2026-10-05
"""

from alembic import op

revision = '014'
down_revision = '013'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS pipeline_results JSON")
    op.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS dimensions JSON")
    op.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS scan_source VARCHAR(32) "
               "NOT NULL DEFAULT 'video'")
    op.execute("ALTER TYPE pipelinestage ADD VALUE IF NOT EXISTS 'ingest_lidar_ply'")


def downgrade():
    op.drop_column('projects', 'scan_source')
    op.drop_column('projects', 'dimensions')
    # pipeline_results predates this migration on some databases — keep it.
    # Postgres can't drop an enum value; leaving 'ingest_lidar_ply' is harmless.
