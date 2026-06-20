"""004 — Semantic geometry engine: new pipeline stages + project columns.

Adds PipelineStage enum values for geometry_healing, semantic_labeling,
geometry_correction. Adds object_labels and confidence_map_key columns.
"""

import sqlalchemy as sa
from alembic import op

revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE pipelinestage ADD VALUE IF NOT EXISTS 'geometry_healing'")
    op.execute("ALTER TYPE pipelinestage ADD VALUE IF NOT EXISTS 'semantic_labeling'")
    op.execute("ALTER TYPE pipelinestage ADD VALUE IF NOT EXISTS 'geometry_correction'")

    op.add_column('projects', sa.Column('object_labels', sa.JSON, nullable=True))
    op.add_column('projects', sa.Column('confidence_map_key', sa.Text, nullable=True))


def downgrade() -> None:
    op.drop_column('projects', 'confidence_map_key')
    op.drop_column('projects', 'object_labels')
    # Postgres does not support removing enum values; leave them in place.
