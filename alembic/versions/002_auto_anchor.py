"""Auto-anchor discovery columns and new enum values.

Revision ID: 002
Revises: 001
Create Date: 2026-05-16

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Extend enums BEFORE adding columns that reference them.
    op.execute("ALTER TYPE projectstatus ADD VALUE IF NOT EXISTS 'awaiting_scale_confirmation'")
    op.execute("ALTER TYPE pipelinestage ADD VALUE IF NOT EXISTS 'auto_anchor'")
    op.execute("ALTER TYPE pipelinestage ADD VALUE IF NOT EXISTS 'apply_scale'")

    # Add new columns to the projects table.
    op.add_column('projects', sa.Column('confirmed_scale_factor', sa.Float(), nullable=True))
    op.add_column('projects', sa.Column('confirmed_scale_source', sa.String(256), nullable=True))
    op.add_column('projects', sa.Column('anchor_candidates', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('projects', 'anchor_candidates')
    op.drop_column('projects', 'confirmed_scale_source')
    op.drop_column('projects', 'confirmed_scale_factor')
    # NOTE: Postgres does not support removing enum values; the enum extensions
    # from upgrade() cannot be reversed without dropping and recreating the types.
    # Downgrade leaves the enum values in place (harmless for the previous revision).
