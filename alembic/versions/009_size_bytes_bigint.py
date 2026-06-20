"""Change size_bytes columns from INTEGER to BIGINT to support files > 2 GB.

Revision ID: 009
Revises: 008
Create Date: 2026-05-30
"""

from alembic import op
import sqlalchemy as sa

revision = '009'
down_revision = '008'
branch_labels = None
depends_on = None


def upgrade():
    op.alter_column('uploads', 'size_bytes',
                    existing_type=sa.Integer(),
                    type_=sa.BigInteger(),
                    existing_nullable=True)
    op.alter_column('outputs', 'size_bytes',
                    existing_type=sa.Integer(),
                    type_=sa.BigInteger(),
                    existing_nullable=True)


def downgrade():
    op.alter_column('outputs', 'size_bytes',
                    existing_type=sa.BigInteger(),
                    type_=sa.Integer(),
                    existing_nullable=True)
    op.alter_column('uploads', 'size_bytes',
                    existing_type=sa.BigInteger(),
                    type_=sa.Integer(),
                    existing_nullable=True)
