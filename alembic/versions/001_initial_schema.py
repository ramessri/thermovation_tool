"""Initial schema creation.

Revision ID: 001
Revises:
Create Date: 2025-04-26

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create projects table
    op.create_table(
        'projects',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('name', sa.String(256), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.Enum('created', 'processing', 'needs_more', 'complete', 'failed', name='projectstatus'), nullable=True),
        sa.Column('coverage_score', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('ix_projects_status', 'status')
    )

    # Create uploads table
    op.create_table(
        'uploads',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('project_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('storage_key', sa.String(1024), nullable=False),
        sa.Column('filename', sa.String(512), nullable=False),
        sa.Column('mime_type', sa.String(128), nullable=False),
        sa.Column('size_bytes', sa.Integer(), nullable=True),
        sa.Column('uploaded_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('ix_uploads_project_id', 'project_id')
    )

    # Create anchors table
    op.create_table(
        'anchors',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('project_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('name', sa.String(256), nullable=False),
        sa.Column('storage_key', sa.String(1024), nullable=False),
        sa.Column('mask_key', sa.String(1024), nullable=True),
        sa.Column('width_mm', sa.Float(), nullable=True),
        sa.Column('height_mm', sa.Float(), nullable=True),
        sa.Column('depth_mm', sa.Float(), nullable=True),
        sa.Column('feature_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('ix_anchors_project_id', 'project_id')
    )

    # Create jobs table
    op.create_table(
        'jobs',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('project_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('celery_id', sa.String(256), nullable=True),
        sa.Column('stage', sa.Enum('extract_frames', 'feature_matching', 'sfm', 'mvs', 'depth_estimation', 'scale_anchor', 'coverage', 'export', name='pipelinestage'), nullable=False),
        sa.Column('status', sa.Enum('pending', 'running', 'success', 'failed', 'retrying', name='jobstatus'), nullable=True),
        sa.Column('progress', sa.Float(), nullable=True),
        sa.Column('message', sa.Text(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('meta', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('ix_jobs_celery_id', 'celery_id'),
        sa.Index('ix_jobs_project_id', 'project_id'),
        sa.Index('ix_jobs_status', 'status')
    )

    # Create outputs table
    op.create_table(
        'outputs',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('project_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('stage', sa.Enum('extract_frames', 'feature_matching', 'sfm', 'mvs', 'depth_estimation', 'scale_anchor', 'coverage', 'export', name='pipelinestage'), nullable=False),
        sa.Column('label', sa.String(256), nullable=False),
        sa.Column('storage_key', sa.String(1024), nullable=False),
        sa.Column('mime_type', sa.String(128), nullable=True),
        sa.Column('size_bytes', sa.Integer(), nullable=True),
        sa.Column('meta', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    op.drop_table('outputs')
    op.drop_table('jobs')
    op.drop_table('anchors')
    op.drop_table('uploads')
    op.drop_table('projects')
