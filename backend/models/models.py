"""
Database models for Photogram.
All IDs are UUIDs generated server-side.
"""

import uuid
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, String, Float, Integer, BigInteger, Boolean,
    DateTime, ForeignKey, JSON, Text, Enum, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


def new_uuid() -> str:
    return str(uuid.uuid4())


# ── Enums ─────────────────────────────────────────────────────────────────────

class ProjectStatus(str, PyEnum):
    CREATED     = "created"
    PROCESSING  = "processing"
    NEEDS_MORE  = "needs_more"   # coverage too low, suggest more shots
    COMPLETE    = "complete"
    FAILED      = "failed"
    # Legacy — kept so existing DB rows with this value don't break ORM reads
    AWAITING_SCALE_CONFIRMATION  = "awaiting_scale_confirmation"


class SceneType(str, PyEnum):
    INDOOR_ROOM = "indoor_room"
    OUTDOOR     = "outdoor"
    OBJECT      = "object"


class JobStatus(str, PyEnum):
    PENDING  = "pending"
    RUNNING  = "running"
    SUCCESS  = "success"
    FAILED   = "failed"
    RETRYING = "retrying"


class PipelineStage(str, PyEnum):
    # Active stages
    EXTRACT_METADATA  = "extract_metadata"
    EXTRACT_FRAMES    = "extract_frames"
    DETECT_ARUCO      = "detect_aruco"
    FEATURE_MATCHING  = "feature_matching"
    SFM               = "sfm"
    MVS               = "mvs"
    SCALE_FROM_ARUCO  = "scale_from_aruco"
    APPLY_SCALE       = "apply_scale"
    COVERAGE          = "coverage"
    EXPORT            = "export"
    GAUSSIAN_SPLATTING = "gaussian_splatting"
    # Legacy — preserved so existing job rows don't break ORM reads
    DEPTH_ESTIMATION  = "depth_estimation"
    SCALE_ANCHOR      = "scale_anchor"
    AUTO_ANCHOR       = "auto_anchor"
    SCENE_UNDERSTANDING   = "scene_understanding"
    DEPTH_FUSION          = "depth_fusion"
    PLANE_FITTING         = "plane_fitting"
    CLOUD_QUALITY_CHECK   = "cloud_quality_check"
    GEOMETRY_HEALING      = "geometry_healing"
    SEMANTIC_LABELING     = "semantic_labeling"
    GEOMETRY_CORRECTION   = "geometry_correction"


# ── Models ────────────────────────────────────────────────────────────────────

class Project(Base):
    __tablename__ = "projects"

    id          = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    name        = Column(String(256), nullable=False)
    description = Column(Text, default="")
    status      = Column(Enum(ProjectStatus, values_callable=lambda x: [e.value for e in x]),
                         default=ProjectStatus.CREATED, index=True)
    # values_callable ensures SQLAlchemy stores/reads lowercase .value strings
    # (e.g. 'indoor_room') rather than enum member names (e.g. 'INDOOR_ROOM').
    scene_type  = Column(
        Enum(SceneType, values_callable=lambda x: [e.value for e in x]),
        default=SceneType.INDOOR_ROOM, nullable=False,
    )
    created_at  = Column(DateTime(timezone=True), server_default=func.now())
    updated_at  = Column(DateTime(timezone=True), onupdate=func.now())

    # Coverage
    coverage_score   = Column(Float, nullable=True)
    suggestions      = Column(JSON, nullable=True)    # re-shoot suggestion list
    coverage_runs    = Column(JSON, nullable=True)    # history: [{ts, score, cloud_key, n_suggestions}]

    # Video metadata (from extract_metadata stage)
    video_metadata   = Column(JSON, nullable=True)    # focal length, GPS, rotation, etc.

    # Camera calibration from a still photo uploaded at project creation.
    # {make, model, fl_mm, fl_35mm, sensor_width_mm, fl_px_at_width,
    #  calibration_width, calibration_height, source: "photo"}
    # fl_px_at_width is the focal length at the calibration image's pixel width.
    # extract_metadata scales it to the video width: fl_px = fl_px_at_width * (W_video / W_calib)
    calibration_data = Column(JSON, nullable=True)

    # ArUco scale (replaces anchor_candidates / VLM anchors)
    aruco_markers          = Column(JSON, nullable=True)    # detections by frame
    confirmed_scale_factor = Column(Float, nullable=True)
    confirmed_scale_source = Column(String(256), nullable=True)  # "aruco" | "manual"
    gravity_up_world       = Column(JSON, nullable=True)          # [x,y,z] world-up vector

    # Outputs
    splat_key        = Column(Text, nullable=True)   # 3DGS .splat for WebGL viewer
    mesh_key         = Column(Text, nullable=True)   # extracted mesh (OBJ/GLB)
    lingbot_cloud_key = Column(Text, nullable=True)  # LingBot-densified fused cloud (PLY)
    lingbot_mesh_key  = Column(Text, nullable=True)  # LingBot-densified fused mesh (OBJ)
    lingbot_enabled   = Column(Boolean, nullable=False, server_default="false")  # per-project densify opt-in

    # Two-pass adaptive pipeline
    pipeline_mode    = Column(String(32), nullable=True)  # 'standard' | 'scout' | 'full'
    scout_calibration = Column(JSON, nullable=True)       # calibration from scout run

    # Per-stage result summary written by emit_stage_complete — keyed by stage name
    pipeline_results = Column(JSON, nullable=True)

    # Legacy columns — preserved so old rows and old code don't crash
    anchor_reminder  = Column(Text, nullable=True)
    anchor_candidates = Column(JSON, nullable=True)
    quality_issues   = Column(JSON, nullable=True)
    object_labels    = Column(JSON, nullable=True)
    confidence_map_key = Column(Text, nullable=True)
    clarifications   = Column(JSON, nullable=True)
    scene_analysis   = Column(JSON, nullable=True)

    uploads   = relationship("Upload",   back_populates="project", cascade="all, delete-orphan")
    anchors   = relationship("Anchor",   back_populates="project", cascade="all, delete-orphan")
    jobs      = relationship("Job",      back_populates="project", cascade="all, delete-orphan")
    outputs   = relationship("Output",   back_populates="project", cascade="all, delete-orphan")


class Upload(Base):
    """A video or image set uploaded to a project."""
    __tablename__ = "uploads"

    id          = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    project_id  = Column(UUID(as_uuid=False), ForeignKey("projects.id"), nullable=False, index=True)
    storage_key = Column(String(1024), nullable=False)   # e.g. proj_xxx/uploads/video.mp4
    filename    = Column(String(512), nullable=False)
    mime_type   = Column(String(128), nullable=False)
    size_bytes  = Column(BigInteger, nullable=True)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())

    project = relationship("Project", back_populates="uploads")


class Anchor(Base):
    """
    A reference object used to establish real-world scale.
    Stores the SAM 2 segmentation mask and user-specified dimensions.
    """
    __tablename__ = "anchors"

    id             = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    project_id     = Column(UUID(as_uuid=False), ForeignKey("projects.id"), nullable=False, index=True)
    name           = Column(String(256), nullable=False)        # e.g. "Credit card"
    storage_key    = Column(String(1024), nullable=False)        # anchor reference image
    mask_key       = Column(String(1024), nullable=True)         # SAM 2 mask output
    # Real-world dimensions in millimetres
    width_mm       = Column(Float, nullable=True)
    height_mm      = Column(Float, nullable=True)
    depth_mm       = Column(Float, nullable=True)
    # Feature descriptors stored as JSON for matching in scene frames
    feature_data   = Column(JSON, nullable=True)
    created_at     = Column(DateTime(timezone=True), server_default=func.now())

    project = relationship("Project", back_populates="anchors")


class Job(Base):
    """One Celery pipeline task run."""
    __tablename__ = "jobs"

    id          = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    project_id  = Column(UUID(as_uuid=False), ForeignKey("projects.id"), nullable=False, index=True)
    celery_id   = Column(String(256), nullable=True, index=True)
    stage       = Column(Enum(PipelineStage, values_callable=lambda x: [e.value for e in x]), nullable=False)
    status      = Column(Enum(JobStatus, values_callable=lambda x: [e.value for e in x]), default=JobStatus.PENDING, index=True)
    progress    = Column(Float, default=0.0)        # 0.0–1.0
    message     = Column(Text, default="")
    error       = Column(Text, nullable=True)
    meta        = Column(JSON, default=dict)         # stage-specific metadata
    created_at  = Column(DateTime(timezone=True), server_default=func.now())
    updated_at  = Column(DateTime(timezone=True), onupdate=func.now())

    project = relationship("Project", back_populates="jobs")


class Output(Base):
    """A file produced by the pipeline (point cloud, mesh, coverage map, etc.)."""
    __tablename__ = "outputs"

    id          = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    project_id  = Column(UUID(as_uuid=False), ForeignKey("projects.id"), nullable=False, index=True)
    stage       = Column(Enum(PipelineStage, values_callable=lambda x: [e.value for e in x]), nullable=False)
    label       = Column(String(256), nullable=False)   # e.g. "Dense Point Cloud"
    storage_key = Column(String(1024), nullable=False)
    mime_type   = Column(String(128), nullable=True)
    size_bytes  = Column(BigInteger, nullable=True)
    meta        = Column(JSON, default=dict)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())

    project = relationship("Project", back_populates="outputs")
