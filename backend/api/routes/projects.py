"""Projects CRUD + video upload endpoints."""

import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import Response
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from backend.api.deps import get_db
from backend.core.storage import get_storage
from backend.core.config import settings
from backend.models.models import Project, Upload, Job, ProjectStatus, SceneType

logger = logging.getLogger(__name__)
router = APIRouter()


class ProjectCreate(BaseModel):
    name: str
    description: str = ""
    scene_type: str = "indoor_room"   # "indoor_room" | "outdoor" | "object"
    lingbot_enabled: bool = False     # opt-in to LingBot depth-fusion densification
    metricanything_enabled: bool = False  # opt-in to MetricAnything depth-fusion densification
    hvac_mode: bool = False           # opt-in to HVAC wall-mount placement (indoor_room only)
    marker_type: str = "aruco"        # "aruco" | "grid" — scale fiducial used in the scan


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class UploadListItem(BaseModel):
    id: str
    filename: str
    storage_key: str
    mime_type: str
    size_bytes: int
    uploaded_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str
    status: str
    scene_type: Optional[str] = None
    coverage_score: float | None = None
    confirmed_scale_factor: float | None = None
    confirmed_scale_source: str | None = None
    suggestions: list | None = None
    coverage_runs: list | None = None
    splat_key: str | None = None
    mesh_key: str | None = None
    lingbot_cloud_key: str | None = None
    lingbot_mesh_key: str | None = None
    lingbot_enabled: bool = False
    metricanything_cloud_key: str | None = None
    metricanything_mesh_key: str | None = None
    metricanything_enabled: bool = False
    hvac_mode: bool = False
    hvac_placement: dict | None = None
    hvac_segmentation: dict | None = None
    marker_type: str = "aruco"
    scan_source: str = "video"
    dimensions: dict | None = None
    aruco_markers: dict | None = None
    gravity_up_world: list | None = None
    calibration_data: dict | None = None
    pipeline_mode: str | None = None
    scout_calibration: dict | None = None
    created_at: Optional[datetime] = None
    pipeline_results: dict | None = None

    class Config:
        from_attributes = True


@router.post("/", response_model=ProjectResponse, status_code=201)
async def create_project(body: ProjectCreate, db: AsyncSession = Depends(get_db)):
    # Validate scene_type
    try:
        st = SceneType(body.scene_type)
    except ValueError:
        raise HTTPException(400, f"Invalid scene_type '{body.scene_type}'. "
                               f"Valid values: indoor_room, outdoor, object")
    if body.marker_type not in ("aruco", "grid"):
        raise HTTPException(400, f"Invalid marker_type '{body.marker_type}'. Valid values: aruco, grid")

    project = Project(
        name=body.name,
        description=body.description,
        scene_type=st,
        status=ProjectStatus.CREATED,
        lingbot_enabled=body.lingbot_enabled,
        metricanything_enabled=body.metricanything_enabled,
        hvac_mode=body.hvac_mode,
        marker_type=body.marker_type,
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)
    return project


@router.get("/", response_model=list[ProjectResponse])
async def list_projects(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Project))
    projects = result.scalars().all()
    return projects


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")
    return project


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(project_id: str, body: ProjectUpdate, db: AsyncSession = Depends(get_db)):
    """Rename or update the description of a project."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")
    if body.name is not None:
        if not body.name.strip():
            raise HTTPException(400, "Name cannot be empty")
        project.name = body.name.strip()
    if body.description is not None:
        project.description = body.description
    await db.commit()
    await db.refresh(project)
    return project


@router.get("/{project_id}/uploads", response_model=list[UploadListItem])
async def list_project_uploads(project_id: str, db: AsyncSession = Depends(get_db)):
    """List all uploads associated with a project."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    if not result.scalar_one_or_none():
        raise HTTPException(404, "Project not found")
    uploads_result = await db.execute(
        select(Upload).where(Upload.project_id == project_id)
    )
    return uploads_result.scalars().all()


@router.get("/{project_id}/jobs")
async def list_project_jobs(project_id: str, db: AsyncSession = Depends(get_db)):
    """Return all pipeline jobs for a project, ordered by creation time."""
    exists = await db.execute(select(Project).where(Project.id == project_id))
    if not exists.scalar_one_or_none():
        raise HTTPException(404, "Project not found")
    jobs_result = await db.execute(
        select(Job).where(Job.project_id == project_id).order_by(Job.created_at)
    )
    jobs = jobs_result.scalars().all()
    out = []
    for j in jobs:
        elapsed = None
        if j.created_at and j.updated_at:
            elapsed = (j.updated_at - j.created_at).total_seconds()
        out.append({
            "id": str(j.id),
            "stage": j.stage.value if j.stage else None,
            "status": j.status.value if j.status else None,
            "progress": j.progress,
            "message": j.message or "",
            "error": j.error,
            "created_at": j.created_at.isoformat() if j.created_at else None,
            "updated_at": j.updated_at.isoformat() if j.updated_at else None,
            "elapsed_seconds": elapsed,
        })
    return out


@router.post("/{project_id}/uploads")
async def upload_video(
    project_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Accept a video upload, stream it to storage.
    Returns the upload ID and storage key.
    """
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")

    storage  = get_storage()
    dest_key = f"{project_id}/uploads/{file.filename}"

    import uuid as _uuid
    tmp_path = settings.TEMP_DIR / f"upload_{project_id}_{_uuid.uuid4().hex[:8]}"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(tmp_path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                out.write(chunk)
        await storage.upload(tmp_path, dest_key)
    finally:
        tmp_path.unlink(missing_ok=True)

    upload = Upload(
        project_id=project_id,
        storage_key=dest_key,
        filename=file.filename,
        mime_type=file.content_type or "application/octet-stream",
        size_bytes=file.size,
    )
    db.add(upload)
    await db.commit()
    await db.refresh(upload)

    return {
        "upload_id":   upload.id,
        "storage_key": dest_key,
        "filename":    file.filename,
        "size_bytes":  file.size,
    }


@router.post("/{project_id}/launch")
async def launch_pipeline(
    project_id: str,
    upload_id: str,
    mode: str = "standard",   # "standard" | "scout"
    exclude_marker_ids: str = "",   # comma-separated ArUco IDs to ignore in scale derivation
    resize_preset: str = "native",  # "native" | "2k" | "4k" | "8k"
    scan_source: str = "video",     # "video" | "lidar_ply"
    lidar_scale_factor: float = 1.0,  # applied only when scan_source="lidar_ply"
    ground_truth_length_m: Optional[float] = None,
    ground_truth_breadth_m: Optional[float] = None,
    ground_truth_height_m: Optional[float] = None,
    db: AsyncSession = Depends(get_db),
):
    """Kick off the reconstruction pipeline.

    mode="standard" — full pipeline in one pass (default)
    mode="scout"    — fast calibration pass (~15 min) that auto-triggers a
                      calibrated full pass on completion
    exclude_marker_ids — comma-separated ArUco IDs to exclude from scale
                         derivation, e.g. "7" or "7,12"
    resize_preset — downscale extracted frames to this max long-edge resolution
                    before SfM/MVS ("native" = no resize, "2k", "4k", "8k").
                    Lower presets run faster but reduce ArUco detection range
                    and dense-cloud detail.
    scan_source — "video" (default) runs the full SfM/MVS chain. "lidar_ply"
                  treats `upload_id` as a pre-built .ply point cloud and runs
                  ingest → refine → export; mode/resize_preset/
                  exclude_marker_ids are ignored.
    lidar_scale_factor — multiply the cloud by this before treating it as
                         metric (e.g. 0.001 for a scanner exporting mm).
    ground_truth_length_m / _breadth_m / _height_m — optional tape-measure
        reference; export reports predicted-vs-truth error % for each given.
    """
    ground_truth_dimensions = {
        k: v for k, v in (("length_m", ground_truth_length_m),
                          ("breadth_m", ground_truth_breadth_m),
                          ("height_m", ground_truth_height_m)) if v is not None
    }

    if scan_source not in ("video", "lidar_ply"):
        raise HTTPException(400, "scan_source must be one of: video, lidar_ply")

    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")

    upload_result = await db.execute(select(Upload).where(Upload.id == upload_id))
    upload = upload_result.scalar_one_or_none()
    if not upload:
        raise HTTPException(404, "Upload not found")

    if scan_source == "lidar_ply":
        from backend.workers.tasks import launch_lidar_pipeline as _launch_lidar
        try:
            scene_type = project.scene_type.value if project.scene_type else "indoor_room"
            task = _launch_lidar(project_id, upload.storage_key, scene_type=scene_type,
                                 lidar_scale_factor=lidar_scale_factor,
                                 ground_truth_dimensions=ground_truth_dimensions or None)
            if not task or not task.id:
                raise HTTPException(500, "Failed to start pipeline")

            project.status = ProjectStatus.PROCESSING
            project.pipeline_mode = "standard"
            project.scan_source = "lidar_ply"
            await db.commit()

            return {"task_id": task.id, "status": "launched", "mode": "standard", "scan_source": "lidar_ply"}
        except HTTPException:
            raise
        except Exception as e:
            logger.warning("launch_lidar_pipeline error: %s", e)
            raise HTTPException(500, f"Failed to launch LiDAR pipeline: {str(e)}")

    # Collect all other uploads so all media is processed in extract_frames
    all_uploads_result = await db.execute(
        select(Upload).where(Upload.project_id == project_id)
    )
    all_uploads = all_uploads_result.scalars().all()
    extra_keys = [u.storage_key for u in all_uploads if u.id != upload_id]

    from backend.workers.tasks import launch_pipeline as _launch_std
    from backend.workers.tasks import launch_scout_pipeline as _launch_scout

    exclude_ids = None
    if exclude_marker_ids.strip():
        try:
            exclude_ids = [int(x.strip()) for x in exclude_marker_ids.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(400, "exclude_marker_ids must be comma-separated integers, e.g. '7,12'")

    if resize_preset not in ("native", "2k", "4k", "8k"):
        raise HTTPException(400, "resize_preset must be one of: native, 2k, 4k, 8k")

    # Video chain picks ground truth up from Redis in extract_metadata
    import redis as _redis, json as _json
    _r = _redis.from_url(settings.REDIS_URL)
    if ground_truth_dimensions:
        _r.set(f"project:{project_id}:ground_truth_dimensions",
               _json.dumps(ground_truth_dimensions), ex=86400)
    else:
        _r.delete(f"project:{project_id}:ground_truth_dimensions")

    try:
        scene_type = project.scene_type.value if project.scene_type else "indoor_room"
        if mode == "scout":
            task = _launch_scout(project_id, upload.storage_key, extra_keys=extra_keys)
        else:
            task = _launch_std(project_id, upload.storage_key, extra_keys=extra_keys,
                                exclude_marker_ids=exclude_ids, resize_preset=resize_preset,
                                scene_type=scene_type, lingbot_enabled=bool(project.lingbot_enabled),
                                metricanything_enabled=bool(project.metricanything_enabled),
                                hvac_mode=bool(project.hvac_mode))

        if not task or not task.id:
            raise HTTPException(500, "Failed to start pipeline")

        project.status = ProjectStatus.PROCESSING
        project.pipeline_mode = mode
        project.scan_source = "video"
        await db.commit()

        return {"task_id": task.id, "status": "launched", "mode": mode}
    except Exception as e:
        logger.warning("launch_pipeline error: %s", e)
        raise HTTPException(500, f"Failed to launch pipeline: {str(e)}")


@router.post("/{project_id}/launch_supplemental")
async def launch_supplemental(
    project_id: str,
    upload_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Launch a supplemental reconstruction pass using a new video upload.
    Project must be in 'needs_more' or 'complete' status.
    """
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")

    if project.status not in (ProjectStatus.NEEDS_MORE, ProjectStatus.COMPLETE):
        raise HTTPException(409, f"Project status '{project.status}' does not allow supplemental upload")

    upload_result = await db.execute(select(Upload).where(Upload.id == upload_id))
    upload = upload_result.scalar_one_or_none()
    if not upload:
        raise HTTPException(404, "Upload not found")

    from backend.workers.tasks import launch_supplemental_pipeline as _launch

    try:
        task = _launch(project_id, upload.storage_key)
        project.status = ProjectStatus.PROCESSING
        await db.commit()
        return {"task_id": task.id, "status": "launched"}
    except Exception as e:
        raise HTTPException(500, f"Failed to launch supplemental pipeline: {e}")


@router.post("/{project_id}/calibration_photo")
async def upload_calibration_photo(
    project_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Accept a still image from the user's phone, extract focal length from EXIF,
    and store calibration_data on the project.

    Returns the extracted calibration dict so the frontend can confirm the device
    and focal length before launching the pipeline.

    The image is not persisted after extraction — only the EXIF-derived numbers
    are stored in projects.calibration_data.
    """
    try:
        result = await db.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()
    except Exception:
        project = None
    if not project:
        raise HTTPException(404, "Project not found")

    import tempfile, os as _os
    from backend.workers.pipeline.calibration import extract_calibration_from_image

    suffix = Path(file.filename or "photo.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            while chunk := await file.read(1024 * 1024):
                tmp.write(chunk)
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(500, f"Failed to read upload: {exc}")

    try:
        calibration = extract_calibration_from_image(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if calibration is None:
        raise HTTPException(422, "No focal length data found in image EXIF. "
                                 "Try a JPEG taken directly by the camera app "
                                 "(not a screenshot or edited photo).")

    project.calibration_data = calibration
    await db.commit()

    return calibration


@router.post("/{project_id}/cancel")
async def cancel_pipeline(project_id: str, db: AsyncSession = Depends(get_db)):
    """Cancel the active pipeline run for a project."""
    import redis.asyncio as aioredis
    from backend.workers.tasks import celery_app

    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")
    if project.status != ProjectStatus.PROCESSING:
        raise HTTPException(400, f"Project is not running (status: {project.status.value})")

    r = aioredis.from_url(settings.REDIS_URL)
    try:
        task_id = await r.get(f"project:{project_id}:active_task")
        if task_id:
            celery_app.control.revoke(task_id.decode(), terminate=True, signal="SIGKILL")
            await r.delete(f"project:{project_id}:active_task")
    finally:
        await r.close()

    project.status = ProjectStatus.FAILED
    await db.commit()
    return {"status": "cancelled"}


# ── Suggestion detail endpoint ─────────────────────────────────────────────

@router.get("/{project_id}/suggestions_detail")
async def get_suggestions_detail(project_id: str, db: AsyncSession = Depends(get_db)):
    """
    For each re-shoot suggestion, find the nearest registered camera frame
    so the frontend can show a reference thumbnail.
    """
    import json as _json
    import numpy as np

    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")

    scale_factor = project.confirmed_scale_factor or 1.0
    storage      = get_storage()
    cameras_key  = f"{project_id}/sfm/cameras.json"
    import tempfile, os
    tmp = tempfile.mkdtemp()
    try:
        cameras_local = os.path.join(tmp, "cameras.json")
        try:
            await storage.download(cameras_key, Path(cameras_local))
            with open(cameras_local) as f:
                cameras_json = _json.load(f)
        except Exception:
            return {"suggestions": [], "error": "cameras.json not available"}

        images = cameras_json.get("images", [])
        cam_positions, cam_names = [], []
        for img in images:
            cfw = img.get("cam_from_world", {})
            mat = cfw.get("matrix_3x4") if isinstance(cfw, dict) else cfw
            if mat is None:
                continue
            m = np.array(mat, dtype=np.float64)
            R, t = m[:, :3], m[:, 3]
            cam_positions.append(-R.T @ t)
            cam_names.append(img["name"])

        if not cam_positions:
            return {"suggestions": [], "error": "no cameras found"}

        cam_pos_arr = np.array(cam_positions)
        suggestions_raw = project.suggestions or []
        if not suggestions_raw:
            return {"suggestions": [], "note": "no suggestions stored"}

        enriched = []
        for s in suggestions_raw:
            pos     = s.get("position", [0, 0, 0])
            pos_sfm = np.array(pos) / scale_factor
            dists   = np.linalg.norm(cam_pos_arr - pos_sfm, axis=1)
            ni      = int(np.argmin(dists))
            enriched.append({
                **s,
                "nearest_frame_key":  f"{project_id}/frames/{cam_names[ni]}",
                "nearest_frame_name": cam_names[ni],
            })

        return {"suggestions": enriched, "scale_factor": scale_factor}
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


# ── Bulk / delete helpers ──────────────────────────────────────────────────

def _raw_delete_projects(ids: list[str]) -> int:
    """Revoke active Celery tasks then delete projects + all child rows via raw SQL."""
    import psycopg2
    import redis as redis_sync
    if not ids:
        return 0

    # Revoke any running Celery tasks before touching the DB
    try:
        r = redis_sync.from_url(settings.REDIS_URL)
        for pid in ids:
            task_id = r.get(f"project:{pid}:active_task")
            if task_id:
                celery_app.control.revoke(task_id.decode(), terminate=True, signal="SIGKILL")
                r.delete(f"project:{pid}:active_task")
        r.close()
    except Exception:
        pass  # best-effort — don't let Redis failure block the delete

    conn = psycopg2.connect(settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgres://"))
    cur  = conn.cursor()
    ph   = ",".join(["%s"] * len(ids))
    for table in ("jobs", "uploads", "outputs", "anchors"):
        try:
            cur.execute(f"DELETE FROM {table} WHERE project_id IN ({ph})", ids)
        except Exception:
            pass
    cur.execute(f"DELETE FROM projects WHERE id IN ({ph})", ids)
    deleted = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    return deleted


@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a single project and all its associated data."""
    _raw_delete_projects([project_id])


class BulkDeleteRequest(BaseModel):
    ids: List[str]


@router.post("/bulk-delete", status_code=200)
async def bulk_delete_projects(body: BulkDeleteRequest):
    """Delete multiple projects by ID."""
    deleted = _raw_delete_projects(body.ids)
    return {"deleted": deleted}


@router.delete("/", status_code=200)
async def delete_failed_projects(db: AsyncSession = Depends(get_db)):
    """Delete all projects in failed status."""
    result = await db.execute(
        select(Project.id).where(Project.status == ProjectStatus.FAILED)
    )
    ids     = [str(row[0]) for row in result.all()]
    deleted = _raw_delete_projects(ids)
    return {"deleted": deleted}


# ── Reprocess from checkpoint ─────────────────────────────────────────────────

_HVAC_STAGES = ["wall_plane_detection", "detect_hvac_fixtures", "locate_rucklauf", "hvac_placement"]

REPROCESS_CHAIN: dict[str, list[str]] = {
    "scale_from_aruco": [
        "scale_from_aruco", "apply_known_scale", "fill_planes",
        "refine_cloud", "lingbot_fusion", "metricanything_fusion", *_HVAC_STAGES, "coverage", "export",
    ],
    "fill_planes": ["fill_planes", "refine_cloud", "lingbot_fusion", "metricanything_fusion",
                    *_HVAC_STAGES, "coverage", "export"],
    "refine_cloud": ["refine_cloud", "lingbot_fusion", "metricanything_fusion", *_HVAC_STAGES,
                     "coverage", "export"],
    "lingbot_fusion": ["lingbot_fusion", "metricanything_fusion", *_HVAC_STAGES, "coverage", "export"],
    "metricanything_fusion": ["metricanything_fusion", *_HVAC_STAGES, "coverage", "export"],
    "wall_plane_detection": [*_HVAC_STAGES, "coverage", "export"],
    "coverage":     ["coverage", "export"],
    "export":       ["export"],
}
# Note: the densifier stages are filtered out at dispatch unless enabled for the
# project (and their ENABLE_* env flag, the fleet-wide master switch).


@router.post("/{project_id}/reprocess")
async def reprocess(
    project_id: str,
    from_stage: str = "scale_from_aruco",
    exclude_marker_ids: str = "",   # comma-separated IDs to exclude, e.g. "7,12"
    ground_truth_length_m: Optional[float] = None,
    ground_truth_breadth_m: Optional[float] = None,
    ground_truth_height_m: Optional[float] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Re-dispatch downstream pipeline stages using the saved checkpoint.

    Reads the reprocess checkpoint written after detect_aruco_sfm and
    dispatches a Celery chain starting at from_stage.

    Valid from_stage values: scale_from_aruco, fill_planes, refine_cloud,
    coverage, export.

    exclude_marker_ids: comma-separated ArUco IDs to ignore in scale derivation
      (e.g. "7" to exclude a duplicate marker).
    """
    import redis as _redis, json as _json, psycopg2
    from celery import chain as _chain
    from backend.workers.tasks import (
        scale_from_aruco_task, apply_known_scale_task,
        fill_planes_task, refine_cloud_task, lingbot_fusion_task,
        metricanything_fusion_task, wall_plane_detection_task, detect_hvac_fixtures_task,
        locate_rucklauf_task, hvac_placement_task, analyze_coverage, export_outputs,
        _create_job_for_task,
    )

    if from_stage not in REPROCESS_CHAIN:
        raise HTTPException(400, f"Invalid from_stage '{from_stage}'. "
                            f"Valid: {list(REPROCESS_CHAIN)}")

    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")

    # Load checkpoint
    r = _redis.from_url(settings.REDIS_URL)
    raw = r.get(f"project:{project_id}:reprocess_checkpoint")
    if not raw:
        raise HTTPException(409, "No reprocess checkpoint found for this project. "
                            "The pipeline must have completed detect_aruco_sfm at least once.")

    prev_result = _json.loads(raw)

    if from_stage == "scale_from_aruco":
        # Clear stale keys that will be recomputed
        stale = ["confirmed_scale_factor", "confirmed_scale_source", "gravity_up_world",
                 "scale_diagnostics", "fitted_planes", "scaled_cloud_key",
                 "layout_cloud_key", "dense_point_count", "refinement"]
        for k in stale:
            prev_result.pop(k, None)
    else:
        # The checkpoint is written once, right after detect_aruco_sfm, so it
        # predates scale_from_aruco. Backfill scale/gravity from the project
        # row (scale_from_aruco persists them there) — without this, later
        # stages run with no scale (wrong voxel size, no metric dimensions).
        prev_result.setdefault("confirmed_scale_factor", project.confirmed_scale_factor)
        prev_result.setdefault("confirmed_scale_source", project.confirmed_scale_source)
        prev_result.setdefault("gravity_up_world", project.gravity_up_world)

        # Same for the cloud: the checkpoint's dense_cloud_key is the raw MVS
        # output. Point it at the artifact of the stage just BEFORE from_stage
        # (never from_stage's own output, which may be a prior broken run's).
        layout  = f"{project_id}/mvs/dense_layout.ply"
        refined = f"{project_id}/mvs/dense_refined.ply"
        scaled  = f"{project_id}/clouds/scaled.ply"
        upstream_candidates = {
            "fill_planes":    [scaled],
            "refine_cloud":   [layout, scaled],
            "lingbot_fusion": [refined, layout, scaled],
            "metricanything_fusion": [refined, layout, scaled],
            "wall_plane_detection":  [refined, layout, scaled],
            "coverage":       [refined, layout, scaled],
            "export":         [refined, layout, scaled],
        }
        storage = get_storage()
        for candidate_key in upstream_candidates.get(from_stage, []):
            if storage.exists(candidate_key):
                prev_result["dense_cloud_key"] = candidate_key
                prev_result["scaled_cloud_key"] = candidate_key
                break

    # Inject excluded marker IDs so scale_from_aruco_task can read them
    if exclude_marker_ids.strip():
        try:
            ids = [int(x.strip()) for x in exclude_marker_ids.split(",") if x.strip()]
            prev_result["exclude_marker_ids"] = ids
        except ValueError:
            raise HTTPException(400, "exclude_marker_ids must be comma-separated integers, e.g. '7,12'")
    else:
        prev_result.pop("exclude_marker_ids", None)

    # Ground truth for the export-stage error check — from_stage=export on a
    # completed project is the cheapest way to test a tape-measure comparison.
    ground_truth_dimensions = {
        k: v for k, v in (("length_m", ground_truth_length_m),
                          ("breadth_m", ground_truth_breadth_m),
                          ("height_m", ground_truth_height_m)) if v is not None
    }
    if ground_truth_dimensions:
        prev_result["ground_truth_dimensions"] = ground_truth_dimensions
    else:
        prev_result.pop("ground_truth_dimensions", None)

    # Build task list
    task_map = {
        "scale_from_aruco": scale_from_aruco_task,
        "apply_known_scale": apply_known_scale_task,
        "fill_planes":       fill_planes_task,
        "refine_cloud":      refine_cloud_task,
        "lingbot_fusion":    lingbot_fusion_task,
        "metricanything_fusion": metricanything_fusion_task,
        "wall_plane_detection": wall_plane_detection_task,
        "detect_hvac_fixtures": detect_hvac_fixtures_task,
        "locate_rucklauf":   locate_rucklauf_task,
        "hvac_placement":    hvac_placement_task,
        "coverage":          analyze_coverage,
        "export":            export_outputs,
    }
    # Densifiers only run when enabled for this project (env flag = master kill-switch).
    _enabled = {
        "lingbot_fusion": settings.ENABLE_LINGBOT_FUSION and bool(project.lingbot_enabled),
        "metricanything_fusion": settings.ENABLE_METRICANYTHING_FUSION and bool(project.metricanything_enabled),
        **dict.fromkeys(_HVAC_STAGES, settings.ENABLE_HVAC_PLACEMENT and bool(project.hvac_mode)
                        and (project.scene_type is None or project.scene_type.value == "indoor_room")),
    }
    stage_seq = [s for s in REPROCESS_CHAIN[from_stage] if _enabled.get(s, True)]
    tasks = [task_map[s].s(project_id) for s in stage_seq]

    # Mark processing
    project.status = ProjectStatus.PROCESSING
    await db.commit()

    result_chain = _chain(*tasks).apply_async(args=[prev_result])
    r.set(f"project:{project_id}:active_task", result_chain.id, ex=86400 * 7)
    _create_job_for_task(project_id, from_stage.upper(), result_chain.id)

    return {
        "task_id":    result_chain.id,
        "from_stage": from_stage,
        "stages":     REPROCESS_CHAIN[from_stage],
    }
