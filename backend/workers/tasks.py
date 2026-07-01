"""
Celery app + pipeline task definitions.

Pipeline (single continuous chain — no user gate):
  extract_metadata → extract_frames → detect_aruco (FAIL FAST) →
  feature_matching → sfm → mvs → scale_from_aruco → apply_scale →
  coverage → export

Scale is derived automatically from ArUco markers.  The pipeline fails
immediately at detect_aruco if no markers are found so expensive COLMAP
stages don't run on unscannable footage.

Progress is pushed via Redis pub/sub → WebSocket endpoint picks it up.
Job status is persisted to the database.
"""

import json
import math as _math
import shutil
import asyncio
import threading
import time as _time
import logging
from pathlib import Path
from celery import Celery
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from backend.core.config import settings
from backend.core.storage import get_storage
from backend.models.models import Job, JobStatus, PipelineStage, ProjectStatus

logger = logging.getLogger(__name__)


def _json_safe(obj):
    """Recursively make a dict/list safe for json.dumps → PostgreSQL JSON column.

    Handles two failure modes:
      1. numpy scalar/array types  → convert via .tolist() / int() / float()
      2. NaN / Infinity floats     → replace with None (PostgreSQL rejects NaN in JSON)
    """
    import numpy as _np
    if isinstance(obj, _np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, _np.integer):
        return int(obj)
    if isinstance(obj, _np.floating):
        v = float(obj)
        return None if not _math.isfinite(v) else v
    if isinstance(obj, float):
        return None if not _math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# ── Redis TTL constants ───────────────────────────────────────────────────────
_TTL_DAY   = 86_400        # 24 h — progress events, last_progress cache
_TTL_WEEK  = 86_400 * 7    # 7 d  — active task ID, reprocess checkpoints
_TTL_MONTH = 86_400 * 30   # 30 d — reprocess checkpoint (kept long for re-use)

celery_app = Celery(
    "photogram",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    task_track_started=True,
    worker_prefetch_multiplier=1,  # one heavy task at a time per worker
    task_routes={
        "pipeline.feature_matching":  {"queue": "gpu"},
        "pipeline.sfm":               {"queue": "gpu"},
        "pipeline.mvs":               {"queue": "gpu"},
        "pipeline.gaussian_splatting": {"queue": "gpu"},
        "pipeline.lingbot_fusion":    {"queue": "gpu"},
    },
)

# Database session for workers
_engine = create_async_engine(settings.DATABASE_URL, future=True)
_async_session_maker = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def job_tmp(job_id: str) -> Path:
    p = settings.TEMP_DIR / job_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _redis_client():
    import redis
    return redis.from_url(settings.REDIS_URL)


def publish_progress(project_id: str, stage: str, progress: float, message: str = ""):
    """Push a progress event to Redis → WebSocket relay."""
    r = _redis_client()
    payload = json.dumps({"type": "progress", "stage": stage, "progress": progress,
                          "message": message, "ts": _time.time()})
    r.publish(f"project:{project_id}:progress", payload)
    r.set(f"project:{project_id}:last_progress", payload, ex=_TTL_DAY)


def publish_event(project_id: str, event: dict):
    """Push any typed event (heartbeat, stage_complete, …) to the same channel."""
    event.setdefault("ts", _time.time())
    r = _redis_client()
    payload = json.dumps(event)
    r.publish(f"project:{project_id}:progress", payload)
    if event.get("type") != "heartbeat":
        r.set(f"project:{project_id}:last_progress", payload, ex=_TTL_DAY)


def _gpu_util() -> dict | None:
    """Return basic GPU stats, or None if unavailable."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=3, text=True,
        ).strip()
        util, used, total = out.split(",")
        return {"util_pct": int(util.strip()), "vram_used_mb": int(used.strip()),
                "vram_total_mb": int(total.strip())}
    except Exception:
        return None


def start_heartbeat(project_id: str, stage: str, interval: int = 15) -> threading.Event:
    """
    Start a background thread that publishes a heartbeat event every `interval`
    seconds.  Returns a stop_event; call stop_event.set() to halt it.
    """
    stop = threading.Event()
    start_ts = _time.time()

    def _run():
        while not stop.wait(interval):
            elapsed = _time.time() - start_ts
            elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s" if elapsed >= 60 else f"{int(elapsed)}s"
            gpu = _gpu_util()
            publish_event(project_id, {
                "type": "heartbeat",
                "stage": stage,
                "elapsed": elapsed_str,
                "gpu": gpu,
            })
            # Refresh last_progress if stale (> 45s)
            r = _redis_client()
            raw = r.get(f"project:{project_id}:last_progress")
            if raw:
                try:
                    last = json.loads(raw)
                    age  = _time.time() - last.get("ts", 0)
                    if age > 45:
                        import re as _re
                        base_msg = _re.sub(r'\s*\(elapsed:[^)]*\)', '', last.get("message", "Running…")).rstrip()
                        payload = json.dumps({
                            "type": "progress",
                            "stage": stage,
                            "progress": last.get("progress", 0),
                            "message": f"{base_msg} (elapsed: {elapsed_str})",
                            "ts": _time.time(),
                        })
                        r.set(f"project:{project_id}:last_progress", payload, ex=_TTL_DAY)
                except Exception:
                    pass

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return stop


def emit_stage_complete(project_id: str, stage: str, start_ts: float,
                        summary: str, metrics: dict | None = None,
                        warnings: list[str] | None = None,
                        errors: list[str] | None = None):
    """Publish a structured stage-completion event and persist metrics to the DB."""
    duration_s = round(_time.time() - start_ts, 1)
    publish_event(project_id, {
        "type": "stage_complete",
        "stage": stage,
        "duration_s": duration_s,
        "summary": summary,
        "metrics": metrics or {},
        "warnings": warnings or [],
        "errors": errors or [],
    })
    # Persist per-stage result to projects.pipeline_results for page-reload display
    try:
        import psycopg2, json as _json
        _conn = psycopg2.connect(settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgres://"))
        _cur  = _conn.cursor()
        finished_at = _time.time()
        stage_data = {
            "_duration_s": duration_s,
            "_message": summary,
            "_started_at": finished_at - duration_s,
            "_finished_at": finished_at,
        }
        if metrics:
            stage_data.update(metrics)
        if warnings:
            stage_data["_warnings"] = warnings
        # Merge into existing pipeline_results JSON using Postgres json operations
        _cur.execute(
            """
            UPDATE projects
            SET pipeline_results = COALESCE(pipeline_results, '{}'::json)::jsonb
                                   || jsonb_build_object(%s, %s::jsonb)
            WHERE id = %s
            """,
            [stage, _json.dumps(stage_data), project_id],
        )
        _conn.commit()
        _cur.close()
        _conn.close()
    except Exception as _e:
        logger.warning("emit_stage_complete: failed to persist stage %s result: %s", stage, _e)


def make_progress_cb(project_id: str, stage: str, job_id: str):
    """Return the standard dual-channel progress callback used by every task.

    Publishes to the Redis WebSocket channel AND updates the jobs table so
    both real-time viewers and the DB stay in sync.
    """
    return lambda p, m: [
        publish_progress(project_id, stage, p, m),
        update_job_status(job_id, JobStatus.RUNNING, p, m),
    ]


def update_job_status(job_id: str, status: JobStatus, progress: float = None,
                      message: str = "", error: str = None):
    """Update job status in the database using synchronous SQL."""
    try:
        import psycopg2

        conn = psycopg2.connect(settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgres://"))
        cur  = conn.cursor()

        updates = ["status = %s::jobstatus"]
        values  = [status.value]
        if progress is not None:
            updates.append("progress = %s")
            values.append(progress)
        if message:
            updates.append("message = %s")
            values.append(message)
        if error:
            updates.append("error = %s")
            values.append(error)

        updates.append("updated_at = NOW()")
        values.append(job_id)
        sql = f"UPDATE jobs SET {', '.join(updates)} WHERE celery_id = %s"
        cur.execute(sql, values)
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning("Error updating job %s: %s", job_id, e)


# Backward compatible alias
sync_update_job_status = update_job_status


# ── Stage 0: Extract Video Metadata ──────────────────────────────────────────

@celery_app.task(bind=True, name="pipeline.extract_metadata")
def extract_metadata_task(self, project_id: str, storage_key: str,
                          extra_keys: list | None = None) -> dict:
    """
    Analyse the primary video with ffprobe + exiftool.
    Stores focal length, GPS, rotation, and stable frame timestamps in the DB.
    Returns {storage_key, all_storage_keys, video_metadata}.
    extra_keys: additional video/still storage keys to process in extract_frames.
    """
    from backend.workers.pipeline.extract_metadata import run_extract_metadata

    tmp    = job_tmp(self.request.id)
    job_id = self.request.id
    t0     = _time.time()
    hb     = start_heartbeat(project_id, "extract_metadata")
    try:
        update_job_status(job_id, JobStatus.RUNNING, 0.0, "Extracting video metadata…")
        result = asyncio.run(
            run_extract_metadata(
                project_id, storage_key, tmp,
                progress_cb=make_progress_cb(project_id, "extract_metadata", job_id),
            )
        )
        profile = result.get("video_metadata", {})
        update_job_status(job_id, JobStatus.SUCCESS, 1.0, "Metadata extraction complete")
        emit_stage_complete(project_id, "extract_metadata", t0,
                            f"fl={profile.get('focal_length_px', 'unknown')} "
                            f"rot={profile.get('rotation_deg', 0)}° "
                            f"hint={profile.get('scene_hint', 'unknown')}",
                            metrics={
                                "focal_length_px": profile.get("focal_length_px"),
                                "rotation_deg":    profile.get("rotation_deg", 0),
                                "stable_frames":   len(profile.get("stable_frame_timestamps", [])),
                            })
        # Inject scout calibration from Redis if this is a scout/full run
        import redis as _redis, json as _json
        _r = _redis.from_url(settings.REDIS_URL)
        _scout_raw = _r.get(f"project:{project_id}:scout_calibration")
        if _scout_raw:
            try:
                result["_calibration"] = _json.loads(_scout_raw)
            except Exception:
                pass
        _exclude_raw = _r.get(f"project:{project_id}:exclude_marker_ids")
        if _exclude_raw:
            try:
                result["exclude_marker_ids"] = _json.loads(_exclude_raw)
            except Exception:
                pass
        _resize_raw = _r.get(f"project:{project_id}:resize_preset")
        if _resize_raw:
            result["resize_preset"] = _resize_raw.decode() if isinstance(_resize_raw, bytes) else _resize_raw
        _scene_type_raw = _r.get(f"project:{project_id}:scene_type")
        if _scene_type_raw:
            result["scene_type"] = _scene_type_raw.decode() if isinstance(_scene_type_raw, bytes) else _scene_type_raw
        result["storage_key"]      = storage_key
        result["all_storage_keys"] = [storage_key] + list(extra_keys or [])
        return result
    except Exception as e:
        update_job_status(job_id, JobStatus.FAILED, error=str(e))
        raise
    finally:
        hb.set()
        shutil.rmtree(tmp, ignore_errors=True)


# ── Stage 1: Extract Frames ───────────────────────────────────────────────────

@celery_app.task(bind=True, name="pipeline.extract_frames")
def extract_frames(self, prev_result: dict, project_id: str) -> dict:
    """
    Extract frames from the uploaded video.
    Receives prev_result from extract_metadata with storage_key and video_metadata.
    Applies device rotation correction and uses stable timestamps when available.
    Returns: {frame_keys, frame_count, rotation_deg, video_metadata, storage_key}
    """
    from backend.workers.pipeline.extractor import run_extract_frames

    storage_key      = prev_result.get("storage_key", "")
    all_storage_keys = prev_result.get("all_storage_keys")
    video_metadata   = prev_result.get("video_metadata")
    calibration      = prev_result.get("_calibration", {})
    resize_preset    = prev_result.get("resize_preset")
    tmp    = job_tmp(self.request.id)
    job_id = self.request.id
    t0     = _time.time()
    hb     = start_heartbeat(project_id, "extract_frames")
    try:
        cal_frames  = calibration.get("target_frames")
        n_sources   = len(all_storage_keys) if all_storage_keys else 1
        source_note = f" ({n_sources} source(s))" if n_sources > 1 else ""
        update_job_status(job_id, JobStatus.RUNNING, 0.0,
                          f"Starting frame extraction{source_note}"
                          + (f" (calibrated: {cal_frames} frames)" if cal_frames else "") + "…")
        result = asyncio.run(
            run_extract_frames(
                project_id, storage_key, tmp, job_id,
                video_metadata=video_metadata,
                max_frames_override=cal_frames,
                all_storage_keys=all_storage_keys,
                resize_preset=resize_preset,
                progress_cb=make_progress_cb(project_id, "extract_frames", job_id),
            )
        )
        n              = result.get("frame_count", 0)
        src_counts     = result.get("source_counts", {})
        blurry_stills  = result.get("blurry_stills_skipped", 0)
        n_sources      = len(result.get("frame_source_boundaries") or [])
        src_note       = (f" from {n_sources} source(s)"
                         f" ({src_counts.get('stills',0)} still(s)"
                         f" + {src_counts.get('videos',0)} video(s))"
                         if n_sources > 1 else "")
        blurry_note    = f", {blurry_stills} blurry still(s) skipped" if blurry_stills else ""
        update_job_status(job_id, JobStatus.SUCCESS, 1.0,
                          f"{n:,} frames extracted{src_note}{blurry_note}")
        emit_stage_complete(project_id, "extract_frames", t0,
                            f"Extracted {n:,} frames{src_note}{blurry_note}"
                            + (f" (rotated {result.get('rotation_deg', 0)}°)"
                               if result.get("rotation_deg") else ""),
                            metrics={"frame_count": n,
                                     "rotation_deg":       result.get("rotation_deg", 0),
                                     "n_sources":          n_sources,
                                     "stills_accepted":    src_counts.get("stills", 0),
                                     "blurry_stills_skipped": blurry_stills})
        # Thread metadata forward through the chain
        result["video_metadata"]   = video_metadata
        result["storage_key"]      = storage_key
        result["all_storage_keys"] = all_storage_keys
        if resize_preset:
            result["resize_preset"] = resize_preset
        if calibration:
            result["_calibration"] = calibration
        return result
    except Exception as e:
        update_job_status(job_id, JobStatus.FAILED, error=str(e))
        raise
    finally:
        hb.set()
        shutil.rmtree(tmp, ignore_errors=True)


# ── Stage 2: Detect ArUco Markers (FAIL FAST) ─────────────────────────────────

@celery_app.task(bind=True, name="pipeline.detect_aruco")
def detect_aruco_task(self, prev_result: dict, project_id: str) -> dict:
    """
    Detect ArUco markers in extracted frames.
    FAILS FAST with a clear error if no markers are found.
    Stores detections + baselines in the DB.

    Expects prev_result with frame_keys and video_metadata.
    Returns prev_result augmented with {aruco_result: {...}}.
    """
    from backend.workers.pipeline.aruco_detector import run_aruco_detection

    frame_keys     = prev_result.get("frame_keys", [])
    video_metadata = prev_result.get("video_metadata") or {}
    job_id = self.request.id
    tmp    = job_tmp(job_id)
    t0     = _time.time()
    hb     = start_heartbeat(project_id, "detect_aruco")
    try:
        update_job_status(job_id, JobStatus.RUNNING, 0.0, "Scanning for ArUco markers…")

        aruco_result = asyncio.run(
            run_aruco_detection(
                project_id, frame_keys, video_metadata, tmp,
                progress_cb=make_progress_cb(project_id, "detect_aruco", job_id),
            )
        )

        # Persist aruco_markers to DB for debugging / UI inspection
        try:
            import psycopg2, json as _json
            conn = psycopg2.connect(
                settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgres://")
            )
            cur = conn.cursor()
            cur.execute(
                "UPDATE projects SET aruco_markers = %s WHERE id = %s",
                (json.dumps(_json_safe(aruco_result.get("aruco_markers", {}))), project_id),
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as db_err:
            logger.warning("[%s] detect_aruco: DB persist failed: %s", project_id, db_err)

        ids   = aruco_result.get("aruco_ids_found", [])
        n_bas = len(aruco_result.get("aruco_baselines", []))
        update_job_status(job_id, JobStatus.SUCCESS, 1.0,
                          f"ArUco: {len(ids)} markers — IDs {ids}")
        emit_stage_complete(project_id, "detect_aruco", t0,
                            f"Found {len(ids)} marker(s) — IDs {ids}; "
                            f"{n_bas} inter-marker baseline(s)",
                            metrics={"n_markers": len(ids),
                                     "marker_ids": ids,
                                     "n_baselines": n_bas})

        result = dict(prev_result)
        result["aruco_result"] = aruco_result
        return result

    except RuntimeError as e:
        # ArUco fail-fast: propagate immediately so pipeline stops before COLMAP.
        logger.error("[%s] detect_aruco FAIL FAST: %s", project_id, e)
        update_job_status(job_id, JobStatus.FAILED, error=str(e))
        try:
            import psycopg2
            conn = psycopg2.connect(
                settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgres://")
            )
            cur = conn.cursor()
            cur.execute(
                "UPDATE projects SET status = %s::projectstatus WHERE id = %s",
                (ProjectStatus.FAILED.value, project_id),
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception:
            pass
        raise
    except Exception as e:
        update_job_status(job_id, JobStatus.FAILED, error=str(e))
        raise
    finally:
        hb.set()
        shutil.rmtree(tmp, ignore_errors=True)


# ── Stage 3: Feature Matching ─────────────────────────────────────────────────

@celery_app.task(bind=True, name="pipeline.feature_matching")
def feature_matching(self, prev_result: dict, project_id: str) -> dict:
    """
    Run LightGlue + DISK feature matching on extracted frames.
    Expects prev_result with frame_keys.
    Returns prev_result augmented with {match_data_key, pair_count, ...}.
    """
    from backend.workers.pipeline.matcher import run_feature_matching

    frame_keys              = prev_result.get("frame_keys", [])
    calibration             = prev_result.get("_calibration", {})
    match_window            = calibration.get("match_window")   # None = use module default
    frame_source_boundaries = prev_result.get("frame_source_boundaries")
    tmp = job_tmp(self.request.id)
    t0  = _time.time()
    hb  = start_heartbeat(project_id, "feature_matching")
    try:
        kw = {}
        if match_window:
            kw["window"] = int(match_window)
        elif prev_result.get("scene_type") == "object":
            # Orbit frames at 0° and 180° share no sequential neighbours.
            # Setting window to len(frame_keys) makes the pair set all-pairs,
            # giving every frame a direct connection across the full orbit.
            kw["window"] = max(999, len(frame_keys))
        if frame_source_boundaries:
            kw["frame_source_boundaries"] = frame_source_boundaries
        result = asyncio.run(
            run_feature_matching(
                project_id, frame_keys, tmp,
                progress_cb=lambda p, m: publish_progress(project_id, "feature_matching", p, m),
                **kw,
            )
        )
        result.update({k: v for k, v in prev_result.items() if k not in result})
        emit_stage_complete(project_id, "feature_matching", t0,
                            f"{result.get('pair_count', 0):,} pairs — "
                            f"{result.get('total_matches', 0):,} matches",
                            metrics={"pair_count": result.get("pair_count", 0),
                                     "total_matches": result.get("total_matches", 0),
                                     "n_frames": result.get("n_frames", len(frame_keys))})
        return result
    finally:
        hb.set()
        shutil.rmtree(tmp, ignore_errors=True)


# ── Stage 4: Structure from Motion ────────────────────────────────────────────

@celery_app.task(bind=True, name="pipeline.sfm")
def run_sfm(self, prev_result: dict, project_id: str) -> dict:
    """
    Run COLMAP SfM.  Injects focal length from video_metadata if available.
    Expects prev_result with match_data_key (and optionally video_metadata).
    Returns prev_result augmented with {sparse_cloud_key, camera_poses_key, ...}.
    """
    from backend.workers.pipeline.sfm import run_sfm_colmap

    match_data_key = prev_result.get("match_data_key", "")
    video_metadata = prev_result.get("video_metadata")
    tmp = job_tmp(self.request.id)
    t0  = _time.time()
    hb  = start_heartbeat(project_id, "sfm")
    try:
        result = asyncio.run(
            run_sfm_colmap(
                project_id, match_data_key, tmp,
                video_metadata=video_metadata,
                progress_cb=lambda p, m: publish_progress(project_id, "sfm", p, m),
            )
        )
        result.update({k: v for k, v in prev_result.items() if k not in result})
        reg   = result.get("registered_images", 0)
        total = len(prev_result.get("frame_keys", []))
        pts   = result.get("num_points3D", 0)
        err   = result.get("mean_reprojection_error", 0)
        warnings = []
        if total and reg < total * 0.8:
            warnings.append(
                f"Only {reg}/{total} frames registered — low overlap or scene too featureless"
            )
        emit_stage_complete(project_id, "sfm", t0,
                            f"Registered {reg}/{total} frames — {pts:,} sparse points, "
                            f"mean reproj error {err:.2f}px",
                            metrics={"registered_images": reg, "total_images": total,
                                     "num_points3D": pts,
                                     "mean_reprojection_error": round(err, 3)},
                            warnings=warnings)
        return result
    finally:
        hb.set()
        shutil.rmtree(tmp, ignore_errors=True)


# ── Stage 5: Dense Reconstruction (MVS) ──────────────────────────────────────

@celery_app.task(bind=True, name="pipeline.mvs")
def run_mvs(self, prev_result: dict, project_id: str) -> dict:
    """
    Run COLMAP MVS dense reconstruction.
    Expects prev_result with sparse_cloud_key and camera_poses_key.
    Returns prev_result augmented with {dense_cloud_key, dense_point_count, ...}.
    """
    from backend.workers.pipeline.mvs import run_mvs_openmvs

    sparse_cloud_key    = prev_result.get("sparse_cloud_key", "")
    camera_poses_key    = prev_result.get("camera_poses_key", "")
    mvs_params_override = dict(prev_result.get("mvs_params_override") or {})
    calibration         = prev_result.get("_calibration", {})
    # Inject calibrated MVS consistency threshold if provided
    if calibration.get("mvs_min_consistent"):
        mvs_params_override.setdefault("min_num_consistent", calibration["mvs_min_consistent"])
    tmp = job_tmp(self.request.id)
    t0  = _time.time()
    hb  = start_heartbeat(project_id, "mvs")
    try:
        result = asyncio.run(
            run_mvs_openmvs(
                project_id, sparse_cloud_key, camera_poses_key, tmp,
                mvs_params_override=mvs_params_override,
                progress_cb=lambda p, m: publish_progress(project_id, "mvs", p, m),
            )
        )
        result.update({k: v for k, v in prev_result.items() if k not in result})
        pts = result.get("dense_point_count", 0)
        emit_stage_complete(project_id, "mvs", t0,
                            f"Dense reconstruction — {pts:,} points",
                            metrics={"dense_point_count": pts,
                                     "runtime_s": round(result.get("runtime_seconds", 0))})
        return result
    finally:
        hb.set()
        shutil.rmtree(tmp, ignore_errors=True)



@celery_app.task(bind=True, name="pipeline.correct_trajectory_jumps")
def correct_trajectory_jumps_task(self, prev_result: dict, project_id: str) -> dict:
    """
    Correct mis-registered "teleport" blocks flagged by SfM
    (sfm_trajectory_jumps): when the camera dips into a small featureless
    space and tracking breaks, the rest of that segment can get re-anchored
    to a duplicate of nearby geometry. This rigidly re-aligns the duplicate
    block back onto the original, using trajectory continuity + ICP.

    Pass-through if no jumps were flagged.
    """
    from backend.workers.pipeline.trajectory_correction import run_trajectory_correction

    job_id = self.request.id
    tmp    = job_tmp(job_id)
    t0     = _time.time()
    hb     = start_heartbeat(project_id, "correct_trajectory_jumps")
    try:
        result = asyncio.run(
            run_trajectory_correction(
                project_id, prev_result, tmp,
                progress_cb=lambda p, m: publish_progress(project_id, "correct_trajectory_jumps", p, m),
            )
        )
        tc = result.get("trajectory_correction", {})
        if tc.get("applied"):
            n_jumps = len(tc.get("jumps_applied", []))
            n_pts   = sum(j["n_points_corrected"] for j in tc["jumps_applied"])
            emit_stage_complete(project_id, "correct_trajectory_jumps", t0,
                                f"Corrected {n_jumps} duplicated section(s), {n_pts:,} points repositioned",
                                metrics={"jumps_corrected": n_jumps, "points_corrected": n_pts})
        else:
            emit_stage_complete(project_id, "correct_trajectory_jumps", t0, "No corrections needed")
        return result
    finally:
        hb.set()
        shutil.rmtree(tmp, ignore_errors=True)


# ── Stage 6: Scale from ArUco ─────────────────────────────────────────────────

@celery_app.task(bind=True, name="pipeline.detect_aruco_sfm")
def detect_aruco_sfm_task(self, prev_result: dict, project_id: str) -> dict:
    """
    Re-scan ArUco markers on SfM-registered frames only.
    Guarantees every detection can be triangulated by scale_from_aruco.
    Fast stage: OpenCV detection only, no GPU needed.
    """
    from backend.workers.pipeline.aruco_sfm import run_aruco_sfm

    job_id = self.request.id
    tmp    = job_tmp(job_id)
    t0     = _time.time()
    hb     = start_heartbeat(project_id, "detect_aruco_sfm")
    try:
        update_job_status(job_id, JobStatus.RUNNING, 0.0, "Re-scanning ArUco on registered frames…")
        result = asyncio.run(
            run_aruco_sfm(
                project_id, prev_result, tmp,
                progress_cb=make_progress_cb(project_id, "detect_aruco_sfm", job_id),
            )
        )
        sfm_result = result.get("aruco_result", {})
        n_frames   = len(sfm_result.get("aruco_markers_sfm", {}))
        n_ids      = len(sfm_result.get("aruco_ids_sfm", []))
        if n_frames == 0:
            logger.warning(
                "[%s] detect_aruco_sfm: 0 registered frames with ArUco markers — "
                "scale_from_aruco will fall back to pre-SfM solvePnP path (less accurate)",
                project_id,
            )
        update_job_status(job_id, JobStatus.SUCCESS, 1.0,
                          f"ArUco SfM: {n_frames} frames, {n_ids} markers")
        emit_stage_complete(project_id, "detect_aruco_sfm", t0,
                            f"{n_frames} registered frames with markers, {n_ids} IDs",
                            metrics={"sfm_frames_with_markers": n_frames, "sfm_marker_ids": n_ids})
        # Save reprocess checkpoint — lets the /reprocess API re-dispatch any
        # downstream stage without manual Redis archaeology.
        try:
            import redis as _rc, json as _jc
            _r = _rc.from_url(settings.REDIS_URL)
            _r.set(f"project:{project_id}:reprocess_checkpoint",
                   _jc.dumps(result), ex=_TTL_MONTH)
        except Exception as _e:
            logger.warning("[%s] detect_aruco_sfm: checkpoint save failed: %s", project_id, _e)
        return result
    except Exception as e:
        update_job_status(job_id, JobStatus.FAILED, error=str(e))
        raise
    finally:
        hb.set()
        shutil.rmtree(tmp, ignore_errors=True)


@celery_app.task(bind=True, name="pipeline.scale_from_aruco")
def scale_from_aruco_task(self, prev_result: dict, project_id: str) -> dict:
    """
    Derive metric scale from ArUco markers + SfM camera poses.
    Triangulates marker positions from 2D detections + SfM cameras.
    Computes physical_dist / sfm_dist for marker pairs.

    Expects prev_result with aruco_result and camera_poses_key.
    Returns prev_result augmented with {confirmed_scale_factor, gravity_up_world, scale_diagnostics}.
    """
    from backend.workers.pipeline.scale_from_aruco import run_scale_from_aruco

    job_id = self.request.id
    tmp    = job_tmp(job_id)
    t0     = _time.time()
    hb     = start_heartbeat(project_id, "scale_from_aruco")
    try:
        update_job_status(job_id, JobStatus.RUNNING, 0.0, "Deriving scale from ArUco markers…")
        result = asyncio.run(
            run_scale_from_aruco(
                project_id, prev_result, tmp,
                progress_cb=make_progress_cb(project_id, "scale_from_aruco", job_id),
            )
        )
        scale  = result.get("confirmed_scale_factor")
        diag   = result.get("scale_diagnostics", {})
        warnings = []
        if scale is None:
            warnings.append(
                f"Scale not derivable ({diag.get('error', 'unknown reason')}). "
                "Cloud will be exported in SfM units. "
                "Ensure 2+ ArUco markers are visible in the same frame."
            )
        update_job_status(job_id, JobStatus.SUCCESS, 1.0,
                          f"Scale: {scale:.6f} m/unit" if scale else "Scale not derived")
        emit_stage_complete(project_id, "scale_from_aruco", t0,
                            f"scale={scale:.6f} m/sfm-unit" if scale else "scale unavailable",
                            metrics={"scale_factor": scale,
                                     "n_triangulated": len(diag.get("triangulated_markers", [])),
                                     "n_estimates": diag.get("n_scale_estimates", 0)},
                            warnings=warnings)
        return result
    except Exception as e:
        update_job_status(job_id, JobStatus.FAILED, error=str(e))
        raise
    finally:
        hb.set()
        shutil.rmtree(tmp, ignore_errors=True)


# ── Stage 7: Apply Scale ──────────────────────────────────────────────────────

@celery_app.task(bind=True, name="pipeline.apply_known_scale")
def apply_known_scale_task(self, prev_result: dict, project_id: str) -> dict:
    """
    Apply the confirmed metric scale factor to the dense point cloud.
    Gets scale_factor from prev_result.confirmed_scale_factor.
    If scale_factor is None, passes through the cloud unscaled (with a warning).

    Expects prev_result with dense_cloud_key and confirmed_scale_factor.
    Returns prev_result augmented with {scaled_cloud_key} (or dense_cloud_key if no scale).
    """
    import open3d as o3d

    scale_factor     = prev_result.get("confirmed_scale_factor")
    dense_cloud_key  = prev_result.get("dense_cloud_key", "")
    job_id           = self.request.id
    tmp              = job_tmp(job_id)
    storage          = get_storage()

    hb = start_heartbeat(project_id, "apply_scale")
    try:
        update_job_status(job_id, JobStatus.RUNNING, 0.0, "Applying metric scale…")
        publish_progress(project_id, "apply_scale", 0.0, "Applying metric scale…")

        if scale_factor is None or scale_factor <= 0:
            logger.warning(
                "[%s] apply_scale: no scale factor — exporting in SfM units (no real-world scale)",
                project_id,
            )
            publish_progress(project_id, "apply_scale", 1.0,
                             "No scale factor available — cloud in SfM units")
            update_job_status(job_id, JobStatus.SUCCESS, 1.0, "Scale not applied (no factor)")
            result = dict(prev_result)
            result["scaled_cloud_key"] = dense_cloud_key
            result["scale_applied"]    = False
            return result

        dense_ply_local = tmp / "dense.ply"
        asyncio.run(storage.download(dense_cloud_key, dense_ply_local))

        publish_progress(project_id, "apply_scale", 0.4,
                         f"Scaling by {scale_factor:.6f} m/unit…")

        pcd = o3d.io.read_point_cloud(str(dense_ply_local))
        pcd.scale(scale_factor, center=(0, 0, 0))

        scaled_ply_local = tmp / "scaled.ply"
        o3d.io.write_point_cloud(str(scaled_ply_local), pcd)

        scaled_cloud_key = f"{project_id}/clouds/scaled.ply"
        asyncio.run(storage.upload(scaled_ply_local, scaled_cloud_key))

        # Persist scale to DB
        try:
            import psycopg2
            conn = psycopg2.connect(
                settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgres://")
            )
            cur = conn.cursor()
            cur.execute(
                "UPDATE projects SET confirmed_scale_factor=%s, confirmed_scale_source=%s WHERE id=%s",
                (scale_factor, prev_result.get("confirmed_scale_source", "aruco"), project_id),
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as db_err:
            logger.warning("[%s] apply_scale: DB persist failed: %s", project_id, db_err)

        update_job_status(job_id, JobStatus.SUCCESS, 1.0, "Scale applied successfully")
        publish_progress(project_id, "apply_scale", 1.0,
                         f"Scale applied — {scale_factor:.6f} m/unit")

        result = dict(prev_result)
        result.update({
            "scaled_cloud_key": scaled_cloud_key,
            "scale_factor":     scale_factor,
            "scale_applied":    True,
        })
        return result

    except Exception as e:
        update_job_status(job_id, JobStatus.FAILED, error=str(e))
        raise
    finally:
        hb.set()
        shutil.rmtree(tmp, ignore_errors=True)


# ── Stage 7b: Room Layout (gravity-aware plane projection) ───────────────────

@celery_app.task(bind=True, name="pipeline.fill_planes")
def fill_planes_task(self, prev_result: dict, project_id: str) -> dict:
    """
    Gravity-aware architectural plane reconstruction.
    Estimates gravity from camera poses, finds floor/walls/ceiling via RANSAC,
    and projects nearby real points onto each plane — no synthetic fill.
    """
    from backend.workers.pipeline.room_layout import run_room_layout

    job_id = self.request.id
    tmp    = job_tmp(job_id)
    t0     = _time.time()
    hb     = start_heartbeat(project_id, "fill_planes")
    try:
        update_job_status(job_id, JobStatus.RUNNING, 0.0,
                          "Estimating gravity and fitting room planes…")
        result = asyncio.run(
            run_room_layout(
                project_id, prev_result, tmp,
                progress_cb=make_progress_cb(project_id, "fill_planes", job_id),
            )
        )
        layout = result.get("room_layout", {})
        n_walls = layout.get("n_walls", 0)
        update_job_status(job_id, JobStatus.SUCCESS, 1.0,
                          f"Room layout: floor={layout.get('has_floor')}, "
                          f"ceiling={layout.get('has_ceiling')}, walls={n_walls}")
        emit_stage_complete(project_id, "fill_planes", t0,
                            f"floor+ceiling+{n_walls} walls projected",
                            metrics={k: (v if not isinstance(v, list) else str(v))
                                     for k, v in layout.items()})
        return result
    except Exception as e:
        update_job_status(job_id, JobStatus.FAILED, error=str(e))
        raise
    finally:
        hb.set()
        shutil.rmtree(tmp, ignore_errors=True)


# ── Stage 7c: Point Cloud Refinement ─────────────────────────────────────────

@celery_app.task(bind=True, name="pipeline.refine_cloud")
def refine_cloud_task(self, prev_result: dict, project_id: str) -> dict:
    """
    Post-scale point cloud refinement:
      - Statistical outlier removal (remove COLMAP noise spikes)
      - Voxel downsampling (uniform spatial density, faster coverage scoring)
    Fast CPU stage, runs in seconds.
    """
    from backend.workers.pipeline.refine_cloud import run_refine_cloud

    job_id = self.request.id
    tmp    = job_tmp(job_id)
    t0     = _time.time()
    hb     = start_heartbeat(project_id, "refine_cloud")
    try:
        scale = prev_result.get("confirmed_scale_factor")
        update_job_status(job_id, JobStatus.RUNNING, 0.0, "Refining point cloud…")
        result = asyncio.run(
            run_refine_cloud(
                project_id, prev_result, tmp,
                scale_factor=scale,
                scene_type=prev_result.get("scene_type"),
                progress_cb=make_progress_cb(project_id, "refine_cloud", job_id),
            )
        )
        ref = result.get("refinement", {})
        update_job_status(job_id, JobStatus.SUCCESS, 1.0,
                          f"{ref.get('n_before',0):,} → {ref.get('n_after',0):,} points")
        emit_stage_complete(project_id, "refine_cloud", t0,
                            f"noise removed: {ref.get('sor_removed',0):,}, "
                            f"final: {ref.get('n_after',0):,} pts",
                            metrics=ref)
        return result
    except Exception as e:
        update_job_status(job_id, JobStatus.FAILED, error=str(e))
        raise
    finally:
        hb.set()
        shutil.rmtree(tmp, ignore_errors=True)


@celery_app.task(bind=True, name="pipeline.lingbot_fusion")
def lingbot_fusion_task(self, prev_result: dict, project_id: str) -> dict:
    """
    Optional densification: fuse LingBot-Map depth maps into the COLMAP
    reconstruction (per-frame affine calibration + TSDF) and emit a SEPARATE
    densified cloud + mesh. Additive — does not alter dense_cloud_key. Depth
    inference runs in an isolated torch-2.8 venv via subprocess.
    """
    from backend.workers.pipeline.lingbot_fusion import run_lingbot_fusion

    job_id = self.request.id
    tmp    = job_tmp(job_id)
    t0     = _time.time()
    hb     = start_heartbeat(project_id, "lingbot_fusion")
    try:
        scale = prev_result.get("confirmed_scale_factor")
        update_job_status(job_id, JobStatus.RUNNING, 0.0, "Densifying with LingBot depth fusion…")
        result = asyncio.run(
            run_lingbot_fusion(
                project_id, prev_result, tmp,
                scale_factor=scale,
                scene_type=prev_result.get("scene_type"),
                progress_cb=make_progress_cb(project_id, "lingbot_fusion", job_id),
            )
        )
        m = result.get("lingbot_fusion", {})
        update_job_status(job_id, JobStatus.SUCCESS, 1.0,
                          f"{m.get('fused_point_count',0):,} pts, {m.get('frames_used',0)} frames")
        emit_stage_complete(project_id, "lingbot_fusion", t0,
                            f"densified: {m.get('fused_point_count',0):,} pts, "
                            f"{m.get('fused_mesh_tris',0):,} tris",
                            metrics=m)
        return result
    except Exception as e:
        # Additive/optional stage: never fail the whole pipeline on densification error.
        logger.exception("[%s] lingbot_fusion failed (non-fatal): %s", project_id, e)
        update_job_status(job_id, JobStatus.FAILED, error=str(e))
        emit_stage_complete(project_id, "lingbot_fusion", t0,
                            f"densification skipped (error): {e}",
                            warnings=[str(e)])
        return prev_result
    finally:
        hb.set()
        shutil.rmtree(tmp, ignore_errors=True)


# ── Stage 8: Coverage Analysis ────────────────────────────────────────────────

@celery_app.task(bind=True, name="pipeline.coverage")
def analyze_coverage(self, prev_result: dict, project_id: str) -> dict:
    """
    Score coverage per point (view count + angle), colour the cloud.
    Generates re-shoot suggestions for under-covered areas.

    Uses scaled_cloud_key if scale was applied, falls back to dense_cloud_key.
    NOTE: cameras.json is in SfM units; coverage uses the unscaled SfM cloud
    to keep coordinate systems consistent.
    """
    from backend.workers.pipeline.coverage import run_coverage_analysis

    # Coverage runs in SfM units — use dense_cloud_key (not scaled).
    # HPR frustum culling requires camera and cloud in the same coordinate frame.
    cloud_key        = prev_result.get("dense_cloud_key", "")
    camera_poses_key = prev_result.get("camera_poses_key") or f"{project_id}/sfm/cameras.json"
    tmp = job_tmp(self.request.id)
    t0  = _time.time()
    hb  = start_heartbeat(project_id, "coverage")
    try:
        _cov_scene_type = prev_result.get("scene_type")
        if not _cov_scene_type:
            try:
                import psycopg2 as _pg_c
                _c_c = _pg_c.connect(settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgres://"))
                _cur_c = _c_c.cursor()
                _cur_c.execute("SELECT scene_type FROM projects WHERE id = %s", (project_id,))
                _row_c = _cur_c.fetchone()
                _c_c.close()
                _cov_scene_type = (_row_c[0] if _row_c else None) or "indoor_room"
            except Exception:
                _cov_scene_type = "indoor_room"

        result = asyncio.run(
            run_coverage_analysis(
                project_id, cloud_key, camera_poses_key, tmp,
                progress_cb=lambda p, m: publish_progress(project_id, "coverage", p, m),
                source_object=prev_result.get("confirmed_scale_source"),
                gravity_up_world=prev_result.get("gravity_up_world"),
                scene_type=_cov_scene_type,
            )
        )
        result.update({k: v for k, v in prev_result.items() if k not in result})
        score = result.get("coverage_score", 0)
        sugg  = result.get("suggestions", [])
        emit_stage_complete(project_id, "coverage", t0,
                            f"Coverage {score*100:.0f}% — {len(sugg)} suggestion(s)",
                            metrics={"coverage_score": round(score, 3),
                                     "n_suggestions": len(sugg)})
        return result
    finally:
        hb.set()
        shutil.rmtree(tmp, ignore_errors=True)


# ── Stage 9: Export ───────────────────────────────────────────────────────────

@celery_app.task(bind=True, name="pipeline.export")
def export_outputs(self, prev_result: dict, project_id: str) -> dict:
    """
    Export final deliverables: .ply, .obj, .las.
    Marks the project complete (or needs_more if coverage is low).
    """
    from backend.workers.pipeline.exporter import run_export

    # Prefer the layout/filled cloud for export when available; fall back to
    # the coverage-colored cloud (which only contains real COLMAP points).
    export_cloud_key = (
        prev_result.get("scaled_cloud_key")   # set by room_layout / fill_planes
        or prev_result.get("dense_cloud_key")
        or prev_result.get("coverage_cloud_key", "")
    )
    coverage_cloud_key = prev_result.get("coverage_cloud_key", "")
    coverage_score     = prev_result.get("coverage_score")
    tmp = job_tmp(self.request.id)
    t0  = _time.time()
    hb  = start_heartbeat(project_id, "export")
    try:
        # scene_type may not survive all reprocess paths — read from DB as fallback
        scene_type = prev_result.get("scene_type")
        if not scene_type:
            try:
                import psycopg2 as _pg
                _c = _pg.connect(settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgres://"))
                _cur = _c.cursor()
                _cur.execute("SELECT scene_type FROM projects WHERE id = %s", (project_id,))
                _row = _cur.fetchone()
                scene_type = (_row[0] if _row else None) or "indoor_room"
                _cur.close(); _c.close()
            except Exception:
                scene_type = "indoor_room"

        result = asyncio.run(
            run_export(
                project_id, export_cloud_key, tmp,
                progress_cb=lambda p, m: publish_progress(project_id, "export", p, m),
                scene_type=scene_type,
            )
        )
        result.update({k: v for k, v in prev_result.items() if k not in result})

        # Mark project complete or needs_more.
        # Objects need higher coverage since all surfaces are expected to be
        # captured; rooms always have some occluded ceiling/corner that's fine.
        is_obj = prev_result.get("scene_type") == "object"
        NEEDS_MORE_THRESHOLD = 0.80 if is_obj else 0.50
        final_status = (
            ProjectStatus.NEEDS_MORE.value
            if coverage_score is not None and coverage_score < NEEDS_MORE_THRESHOLD
            else ProjectStatus.COMPLETE.value
        )

        try:
            import psycopg2, json as _json
            conn = psycopg2.connect(
                settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgres://")
            )
            cur = conn.cursor()
            suggestions = result.get("suggestions", [])
            run_ts  = result.get("coverage_run_ts")
            ts_key  = result.get("coverage_ts_key")
            cur.execute("SELECT coverage_runs FROM projects WHERE id=%s", (project_id,))
            row = cur.fetchone()
            raw = row[0] if row else None
            runs = raw if isinstance(raw, list) else (_json.loads(raw) if raw else [])
            runs.append({
                "ts": run_ts,
                "score": coverage_score,
                "cloud_key": ts_key,
                "n_suggestions": len(suggestions),
            })
            splat_key = result.get("splat_key")
            mesh_key  = result.get("mesh_key")
            lingbot_cloud_key = result.get("lingbot_cloud_key")
            lingbot_mesh_key  = result.get("lingbot_mesh_key")
            cur.execute(
                "UPDATE projects SET status=%s::projectstatus, coverage_score=%s, "
                "suggestions=%s, coverage_runs=%s, splat_key=%s, mesh_key=%s, "
                "lingbot_cloud_key=%s, lingbot_mesh_key=%s WHERE id=%s",
                (final_status, coverage_score, _json.dumps(suggestions),
                 _json.dumps(runs), splat_key, mesh_key,
                 lingbot_cloud_key, lingbot_mesh_key, project_id),
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as db_err:
            logger.error("[%s] export: failed to mark project complete: %s", project_id, db_err)

        emit_stage_complete(project_id, "export", t0,
                            f"Exported {len(result.get('exports', []))} file(s)",
                            metrics={"n_exports": len(result.get("exports", [])),
                                     "coverage_score": coverage_score})
        return result
    finally:
        hb.set()
        shutil.rmtree(tmp, ignore_errors=True)


# ── 3DGS (optional separate chain) ───────────────────────────────────────────

@celery_app.task(bind=True, name="pipeline.gaussian_splatting")
def gaussian_splatting_task(self, prev_result: dict, project_id: str) -> dict:
    """
    Train a 3D Gaussian Splatting model using nerfstudio splatfacto.
    Replaces COLMAP MVS for the 3DGS pipeline variant.
    """
    import os
    from backend.workers.pipeline.gaussian_splatting import run_gaussian_splatting

    workspace_key    = prev_result.get("workspace_key", "")
    camera_poses_key = prev_result.get("camera_poses_key", "")
    sparse_cloud_key = prev_result.get("sparse_cloud_key", "")
    frame_keys       = prev_result.get("frame_keys") or prev_result.get("image_keys", [])
    n_iter           = int(os.environ.get("GSPLAT_ITERATIONS", "15000"))
    tmp              = job_tmp(self.request.id)

    if not workspace_key or not camera_poses_key:
        logger.error("[%s] gaussian_splatting: missing SfM outputs in prev_result", project_id)
        return dict(prev_result)

    try:
        result = asyncio.run(
            run_gaussian_splatting(
                project_id=project_id,
                workspace_key=workspace_key,
                camera_poses_key=camera_poses_key,
                sparse_cloud_key=sparse_cloud_key,
                frame_keys=frame_keys,
                tmp=tmp,
                progress_cb=lambda p, m: publish_progress(project_id, "gaussian_splatting", p, m),
                n_iterations=n_iter,
            )
        )
        merged = dict(prev_result)
        merged.update(result)
        return merged
    except Exception as e:
        logger.error("[%s] gaussian_splatting failed: %s", project_id, e, exc_info=True)
        return dict(prev_result)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            import torch as _torch
            _torch.cuda.empty_cache()
        except Exception:
            pass


# ── Scout Calibration Task ────────────────────────────────────────────────────

@celery_app.task(bind=True, name="pipeline.scout_calibrate")
def scout_calibrate_task(self, prev_result: dict, project_id: str) -> dict:
    """
    Analyse the scout SfM result, compute calibrated parameters for the full
    run, store them in DB, and trigger the full pipeline chain.
    Runs on CPU queue (fast — pure analysis + chain launch).
    """
    from backend.workers.pipeline.scout_calibrate import run_scout_calibrate

    job_id = self.request.id
    tmp    = job_tmp(job_id)
    t0     = _time.time()
    hb     = start_heartbeat(project_id, "scout_calibrate")
    try:
        update_job_status(job_id, JobStatus.RUNNING, 0.0, "Computing calibration from scout results…")
        result = asyncio.run(
            run_scout_calibrate(
                project_id, prev_result, tmp,
                progress_cb=make_progress_cb(project_id, "scout_calibrate", job_id),
            )
        )
        cal = result.get("_scout_calibration") or prev_result
        update_job_status(job_id, JobStatus.SUCCESS, 1.0, "Calibration complete — full pipeline launched")
        emit_stage_complete(project_id, "scout_calibrate", t0, "Full pipeline launched with calibrated params")
        return result
    except Exception as e:
        update_job_status(job_id, JobStatus.FAILED, error=str(e))
        raise
    finally:
        hb.set()
        shutil.rmtree(tmp, ignore_errors=True)


# ── Full Pipeline Chain ───────────────────────────────────────────────────────

def launch_pipeline(project_id: str, storage_key: str, extra_keys: list | None = None,
                     exclude_marker_ids: list | None = None, resize_preset: str | None = None,
                     scene_type: str | None = None, lingbot_enabled: bool = False):
    """
    Kick off the full pipeline as a single continuous Celery chain.

    Chain (indoor_room / outdoor):
      extract_metadata → extract_frames → detect_aruco (FAIL FAST) →
      feature_matching → sfm → mvs → correct_trajectory_jumps →
      detect_aruco_sfm → scale_from_aruco → apply_known_scale →
      fill_planes → refine_cloud → coverage → export

    Chain (object):
      Same as above but fill_planes is omitted — floor/wall RANSAC is
      irrelevant when orbiting a discrete object.

    extra_keys: additional video/still storage keys to include in extract_frames.
    exclude_marker_ids: ArUco IDs to ignore in scale derivation (e.g. duplicate
    physical markers), injected into prev_result by extract_metadata_task.
    The pipeline fails immediately at detect_aruco if no ArUco markers are
    found.  No user gate — scale is derived automatically.
    """
    from celery import chain

    import redis as _redis, json as _json
    _r = _redis.from_url(settings.REDIS_URL)
    if exclude_marker_ids:
        _r.set(
            f"project:{project_id}:exclude_marker_ids",
            _json.dumps(exclude_marker_ids),
            ex=_TTL_DAY,
        )
    if resize_preset:
        _r.set(
            f"project:{project_id}:resize_preset",
            resize_preset,
            ex=_TTL_DAY,
        )
    if scene_type:
        _r.set(
            f"project:{project_id}:scene_type",
            scene_type,
            ex=_TTL_DAY,
        )

    is_object  = (scene_type == "object")
    is_outdoor = (scene_type == "outdoor")

    tasks = [
        extract_metadata_task.s(project_id, storage_key, extra_keys or []),
        extract_frames.s(project_id),
        detect_aruco_task.s(project_id),
        feature_matching.s(project_id),
        run_sfm.s(project_id),
        run_mvs.s(project_id),
        # trajectory correction targets room-scan teleport jumps; orbit paths
        # are continuous so this stage is skipped for object scans.
        *([correct_trajectory_jumps_task.s(project_id)] if not is_object else []),
        # 3DGS is the natural deliverable for object scans.
        *([gaussian_splatting_task.s(project_id)] if is_object else []),
        detect_aruco_sfm_task.s(project_id),
        scale_from_aruco_task.s(project_id),
        apply_known_scale_task.s(project_id),
        # fill_planes fits floor/wall/ceiling planes — meaningless for object
        # orbits and for outdoor terrain which is never flat.
        *([fill_planes_task.s(project_id)] if not is_object and not is_outdoor else []),
        refine_cloud_task.s(project_id),
        # Optional LingBot densification — per-project opt-in, with the env flag
        # as a fleet-wide master kill-switch.
        *([lingbot_fusion_task.s(project_id)]
          if (settings.ENABLE_LINGBOT_FUSION and lingbot_enabled) else []),
        analyze_coverage.s(project_id),
        export_outputs.s(project_id),
    ]

    pipeline = chain(*tasks)
    result   = pipeline.apply_async()

    _r.set(
        f"project:{project_id}:active_task", result.id, ex=_TTL_WEEK
    )
    _create_job_for_task(project_id, "extract_metadata", result.id)
    return result


def launch_full_pipeline(project_id: str, storage_key: str, calibration: dict,
                         extra_keys: list | None = None):
    """
    Launch the full pipeline with calibration parameters injected into prev_result.
    Called automatically by scout_calibrate_task after the scout run completes.

    Calibration keys consumed by downstream tasks:
        _calibration.target_frames      → extract_frames (max frame count)
        _calibration.match_window       → feature_matching (pair window)
        _calibration.mvs_min_consistent → run_mvs (depth consistency threshold)
    """
    from celery import chain
    import psycopg2 as _pg2

    # Read per-project densify opt-in (scout→full path has no caller param).
    lingbot_enabled = False
    # Mark project as starting full run
    try:
        conn = _pg2.connect(settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgres://"))
        cur  = conn.cursor()
        try:
            cur.execute("SELECT lingbot_enabled FROM projects WHERE id=%s", (project_id,))
            _row = cur.fetchone()
            lingbot_enabled = bool(_row[0]) if _row else False
        except Exception:
            lingbot_enabled = False
        cur.execute("UPDATE projects SET status=%s, pipeline_mode=%s WHERE id=%s",
                    ("processing", "full", project_id))
        conn.commit(); cur.close(); conn.close()
    except Exception as e:
        logger.warning("[%s] launch_full_pipeline: DB mode update failed: %s", project_id, e)

    tasks = [
        extract_metadata_task.s(project_id, storage_key, extra_keys or []),
        extract_frames.s(project_id),
        detect_aruco_task.s(project_id),
        feature_matching.s(project_id),
        run_sfm.s(project_id),
        run_mvs.s(project_id),
        correct_trajectory_jumps_task.s(project_id),
        detect_aruco_sfm_task.s(project_id),
        scale_from_aruco_task.s(project_id),
        apply_known_scale_task.s(project_id),
        fill_planes_task.s(project_id),
        refine_cloud_task.s(project_id),
        *([lingbot_fusion_task.s(project_id)]
          if (settings.ENABLE_LINGBOT_FUSION and lingbot_enabled) else []),
        analyze_coverage.s(project_id),
        export_outputs.s(project_id),
    ]

    # Store calibration in Redis so extract_metadata_task can inject it into the
    # chain's first prev_result.  The key must be written BEFORE apply_async()
    # because the first task may start immediately.
    import redis as _redis, json as _json
    _r = _redis.from_url(settings.REDIS_URL)
    _r.set(
        f"project:{project_id}:scout_calibration",
        _json.dumps(calibration),
        ex=_TTL_DAY,
    )

    pipeline = chain(*tasks)
    result   = pipeline.apply_async()

    _r.set(
        f"project:{project_id}:active_task", result.id, ex=_TTL_WEEK
    )
    _create_job_for_task(project_id, "extract_metadata", result.id)
    return result


def launch_scout_pipeline(project_id: str, storage_key: str, extra_keys: list | None = None):
    """
    Launch the scout (fast calibration) pipeline.
    Uses 40 frames, sequential-only matching (window=5), no MVS.
    On completion, scout_calibrate_task auto-launches the full pipeline.
    """
    from celery import chain
    import psycopg2 as _pg2

    try:
        conn = _pg2.connect(settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgres://"))
        cur  = conn.cursor()
        cur.execute("UPDATE projects SET status=%s, pipeline_mode=%s WHERE id=%s",
                    ("processing", "scout", project_id))
        conn.commit(); cur.close(); conn.close()
    except Exception as e:
        logger.warning("[%s] launch_scout_pipeline: DB mode update failed: %s", project_id, e)

    # Scout calibration: 40 frames, tight sequential matching
    scout_cal = {"target_frames": 40, "match_window": 5, "_is_scout": True}

    tasks = [
        extract_metadata_task.s(project_id, storage_key, extra_keys or []),
        extract_frames.s(project_id),        # reads _calibration.target_frames = 40
        detect_aruco_task.s(project_id),
        feature_matching.s(project_id),      # reads _calibration.match_window = 5
        run_sfm.s(project_id),
        detect_aruco_sfm_task.s(project_id),
        scale_from_aruco_task.s(project_id),
        scout_calibrate_task.s(project_id),  # analyses + triggers full run
    ]

    # Store scout calibration in Redis so extract_metadata_task can inject it
    import redis as _redis, json as _json
    r = _redis.from_url(settings.REDIS_URL)
    r.set(f"project:{project_id}:scout_calibration", _json.dumps(scout_cal), ex=_TTL_DAY)

    pipeline = chain(*tasks)
    result   = pipeline.apply_async()

    r.set(f"project:{project_id}:active_task", result.id, ex=_TTL_WEEK)
    _create_job_for_task(project_id, "extract_metadata", result.id)
    return result


def launch_supplemental_pipeline(project_id: str, new_storage_key: str):
    """
    Launch a supplemental reconstruction pass using a new video.
    New frames are extracted and registered into the existing sparse model.
    Re-runs the full pipeline using the new upload.
    """
    return launch_pipeline(project_id, new_storage_key)


def _create_job_for_task(project_id: str, stage_name: str, celery_id: str):
    """Helper to create a Job record in the database using synchronous SQL."""
    try:
        import psycopg2, uuid

        conn = psycopg2.connect(
            settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgres://")
        )
        cur = conn.cursor()

        job_id = str(uuid.uuid4())
        stage  = PipelineStage[stage_name.upper()].value
        status = JobStatus.PENDING.value

        sql = """INSERT INTO jobs (id, project_id, celery_id, stage, status, progress, message)
                 VALUES (%s, %s, %s, %s::pipelinestage, %s::jobstatus, %s, %s)"""
        cur.execute(sql, (job_id, project_id, celery_id, stage, status, 0.0, "Queued"))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning("Error creating job record: %s", e)
