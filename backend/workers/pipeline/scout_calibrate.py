"""
Scout calibration stage.

Analyses the scout SfM result to derive optimal parameters for the full
pipeline run. Stores calibration in the DB and triggers the full pipeline
as a new Celery chain.

Calibration logic is intentionally conservative — it scales parameters
based on observable scene properties, not video-specific assumptions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def compute_calibration(prev_result: dict) -> dict:
    """
    Derive full-run parameters from scout SfM metrics.

    Inputs (from prev_result):
        registered_images       int   — cameras SfM registered
        image_keys / frame_keys list  — all frames passed to SfM
        mean_reprojection_error float — geometric accuracy
        num_points3D            int   — sparse point count
        confirmed_scale_factor  float | None — scale from ArUco (if available)

    Outputs:
        target_frames       int   — how many frames to extract for full run
        match_window        int   — LightGlue sliding window size
        mvs_min_consistent  int   — COLMAP patch_match_stereo min consistency
        scout_*             various — diagnostics, passed through to result
    """
    total = len(prev_result.get("image_keys") or prev_result.get("frame_keys") or [])
    registered = prev_result.get("registered_images", 0)
    reproj = float(prev_result.get("mean_reprojection_error") or 1.5)
    n_pts = int(prev_result.get("num_points3D") or 0)
    scale = prev_result.get("confirmed_scale_factor")

    reg_rate = registered / max(total, 1)

    # ── Frame count ───────────────────────────────────────────────────────────
    # Start from 200 frames baseline. Scale up when registration rate is poor
    # (scene needs more overlap) or when the scene is large (many sparse points).
    # Cap at 400 to keep MVS tractable.
    base = 200
    if reg_rate < 0.60:
        target_frames = min(400, int(base * 1.6))   # 320 — bad connectivity
    elif reg_rate < 0.75:
        target_frames = min(400, int(base * 1.3))   # 260 — below-average
    elif reg_rate > 0.90 and reproj < 0.8:
        target_frames = max(120, int(base * 0.7))   # 140 — excellent, fewer needed
    else:
        target_frames = base                         # 200 — default

    # ── Matching window ───────────────────────────────────────────────────────
    # Larger window → more pairs → better loop closure → slower.
    # Use wider window when connectivity is poor.
    if reg_rate < 0.60:
        match_window = 25
    elif reg_rate < 0.80:
        match_window = 15   # default
    else:
        match_window = 10   # tight sequence, few long-range needed

    # ── MVS consistency threshold ─────────────────────────────────────────────
    # Higher = fewer but cleaner points. Use higher when SfM quality is good.
    # Lower = more points, noisier — needed when cameras have limited overlap.
    if reproj < 0.8 and reg_rate > 0.85:
        min_consistent = 4   # high quality — push for clean cloud
    elif reg_rate < 0.65:
        min_consistent = 2   # poor overlap — be generous to get any points
    else:
        min_consistent = 3   # standard

    calibration = {
        "target_frames":      target_frames,
        "match_window":       match_window,
        "mvs_min_consistent": min_consistent,
        "scout_reg_rate":     round(reg_rate, 3),
        "scout_registered":   registered,
        "scout_total_frames": total,
        "scout_reproj_error": round(reproj, 3),
        "scout_n_points":     n_pts,
        "scout_scale_factor": scale,
    }

    logger.info(
        "scout_calibrate: reg=%.0f%% reproj=%.2fpx → "
        "frames=%d window=%d mvs_min=%d",
        reg_rate * 100, reproj, target_frames, match_window, min_consistent,
    )
    return calibration


async def run_scout_calibrate(
    project_id: str,
    prev_result: dict,
    tmp: Path,
    progress_cb: Callable[[float, str], None],
) -> dict:
    """
    1. Compute calibration from scout SfM result.
    2. Persist calibration to DB.
    3. Trigger the full pipeline chain with calibrated parameters.
    4. Return prev_result (scout chain is now done).
    """
    from backend.core.config import settings
    from backend.workers.tasks import launch_full_pipeline

    progress_cb(0.1, "Analysing scout SfM results…")
    calibration = compute_calibration(prev_result)

    cal = calibration
    progress_cb(0.4, (
        f"Scout: reg={cal['scout_reg_rate']:.0%} reproj={cal['scout_reproj_error']:.2f}px — "
        f"full run: {cal['target_frames']} frames, window={cal['match_window']}, "
        f"MVS min_consistent={cal['mvs_min_consistent']}"
    ))

    # Persist calibration to DB
    try:
        import psycopg2
        conn = psycopg2.connect(
            settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgres://")
        )
        cur = conn.cursor()
        cur.execute(
            "UPDATE projects SET scout_calibration=%s, pipeline_mode=%s WHERE id=%s",
            (json.dumps(calibration), "scout_calibrating", project_id),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning("[%s] scout_calibrate: DB persist failed: %s", project_id, e)

    # Get the original storage key(s) to re-run the full pipeline
    storage_key = prev_result.get("storage_key")
    if not storage_key:
        logger.error("[%s] scout_calibrate: no storage_key in prev_result — cannot launch full run", project_id)
        progress_cb(1.0, "Scout calibration complete — could not auto-launch full run (no storage key)")
        return prev_result

    all_storage_keys = prev_result.get("all_storage_keys") or [storage_key]
    extra_keys = [k for k in all_storage_keys if k != storage_key]

    progress_cb(0.8, "Launching full pipeline with calibrated parameters…")

    try:
        launch_full_pipeline(project_id, storage_key, calibration, extra_keys=extra_keys)
        logger.info("[%s] scout_calibrate: full pipeline launched", project_id)
        progress_cb(1.0, f"Full pipeline launched — {calibration['target_frames']} frames, calibrated")
    except Exception as e:
        logger.error("[%s] scout_calibrate: failed to launch full pipeline: %s", project_id, e)
        progress_cb(1.0, f"Scout done but full pipeline launch failed: {e}")

    return prev_result
