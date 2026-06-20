"""
Stage 0: Extract video metadata.

Downloads the uploaded video from storage, runs ffprobe + exiftool to
extract camera intrinsics, rotation, GPS, and stable frame timestamps.

Stores the profile in projects.video_metadata.

Returns:
    {
      "storage_key":    str   — original upload storage key (threaded through for extract_frames)
      "video_metadata": dict  — full profile from ml/metadata/extractor.py
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


async def run_extract_metadata(
    project_id: str,
    storage_key: str,
    tmp: Path,
    progress_cb: Callable[[float, str], None],
) -> dict:
    from backend.core.storage import get_storage
    from ml.metadata.extractor import extract_video_metadata

    storage = get_storage()

    progress_cb(0.0, "Downloading video for metadata extraction…")
    video_path = tmp / "input_video.mp4"
    await storage.download(storage_key, video_path)

    progress_cb(0.3, "Extracting video metadata (ffprobe + exiftool)…")
    profile = extract_video_metadata(video_path)

    # ── Minimum resolution gate (FAIL FAST) ───────────────────────────────────
    w, h = profile.get("width", 0), profile.get("height", 0)
    if w and h and min(w, h) < 1080:
        raise ValueError(
            f"Video resolution {w}x{h} is below the 1080p minimum required for "
            f"reliable feature matching and ArUco marker detection. "
            f"Please re-shoot at 1080p or higher."
        )

    # ── Inject calibration photo focal length (overrides video EXIF / heuristic) ──
    try:
        import psycopg2 as _pg2
        from backend.core.config import settings as _settings
        _conn = _pg2.connect(
            _settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgres://")
        )
        _cur = _conn.cursor()
        _cur.execute("SELECT calibration_data FROM projects WHERE id = %s", (project_id,))
        row = _cur.fetchone()
        _cur.close()
        _conn.close()
        calib = row[0] if row and row[0] else None
    except Exception as _e:
        logger.warning("[%s] extract_metadata: could not read calibration_data: %s", project_id, _e)
        calib = None

    if calib:
        from backend.workers.pipeline.calibration import focal_length_for_video
        video_w = profile.get("width", 0)
        if video_w > 0:
            fl_from_calib = focal_length_for_video(calib, video_w)
            old_fl = profile.get("focal_length_px")
            profile["focal_length_px"]     = fl_from_calib
            profile["focal_length_source"] = "calibration_photo"
            logger.info(
                "[%s] calibration photo override: fl_px %.1f → %.1f "
                "(was '%s', device: %s %s)",
                project_id, old_fl or 0, fl_from_calib,
                "heuristic" if old_fl is None else "video_exif",
                calib.get("make", ""), calib.get("model", ""),
            )
        else:
            logger.warning("[%s] extract_metadata: video width unknown, skipping calibration override", project_id)
    else:
        # Mark the source so SfM logs are clear
        if profile.get("focal_length_px"):
            profile.setdefault("focal_length_source", "video_exif")
        else:
            profile.setdefault("focal_length_source", "heuristic")

    # Log key fields for diagnostics
    fl = profile.get("focal_length_px")
    rot = profile.get("rotation_deg", 0)
    w, h = profile.get("width", 0), profile.get("height", 0)
    make = profile.get("make", "")
    model = profile.get("model", "")
    hint = profile.get("scene_hint", "unknown")
    n_ts = len(profile.get("stable_frame_timestamps", []))
    fl_src = profile.get("focal_length_source", "unknown")

    logger.info(
        "[%s] metadata: %dx%d rot=%d° fl=%s (%s) make=%s %s hint=%s stable_ts=%d",
        project_id, w, h, rot,
        f"{fl:.0f}px" if fl else "unknown",
        fl_src, make, model, hint, n_ts,
    )
    parts = []
    if w and h:
        parts.append(f"{w}×{h}")
    if fl:
        parts.append(f"fl={fl:.0f}px ({fl_src})")
    else:
        parts.append(f"fl=unknown ({fl_src})")
    if make or model:
        parts.append(" ".join(filter(None, [make, model])))
    if rot:
        parts.append(f"rotation {rot}°")
    if n_ts:
        parts.append(f"{n_ts} stable frames")
    progress_cb(0.7, "Metadata: " + " · ".join(parts))

    # Persist to DB
    try:
        import psycopg2
        from backend.core.config import settings
        conn = psycopg2.connect(
            settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgres://")
        )
        cur = conn.cursor()
        cur.execute(
            "UPDATE projects SET video_metadata = %s WHERE id = %s",
            (json.dumps(profile), project_id),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning("[%s] extract_metadata: DB persist failed: %s", project_id, e)

    progress_cb(1.0, "Metadata extraction complete")

    return {
        "storage_key":    storage_key,
        "video_metadata": profile,
    }
