"""
Post-SfM ArUco re-scan.

Runs on only the SfM-registered frames, guaranteeing every detection can be
triangulated.  The pre-SfM detect_aruco stage provides the fail-fast check;
this stage provides the metric data for scale_from_aruco.

The result is stored as aruco_markers_sfm (indexed by image filename) and
merged back into the existing aruco_result so scale_from_aruco sees it.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from backend.core.storage import get_storage
from backend.workers.pipeline.aruco_detector import (
    ARUCO_MARKER_SIZE_M,
    _build_camera_matrix,
    detect_markers_in_frame,
)

logger = logging.getLogger(__name__)


async def run_aruco_sfm(
    project_id: str,
    prev_result: dict,
    tmp: Path,
    progress_cb: Callable[[float, str], None],
) -> dict:
    """
    Re-detect ArUco markers in SfM-registered frames only.

    Returns prev_result updated with:
        aruco_markers_sfm   — dict[filename, list[detection]]  (always matches cameras.json)
        aruco_baselines_sfm — baselines recomputed from registered-frame detections
    """
    storage = get_storage()

    camera_poses_key = prev_result.get("camera_poses_key",
                                        f"{project_id}/sfm/cameras.json")
    progress_cb(0.05, "Loading SfM camera list…")

    cameras_local = tmp / "aruco_sfm_cameras.json"
    try:
        await storage.download(camera_poses_key, cameras_local)
        cameras_json = json.loads(cameras_local.read_text())
    except Exception as e:
        logger.warning("[%s] aruco_sfm: cannot load cameras.json: %s — skipping", project_id, e)
        return prev_result

    # Build map: filename → camera intrinsics for per-frame focal length
    cameras_by_name = {img["name"]: img for img in cameras_json.get("images", [])}
    registered_names = list(cameras_by_name.keys())
    n = len(registered_names)

    if n == 0:
        logger.warning("[%s] aruco_sfm: no registered cameras — skipping", project_id)
        return prev_result

    # Build map: filename → storage key using image_keys from prev_result
    image_keys: list[str] = prev_result.get("image_keys", prev_result.get("frame_keys", []))
    key_by_name: dict[str, str] = {Path(k).name: k for k in image_keys}

    aruco_result = prev_result.get("aruco_result", {})

    # Always use COLMAP camera intrinsics from cameras.json — these are the actual
    # values pycolmap solved for, guaranteed accurate regardless of video metadata.
    # SIMPLE_RADIAL params = [f, cx, cy, k1]; PINHOLE params = [fx, fy, cx, cy]
    colmap_cams = cameras_json.get("cameras", [])
    fl, cx, cy, w, h = None, None, None, 1920, 1080
    if colmap_cams:
        cam0 = colmap_cams[0]
        w  = int(cam0.get("width",  1920))
        h  = int(cam0.get("height", 1080))
        params = cam0.get("params", [])
        model  = cam0.get("model", "SIMPLE_RADIAL")
        if model in ("SIMPLE_RADIAL", "SIMPLE_PINHOLE") and len(params) >= 3:
            fl, cx, cy = float(params[0]), float(params[1]), float(params[2])
        elif model in ("RADIAL", "PINHOLE") and len(params) >= 4:
            fl, cx, cy = float(params[0]), float(params[2]), float(params[3])
        elif len(params) >= 1:
            fl = float(params[0])
            cx, cy = w / 2.0, h / 2.0

    if fl is None or fl <= 0:
        # Last resort: video metadata or heuristic
        vm = prev_result.get("video_metadata", {})
        fl = vm.get("focal_length_px") or 0.7 * (w ** 2 + h ** 2) ** 0.5
        cx = vm.get("cx", w / 2.0)
        cy = vm.get("cy", h / 2.0)

    logger.info("[%s] aruco_sfm: camera intrinsics fl=%.1f cx=%.1f cy=%.1f (%dx%d)",
                project_id, fl, cx, cy, w, h)

    camera_matrix = _build_camera_matrix(fl, cx, cy)
    dist_coeffs   = np.zeros((4, 1), dtype=np.float64)
    marker_size_m = aruco_result.get("aruco_marker_size_m", ARUCO_MARKER_SIZE_M)
    reliable_ids  = set(aruco_result.get("aruco_ids_found", []))

    frames_dir = tmp / "aruco_sfm_frames"
    frames_dir.mkdir(exist_ok=True)

    detections_by_name: dict[str, list[dict]] = {}
    n_detected = 0

    progress_cb(0.10, f"Scanning {n} registered frames for ArUco markers…")

    for i, fname in enumerate(registered_names):
        storage_key = key_by_name.get(fname)
        if not storage_key:
            continue

        local = frames_dir / fname
        try:
            if not local.exists():
                await storage.download(storage_key, local)
        except Exception as e:
            logger.debug("[%s] aruco_sfm: skip %s: %s", project_id, fname, e)
            continue

        img = cv2.imread(str(local))
        if img is None:
            continue

        dets = detect_markers_in_frame(img, camera_matrix, dist_coeffs, marker_size_m)
        # Keep all detected markers — don't filter by reliable_ids here so we
        # capture markers that may not have met the pre-SfM min-frames threshold
        if dets:
            detections_by_name[fname] = dets
            n_detected += len(dets)

        if i % 30 == 0 or i == n - 1:
            progress_cb(0.10 + 0.75 * (i + 1) / n,
                        f"ArUco SfM scan: {i+1}/{n} frames ({len(detections_by_name)} with markers)…")

    # Recompute baselines from registered-frame detections
    baselines_sfm: list[dict] = []
    for fname, dets in detections_by_name.items():
        if len(dets) < 2:
            continue
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                ti = np.array(dets[i]["tvec"])
                tj = np.array(dets[j]["tvec"])
                dist_m = float(np.linalg.norm(ti - tj))
                baselines_sfm.append({
                    "frame": fname,
                    "id_a":  dets[i]["id"],
                    "id_b":  dets[j]["id"],
                    "dist_m": round(dist_m, 4),
                })

    ids_sfm = sorted({d["id"] for dets in detections_by_name.values() for d in dets})
    logger.info(
        "[%s] aruco_sfm: %d registered frames with markers, IDs=%s, %d baselines",
        project_id, len(detections_by_name), ids_sfm, len(baselines_sfm),
    )

    progress_cb(1.0, f"ArUco SfM scan: {len(detections_by_name)} frames, "
                     f"{len(ids_sfm)} IDs, {len(baselines_sfm)} baselines")

    # Merge into result: preserve original aruco_result, add sfm-specific fields
    updated_aruco = dict(aruco_result)
    updated_aruco["aruco_markers_sfm"]   = detections_by_name   # indexed by filename
    updated_aruco["aruco_baselines_sfm"] = baselines_sfm
    updated_aruco["aruco_ids_sfm"]       = ids_sfm

    result = dict(prev_result)
    result["aruco_result"] = updated_aruco
    return result
