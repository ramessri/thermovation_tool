"""
ArUco marker detection stage.

Runs before feature matching.  Detects ArUco markers in sampled frames,
estimates their 3D positions using solvePnP, and derives a metric scale
factor by comparing the physical marker size to the inter-marker distances
visible in the video.

Scale is stored in the result dict and in the projects table so Phase 2
can apply it automatically — no user confirmation gate.

Each project chooses its scale fiducial (projects.marker_type):
  "aruco" — ArUco markers only (behaviour unchanged)
  "grid"  — the custom 3×3 grid marker sheet only (grid_marker_detector.py)

FAIL FAST: if the chosen marker is not found, the task raises an error and
the pipeline halts immediately (before the expensive COLMAP stages).

Environment variables
---------------------
ARUCO_MARKER_SIZE_M   Physical side length of printed markers in metres (default 0.15)
ARUCO_DICT            ArUco dictionary name (default DICT_4X4_100)
ARUCO_MIN_FRAMES      Minimum number of frames a marker must appear in (default 2)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ARUCO_MARKER_SIZE_M = float(os.environ.get("ARUCO_MARKER_SIZE_M", "0.15"))
ARUCO_MIN_FRAMES    = int(os.environ.get("ARUCO_MIN_FRAMES", "2"))

_DICT_MAP = {
    "DICT_4X4_50":    cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100":   cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250":   cv2.aruco.DICT_4X4_250,
    "DICT_5X5_100":   cv2.aruco.DICT_5X5_100,
    "DICT_6X6_100":   cv2.aruco.DICT_6X6_100,
}

_ARUCO_DICT_NAME = os.environ.get("ARUCO_DICT", "DICT_4X4_100")


def _get_aruco_dict():
    dict_id = _DICT_MAP.get(_ARUCO_DICT_NAME, cv2.aruco.DICT_4X4_100)
    return cv2.aruco.getPredefinedDictionary(dict_id)


def _build_camera_matrix(focal_length_px: float, cx: float, cy: float) -> np.ndarray:
    return np.array([
        [focal_length_px, 0,              cx],
        [0,               focal_length_px, cy],
        [0,               0,              1],
    ], dtype=np.float64)


def _marker_3d_corners(size_m: float = ARUCO_MARKER_SIZE_M) -> np.ndarray:
    """4 corners of a flat marker in its own coordinate frame (z=0)."""
    h = size_m / 2.0
    return np.array([
        [-h,  h, 0],
        [ h,  h, 0],
        [ h, -h, 0],
        [-h, -h, 0],
    ], dtype=np.float64)


def detect_markers_in_frame(
    img: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_size_m: float = ARUCO_MARKER_SIZE_M,
) -> list[dict]:
    """
    Detect ArUco markers in a single BGR frame.

    Returns a list of dicts:
        {
          "id": int,
          "corners_px": [[x,y], ...] (4 points),
          "rvec": [rx, ry, rz],
          "tvec": [tx, ty, tz],   # marker centre in camera space (metres)
        }
    """
    aruco_dict   = _get_aruco_dict()
    params       = cv2.aruco.DetectorParameters()
    detector     = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners_list, ids, _ = detector.detectMarkers(img)

    if ids is None or len(ids) == 0:
        return []

    obj_pts = _marker_3d_corners(marker_size_m)
    results = []
    for corners, marker_id in zip(corners_list, ids.flatten()):
        corners_2d = corners[0]   # shape (4, 2)
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, corners_2d, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            continue
        results.append({
            "id":         int(marker_id),
            "corners_px": corners_2d.tolist(),
            "rvec":       rvec.flatten().tolist(),
            "tvec":       tvec.flatten().tolist(),
        })
    return results


def _estimate_scale_from_detections(
    detections_by_frame: dict[int, list[dict]],
    video_metadata: dict,
) -> Optional[float]:
    """
    Estimate scale factor: physical_distance / sfm_distance.

    Strategy 1 — same-frame baseline: if two markers are visible in the
    same frame, the baseline between their camera-space tvecs is in metres.
    We compare this to the euclidean distance between 2D centroids (pixels)
    projected back via focal length to get the SfM unit ratio.

    For the pre-SfM ArUco stage we store the physical distances between
    marker pairs (in metres) for later comparison with post-SfM 3D distances.
    The scale is finalised in scale_from_aruco.py after SfM runs.

    Here we return None to indicate scale will be resolved post-SfM,
    but log what we found for diagnostics.
    """
    marker_ids: set[int] = set()
    for dets in detections_by_frame.values():
        for d in dets:
            marker_ids.add(d["id"])

    logger.info("ArUco: %d unique markers detected across %d frames",
                len(marker_ids), len(detections_by_frame))

    # Physical baseline between two markers in same frame
    baselines = []
    for frame_idx, dets in detections_by_frame.items():
        if len(dets) < 2:
            continue
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                ti = np.array(dets[i]["tvec"])
                tj = np.array(dets[j]["tvec"])
                dist_m = float(np.linalg.norm(ti - tj))
                baselines.append({
                    "frame": frame_idx,
                    "id_a":  dets[i]["id"],
                    "id_b":  dets[j]["id"],
                    "dist_m": round(dist_m, 4),
                })

    if baselines:
        logger.info("ArUco: %d inter-marker baselines measured (pre-SfM)", len(baselines))

    return baselines  # returned in result for use by scale_from_aruco stage


async def run_aruco_detection(
    project_id: str,
    frame_keys: list[str],
    video_metadata: dict,
    tmp: Path,
    progress_cb: Callable[[float, str], None],
    marker_type: str = "aruco",
) -> dict:
    """
    Detect the project's scale markers (ArUco or 3×3 grid) in extracted frames.

    Returns:
        {
          "aruco_markers":     dict — marker observations by frame,
          "aruco_ids_found":   list[int],
          "aruco_baselines":   list[dict],  # physical distances pre-SfM
          "aruco_marker_size_m": float,
          "aruco_frames_checked": int,
          "aruco_floor_marker_id": int | None,
          "marker_type":       "aruco" | "grid",
          "grid_markers":      dict — grid detections by sampled frame index (grid mode),
          "grid_frames_found": int,
        }

    Raises RuntimeError if the chosen marker is not found (fail fast).
    """
    from backend.core.storage import get_storage
    from backend.workers.pipeline.grid_marker_detector import detect_marker
    storage = get_storage()
    use_grid = marker_type == "grid"
    marker_label = "grid marker" if use_grid else "ArUco markers"

    # Camera matrix from metadata (or reasonable default)
    fl = video_metadata.get("focal_length_px")
    w  = video_metadata.get("width",  1920)
    h  = video_metadata.get("height", 1080)
    cx = video_metadata.get("cx", w / 2.0)
    cy = video_metadata.get("cy", h / 2.0)

    if fl is None:
        # COLMAP-style prior: assume 70% of image diagonal as focal length
        fl = 0.7 * (w ** 2 + h ** 2) ** 0.5
        logger.warning("ArUco: no focal length in metadata, using heuristic %.0fpx", fl)

    dist_coeffs   = np.zeros((4, 1), dtype=np.float64)

    # Sample frames — use every 3rd frame key to keep processing time reasonable
    # while still getting good coverage of the video
    step          = max(1, len(frame_keys) // 150)
    sampled_keys  = frame_keys[::step]
    n_sampled     = len(sampled_keys)
    progress_cb(0.05, f"Scanning {n_sampled} frames for {marker_label}…")

    detections_by_frame: dict[int, list[dict]] = {}
    marker_frame_count: dict[int, int] = {}
    grid_by_frame: dict[str, dict] = {}
    camera_matrix: np.ndarray | None = None

    for i, key in enumerate(sampled_keys):
        if i % 20 == 0:
            progress_cb(0.05 + 0.80 * i / n_sampled,
                        f"{marker_label} scan {i+1}/{n_sampled}…")

        local = tmp / f"aruco_frame_{i:04d}.jpg"
        try:
            await storage.download(key, local)
        except Exception as e:
            logger.debug("ArUco: skip frame %s: %s", key, e)
            continue

        img = cv2.imread(str(local))
        if img is None:
            continue
        local.unlink(missing_ok=True)

        if camera_matrix is None:
            # Extracted frames may have been downscaled (resize_preset) relative
            # to video_metadata's resolution — scale fl/cx/cy to match the actual
            # frame size so solvePnP-derived baselines are correct.
            frame_h, frame_w = img.shape[:2]
            scale = (frame_w / w) if w else 1.0
            camera_matrix = _build_camera_matrix(fl * scale, cx * scale, cy * scale)
            if abs(scale - 1.0) > 1e-3:
                logger.info(
                    "ArUco: scaling camera intrinsics by %.3f (%dx%d frames vs %dx%d metadata)",
                    scale, frame_w, frame_h, w, h,
                )

        if use_grid:
            try:
                grid = detect_marker(img)
            except Exception as e:
                logger.debug("Grid marker: detection error on %s: %s", key, e)
                grid = None
            if grid is not None:
                grid_by_frame[str(i)] = grid
            continue

        dets = detect_markers_in_frame(img, camera_matrix, dist_coeffs)
        if dets:
            detections_by_frame[i] = dets
            for d in dets:
                marker_frame_count[d["id"]] = marker_frame_count.get(d["id"], 0) + 1

    progress_cb(0.85, "Analysing detections…")

    if use_grid:
        logger.info("Grid marker: found in %d/%d sampled frames", len(grid_by_frame), n_sampled)
        if len(grid_by_frame) < ARUCO_MIN_FRAMES:
            raise RuntimeError(
                f"Grid marker found in {len(grid_by_frame)} frame(s); need {ARUCO_MIN_FRAMES}+. "
                "Place the printed 3×3 grid marker sheet (28.6 × 20.2 cm) flat in the scene and "
                "keep it clearly visible from several viewpoints. "
                "(Or create the project with ArUco markers selected.)"
            )
        progress_cb(1.0, f"Grid marker found in {len(grid_by_frame)} frames")
        return {
            "marker_type":           "grid",
            "aruco_markers":         {},
            "aruco_ids_found":       [],
            "aruco_baselines":       [],
            "aruco_marker_size_m":   ARUCO_MARKER_SIZE_M,
            "aruco_frames_checked":  n_sampled,
            "aruco_floor_marker_id": None,
            "aruco_sampled_keys":    sampled_keys,
            "grid_markers":          grid_by_frame,
            "grid_frames_found":     len(grid_by_frame),
        }

    # Filter: only keep markers seen in ARUCO_MIN_FRAMES+ frames
    reliable_ids = {mid for mid, cnt in marker_frame_count.items()
                    if cnt >= ARUCO_MIN_FRAMES}

    all_ids = set(marker_frame_count.keys())
    logger.info(
        "ArUco: %d markers total, %d reliable (>= %d frames): %s",
        len(all_ids), len(reliable_ids), ARUCO_MIN_FRAMES, sorted(reliable_ids)
    )

    if not all_ids:
        raise RuntimeError(
            "No ArUco markers detected in any frame. "
            "Print ArUco markers (DICT_4X4_100) and place them in the scene before scanning. "
            f"Marker side length must be set via ARUCO_MARKER_SIZE_M (currently {ARUCO_MARKER_SIZE_M}m)."
        )

    if not reliable_ids:
        ids_str = ", ".join(str(i) for i in sorted(all_ids))
        raise RuntimeError(
            f"ArUco markers detected (IDs: {ids_str}) but none appeared in {ARUCO_MIN_FRAMES}+ frames. "
            "Ensure markers are clearly visible throughout the scan and large enough to detect from scanning distance."
        )

    # Compute baselines between reliable markers in same frame
    reliable_detections = {
        fi: [d for d in dets if d["id"] in reliable_ids]
        for fi, dets in detections_by_frame.items()
        if any(d["id"] in reliable_ids for d in dets)
    }
    baselines = _estimate_scale_from_detections(reliable_detections, video_metadata)

    # Heuristic: the marker with the lowest ID on the floor is conventional.
    # Users can override via ARUCO_FLOOR_MARKER_ID env var.
    env_floor = os.environ.get("ARUCO_FLOOR_MARKER_ID")
    floor_marker_id: Optional[int] = int(env_floor) if env_floor else (
        min(reliable_ids) if reliable_ids else None
    )
    logger.info("ArUco: floor marker ID assumed to be %s", floor_marker_id)

    # Serialise detections — store only reliable markers.
    # Include frame_key so scale_from_aruco can match detections to SfM cameras.
    serialisable = {
        str(fi): [d for d in dets if d["id"] in reliable_ids]
        for fi, dets in detections_by_frame.items()
        if any(d["id"] in reliable_ids for d in dets)
    }

    progress_cb(1.0, f"ArUco: {len(reliable_ids)} markers found — IDs {sorted(reliable_ids)}")

    return {
        "marker_type":          "aruco",
        "aruco_markers":        serialisable,
        "aruco_ids_found":      sorted(reliable_ids),
        "aruco_baselines":      baselines if baselines else [],
        "aruco_marker_size_m":  ARUCO_MARKER_SIZE_M,
        "aruco_frames_checked": n_sampled,
        "aruco_floor_marker_id": floor_marker_id,
        # Keep the ordered list of sampled frame keys so scale_from_aruco
        # can resolve frame index → filename → SfM camera pose.
        "aruco_sampled_keys":   sampled_keys,
    }
