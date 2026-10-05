"""
Post-SfM ArUco scale derivation.

After COLMAP SfM runs, the sparse point cloud is in arbitrary units.
This stage:
  1. Matches each ArUco frame detection to its SfM-registered camera pose.
  2. For markers seen in 2+ views, triangulates their 3D position in SfM units.
  3. Computes inter-marker SfM distance vs physical distance (from solvePnP baselines).
  4. Derives scale factor: physical_m / sfm_units.
  5. Also estimates gravity alignment from floor marker normal.

The result is stored in projects.confirmed_scale_factor (source = "aruco").
No user gate — scale is automatic.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Camera pose helpers ───────────────────────────────────────────────────────

def _load_cameras(cameras_json: dict) -> dict[str, dict]:
    """
    Parse cameras.json produced by sfm.py.
    Returns {image_name: {"K": np.ndarray 3x3, "R": np.ndarray 3x3, "t": np.ndarray 3}}
    """
    cameras = {}
    for img in cameras_json.get("images", []):
        name = img.get("name", "")
        K    = img.get("K")
        cfw  = img.get("cam_from_world", {})

        # Reconstruct K from camera params if not stored directly
        if K is None:
            # Look up camera params from the cameras list
            cam_id = img.get("camera_id")
            for cam in cameras_json.get("cameras", []):
                if cam.get("camera_id") == cam_id:
                    params = cam.get("params", [])
                    if len(params) >= 3:
                        # SIMPLE_RADIAL: [f, cx, cy, k1]
                        f, cx_v, cy_v = float(params[0]), float(params[1]), float(params[2])
                        K = [[f, 0, cx_v], [0, f, cy_v], [0, 0, 1]]
                    break

        if isinstance(cfw, dict):
            mat = cfw.get("matrix_3x4")
        else:
            mat = cfw
        if K is None or mat is None:
            continue
        K_arr   = np.array(K, dtype=np.float64).reshape(3, 3)
        mat_arr = np.array(mat, dtype=np.float64).reshape(3, 4)
        R = mat_arr[:, :3]
        t = mat_arr[:, 3]
        cameras[name] = {"K": K_arr, "R": R, "t": t}
    return cameras


def _backproject_marker_centre(
    corners_px: list[list[float]],
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Back-project the 2D centre of a marker into a world-space ray.
    Returns (cam_centre_world, ray_direction_world).
    """
    pts = np.array(corners_px, dtype=np.float64)
    cx_px, cy_px = pts.mean(axis=0)

    fx, fy = K[0, 0], K[1, 1]
    ppx, ppy = K[0, 2], K[1, 2]
    ray_cam = np.array([
        (cx_px - ppx) / fx,
        (cy_px - ppy) / fy,
        1.0,
    ])
    ray_cam /= np.linalg.norm(ray_cam)

    R_t = R.T
    cam_centre = -R_t @ t
    ray_world  = R_t @ ray_cam
    ray_world /= np.linalg.norm(ray_world)

    return cam_centre, ray_world


def _triangulate_marker(
    observations: list[tuple[np.ndarray, np.ndarray]],
) -> Optional[np.ndarray]:
    """
    Linear triangulation of a 3D point from multiple camera rays.
    observations: list of (cam_centre, ray_direction) pairs.
    Returns 3D point or None if underdetermined.
    """
    if len(observations) < 2:
        return None

    # Collect (I - d d^T) rows for least-squares solve
    A_rows: list[np.ndarray] = []
    b_rows: list[np.ndarray] = []
    for origin, direction in observations:
        d = direction
        M = np.eye(3) - np.outer(d, d)
        A_rows.append(M)
        b_rows.append(M @ origin)

    A_stack = np.vstack(A_rows)
    b_stack = np.concatenate(b_rows)

    try:
        pt, _, _, _ = np.linalg.lstsq(A_stack, b_stack, rcond=None)
        return pt
    except Exception:
        return None


# ── Scale derivation ──────────────────────────────────────────────────────────

def derive_scale_factor(
    aruco_result: dict,
    cameras_json: dict,
    exclude_ids: set[int] | None = None,
) -> tuple[Optional[float], dict]:
    """
    Derive a metric scale factor (metres per SfM unit) by:
      1. Matching ArUco frame detections to SfM camera poses.
      2. Triangulating each marker's 3D position from multiple views.
      3. Computing physical_dist / sfm_dist for pairs with known physical baseline.

    exclude_ids: marker IDs to ignore entirely (e.g. duplicates in the scene).

    Returns (scale_factor_m_per_sfm_unit, diagnostics_dict).
    scale_factor is None if derivation fails.
    """
    exclude_ids   = set(exclude_ids or [])
    cameras       = _load_cameras(cameras_json)
    aruco_markers = aruco_result.get("aruco_markers", {})
    baselines     = aruco_result.get("aruco_baselines", [])
    marker_size_m = aruco_result.get("aruco_marker_size_m", 0.15)
    ids_found     = [i for i in aruco_result.get("aruco_ids_found", []) if i not in exclude_ids]
    sampled_keys  = aruco_result.get("aruco_sampled_keys", [])

    if exclude_ids:
        logger.info("scale_from_aruco: excluding marker IDs %s from scale derivation", sorted(exclude_ids))

    diagnostics = {
        "strategy":       "triangulation_then_baselines",
        "n_markers":      len(ids_found),
        "n_sfm_cameras":  len(cameras),
        "marker_size_m":  marker_size_m,
        "n_baselines":    len(baselines),
        "excluded_ids":   sorted(exclude_ids),
    }

    if not cameras:
        logger.warning("scale_from_aruco: no cameras from SfM")
        return None, {**diagnostics, "error": "no SfM cameras"}

    # ── Step 1: match frame detections to SfM cameras ────────────────────────
    # Prefer aruco_markers_sfm (indexed directly by filename from the post-SfM
    # re-scan stage — guaranteed 100% match rate).  Fall back to the pre-SfM
    # aruco_markers (indexed by sample position) when the post-SfM stage didn't
    # run or found nothing.
    marker_observations: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    matched_frames = 0

    markers_sfm = aruco_result.get("aruco_markers_sfm", {})   # {filename: [dets]}
    if markers_sfm:
        # Fast path: filename keys always match cameras dict
        baselines = aruco_result.get("aruco_baselines_sfm") or baselines
        for fname, dets in markers_sfm.items():
            cam = cameras.get(fname)
            if cam is None:
                continue
            matched_frames += 1
            for d in dets:
                mid = d["id"]
                corners = d["corners_px"]
                origin, ray = _backproject_marker_centre(corners, cam["K"], cam["R"], cam["t"])
                if mid in exclude_ids:
                    continue
                if mid not in marker_observations:
                    marker_observations[mid] = []
                marker_observations[mid].append((origin, ray))
        diagnostics["source"] = "post_sfm_rescan"
    else:
        # Legacy path: pre-SfM markers indexed by sample position
        for frame_idx_str, dets in aruco_markers.items():
            fi = int(frame_idx_str)
            frame_key = sampled_keys[fi] if fi < len(sampled_keys) else None
            if not frame_key:
                continue
            fname = Path(frame_key).name
            cam   = cameras.get(fname)
            if cam is None:
                continue
            matched_frames += 1
            for d in dets:
                mid = d["id"]
                corners = d["corners_px"]
                origin, ray = _backproject_marker_centre(corners, cam["K"], cam["R"], cam["t"])
                if mid in exclude_ids:
                    continue
                if mid not in marker_observations:
                    marker_observations[mid] = []
                marker_observations[mid].append((origin, ray))
        diagnostics["source"] = "pre_sfm_fallback"

    diagnostics["matched_frames"] = matched_frames
    diagnostics["markers_with_observations"] = len(marker_observations)

    # ── Step 2: triangulate each marker ──────────────────────────────────────
    marker_world_pos: dict[int, np.ndarray] = {}
    for mid, obs in marker_observations.items():
        pt = _triangulate_marker(obs)
        if pt is not None:
            marker_world_pos[mid] = pt
            logger.debug("scale_from_aruco: triangulated marker %d from %d views → %s",
                         mid, len(obs), pt.round(4))

    diagnostics["triangulated_markers"] = list(marker_world_pos.keys())
    diagnostics["_marker_world_pos"]    = marker_world_pos   # internal; used by plane fitting

    if not marker_world_pos:
        logger.warning(
            "scale_from_aruco: could not triangulate any markers "
            "(matched %d frames, need ≥2 views per marker). "
            "Falling back to solvePnP baselines if available.",
            matched_frames,
        )
        return _scale_from_baselines_only(baselines, diagnostics)

    # ── Step 3: scale from triangulated pair distances + physical baselines ───
    # Minimum inter-marker distance to trust a solvePnP baseline.
    # Below this, depth estimates are unreliable (markers close together,
    # or photographed from a shallow angle).
    MIN_BASELINE_M = 0.30

    # Build a lookup from sorted marker-ID pair → physical distance so we
    # avoid an O(n³) search (n_markers² × n_baselines) in the pair loop.
    baseline_lookup: dict[tuple, float] = {
        (min(b["id_a"], b["id_b"]), max(b["id_a"], b["id_b"])): float(b["dist_m"])
        for b in baselines
    }

    scale_estimates: list[float] = []
    marker_ids = list(marker_world_pos.keys())

    for i in range(len(marker_ids)):
        for j in range(i + 1, len(marker_ids)):
            id_a, id_b = marker_ids[i], marker_ids[j]
            pair_key = (min(id_a, id_b), max(id_a, id_b))
            physical_m = baseline_lookup.get(pair_key)
            if physical_m is None:
                continue

            sfm_dist = float(np.linalg.norm(
                marker_world_pos[id_a] - marker_world_pos[id_b]
            ))
            if sfm_dist < 1e-9:
                continue

            if physical_m < MIN_BASELINE_M:
                logger.debug(
                    "scale_from_aruco: skip (%d,%d) phys=%.4fm < %.2fm min",
                    id_a, id_b, physical_m, MIN_BASELINE_M,
                )
                continue

            est = physical_m / sfm_dist
            scale_estimates.append(est)
            logger.info(
                "scale_from_aruco: markers (%d, %d) — sfm=%.4f, phys=%.4fm → scale=%.5f",
                id_a, id_b, sfm_dist, physical_m, est,
            )

    diagnostics["n_scale_estimates"] = len(scale_estimates)

    if not scale_estimates:
        # Triangulated markers but no matching baselines — try single-marker solvePnP fallback
        logger.warning(
            "scale_from_aruco: triangulated %d marker(s) but no matching baselines. "
            "Need 2+ markers visible in the same frame to establish a scale baseline.",
            len(marker_world_pos),
        )
        return _scale_from_baselines_only(baselines, diagnostics)

    # Reject outliers via 1.5× IQR before averaging
    estimates = np.array(scale_estimates)
    if len(estimates) >= 4:
        q1, q3 = np.percentile(estimates, [25, 75])
        iqr     = q3 - q1
        mask    = (estimates >= q1 - 1.5 * iqr) & (estimates <= q3 + 1.5 * iqr)
        estimates = estimates[mask]
        diagnostics["scale_outliers_removed"] = int((~mask).sum())

    scale_factor = float(np.median(estimates))
    diagnostics["scale_factor"]    = round(scale_factor, 6)
    diagnostics["scale_std"]       = round(float(np.std(estimates)), 6)
    diagnostics["scale_n_inliers"] = int(len(estimates))

    logger.info(
        "scale_from_aruco: scale=%.6f m/unit (from %d estimate(s), std=%.6f)",
        scale_factor, len(estimates), float(np.std(estimates)),
    )
    return scale_factor, diagnostics


def fit_planes_from_markers(
    marker_world_pos: dict,
    aruco_result: dict,
    cameras_json: dict,
    gravity_down: np.ndarray | None = None,
    n_floor: int = 3,
    n_ceiling: int = 3,
    scale_factor: float = 1.0,
) -> list[dict]:
    """
    Fit a floor plane from the lowest ArUco marker positions.

    Only the floor plane is returned. Ceiling and wall fitting from marker
    positions is not reliable: markers don't tell you which surface they're
    attached to, so two markers on perpendicular walls produce a diagonal
    plane, and floor markers with any height variation pass the ceiling-gap
    check and get mislabelled. Walls/ceiling should be detected by RANSAC on
    the actual point cloud (a separate future stage).

    Returns list with at most one entry: {"label": "floor", ...}.
    """
    if not marker_world_pos:
        return []
    n_floor = min(n_floor, len(marker_world_pos))

    if gravity_down is None:
        gravity_down = np.array([0., 1., 0.])
        cameras = _load_cameras(cameras_json)
        if cameras:
            g = np.zeros(3)
            for cam in cameras.values():
                g += cam["R"].T @ np.array([0., 1., 0.])
            if np.linalg.norm(g) > 1e-6:
                gravity_down = g / np.linalg.norm(g)

    gravity_up = -gravity_down

    # Take the n_floor lowest markers as the floor sample
    marker_ids = list(marker_world_pos.keys())
    pts_arr    = np.array([marker_world_pos[m] for m in marker_ids])
    heights    = pts_arr @ gravity_up
    sorted_idx = np.argsort(heights)
    floor_ids  = [marker_ids[i] for i in sorted_idx[:n_floor]]

    floor_pts    = np.array([marker_world_pos[m] for m in floor_ids])
    floor_normal = gravity_up.copy()
    floor_offset = float(np.dot(floor_normal, floor_pts.mean(axis=0)))

    logger.info(
        "scale_from_aruco: floor plane from %d lowest markers %s, height=%.4f SfM-units",
        len(floor_ids), sorted(floor_ids), floor_offset,
    )

    return [{
        "label":      "floor",
        "normal":     floor_normal.tolist(),
        "offset":     round(floor_offset, 4),
        "marker_ids": sorted(floor_ids),
        "n_markers":  len(floor_ids),
    }]


def _scale_from_baselines_only(
    baselines: list[dict],
    diagnostics: dict,
) -> tuple[Optional[float], dict]:
    """
    Fallback: we have solvePnP baselines (metric) but couldn't triangulate SfM
    positions. Return None so apply_scale can skip rather than use a wrong scale.
    """
    if baselines:
        avg = float(np.mean([b["dist_m"] for b in baselines]))
        diagnostics["avg_baseline_m"] = round(avg, 4)
    diagnostics["error"] = (
        "Cannot derive scale without matching ArUco detections to SfM cameras. "
        "Ensure at least 2 markers visible in frames that were successfully registered by SfM."
    )
    logger.warning("scale_from_aruco: %s", diagnostics["error"])
    return None, diagnostics


# ── Gravity alignment ─────────────────────────────────────────────────────────

def gravity_vector_from_floor_marker(
    aruco_result: dict,
    cameras_json: dict,
) -> Optional[np.ndarray]:
    """
    Estimate the gravity direction (world -Y) from the floor marker normal.

    The floor marker lies flat on the floor; its plane normal points upward.
    We recover this from the rvec returned by solvePnP for the floor marker,
    rotate to world space using the SfM camera pose, and average across views.

    Returns a unit 3-vector pointing UP, or None if insufficient data.
    """
    import cv2 as _cv2

    floor_id      = aruco_result.get("aruco_floor_marker_id")
    aruco_markers = aruco_result.get("aruco_markers", {})
    sampled_keys  = aruco_result.get("aruco_sampled_keys", [])
    cameras       = _load_cameras(cameras_json)

    if floor_id is None or not cameras:
        return None

    normals_world: list[np.ndarray] = []

    for frame_idx_str, dets in aruco_markers.items():
        floor_dets = [d for d in dets if d["id"] == floor_id]
        if not floor_dets:
            continue

        # Match to SfM camera
        fi        = int(frame_idx_str)
        frame_key = sampled_keys[fi] if fi < len(sampled_keys) else None
        if not frame_key:
            continue
        cam = cameras.get(Path(frame_key).name)
        if cam is None:
            continue

        d    = floor_dets[0]
        rvec = np.array(d["rvec"], dtype=np.float64).reshape(3, 1)
        R_marker, _ = _cv2.Rodrigues(rvec)
        # Marker z-axis (plane normal) in camera space
        normal_cam = R_marker[:, 2]
        if normal_cam[2] < 0:
            normal_cam = -normal_cam

        # Rotate to world space
        normal_world = cam["R"].T @ normal_cam
        normals_world.append(normal_world)

    if not normals_world:
        return None

    avg_normal = np.mean(normals_world, axis=0)
    mag = np.linalg.norm(avg_normal)
    if mag < 1e-6:
        return None
    return avg_normal / mag


# ── Public stage entry point ──────────────────────────────────────────────────

async def run_scale_from_aruco(
    project_id: str,
    prev_result: dict,
    tmp: Path,
    progress_cb: Callable[[float, str], None],
) -> dict:
    # Marker IDs to exclude from scale derivation (e.g. duplicates in the scene).
    # Passed via prev_result["exclude_marker_ids"] = [7, 12, ...]
    _exclude_ids: set[int] = set(prev_result.get("exclude_marker_ids") or [])
    """
    Post-SfM scale derivation from ArUco markers and/or the 3×3 grid marker.

    Uses the ArUco detections from the detect_aruco stage and the
    COLMAP camera poses to derive a metric scale factor.

    Returns prev_result augmented with:
        confirmed_scale_factor  float | None
        confirmed_scale_source  "aruco" | "grid_marker"
        gravity_up_world        [x,y,z] | None
        scale_diagnostics       dict
    """
    from backend.core.storage import get_storage

    storage = get_storage()
    aruco_result     = prev_result.get("aruco_result", {})
    camera_poses_key = prev_result.get("camera_poses_key",
                                       f"{project_id}/sfm/cameras.json")

    progress_cb(0.1, "Loading SfM camera poses…")
    cameras_local = tmp / "cameras_aruco.json"
    try:
        await storage.download(camera_poses_key, cameras_local)
        cameras_json = json.loads(cameras_local.read_text())
    except Exception as e:
        logger.warning("[%s] scale_from_aruco: cannot load cameras.json: %s", project_id, e)
        result = dict(prev_result)
        result.update({
            "confirmed_scale_factor": None,
            "confirmed_scale_source": "grid_marker" if aruco_result.get("marker_type") == "grid" else "aruco",
            "gravity_up_world":       None,
            "scale_diagnostics":      {"error": str(e)},
        })
        return result

    is_grid = aruco_result.get("marker_type") == "grid"
    scale_source = "grid_marker" if is_grid else "aruco"

    if is_grid:
        from backend.workers.pipeline.grid_marker_scale import derive_grid_scale
        grid_dets = aruco_result.get("grid_markers_sfm") or {}
        progress_cb(0.3, f"Triangulating grid marker from {len(grid_dets)} frame(s)…")
        try:
            scale_factor, diagnostics = derive_grid_scale(grid_dets, cameras_json)
        except Exception as e:
            logger.warning("[%s] scale_from_aruco: grid scale failed: %s", project_id, e)
            scale_factor, diagnostics = None, {"error": str(e)}
        diagnostics["strategy"] = "grid_marker_similarity_fit"
    else:
        n_sfm_markers = len(aruco_result.get("aruco_markers_sfm", {}))
        source_note = f" (post-SfM: {n_sfm_markers} registered frames)" if n_sfm_markers else " (pre-SfM fallback)"
        progress_cb(0.3, f"Triangulating ArUco markers from SfM cameras{source_note}…")
        scale_factor, diagnostics = derive_scale_factor(aruco_result, cameras_json, exclude_ids=_exclude_ids)
    diagnostics["scale_source"] = scale_source

    # Sanitize: numpy returns nan for mean/std of empty arrays — treat as None
    import math as _math
    if scale_factor is not None and (not isinstance(scale_factor, (int, float)) or _math.isnan(scale_factor)):
        scale_factor = None

    progress_cb(0.6, "Computing gravity alignment from floor marker…")
    gravity_up = gravity_vector_from_floor_marker(aruco_result, cameras_json)

    progress_cb(0.75, "Fitting planes from coplanar markers…")
    try:
        marker_world_pos = diagnostics.get("_marker_world_pos", {})
        fitted_planes = fit_planes_from_markers(
            marker_world_pos, aruco_result, cameras_json,
            scale_factor=scale_factor or 1.0,
        )
        logger.info("[%s] scale_from_aruco: %d planes fitted from markers",
                    project_id, len(fitted_planes))
    except Exception as e:
        logger.warning("[%s] scale_from_aruco: plane fitting failed: %s", project_id, e)
        fitted_planes = []

    if scale_factor is None:
        logger.warning(
            "[%s] scale_from_aruco: scale not derivable — apply_scale will pass through unscaled cloud",
            project_id,
        )

    if gravity_up is not None:
        gl = gravity_up.tolist()
        gravity_list = gl if all(not _math.isnan(v) for v in gl) else None
    else:
        gravity_list = None

    # Persist scale and gravity in a single DB roundtrip
    if scale_factor is not None or gravity_list is not None:
        try:
            import psycopg2, json as _json
            from backend.core.config import settings as _s
            conn = psycopg2.connect(
                _s.DATABASE_URL.replace("postgresql+asyncpg://", "postgres://")
            )
            cur = conn.cursor()
            if scale_factor is not None:
                cur.execute(
                    "UPDATE projects SET confirmed_scale_factor=%s, confirmed_scale_source=%s WHERE id=%s",
                    (scale_factor, scale_source, project_id),
                )
                logger.info("[%s] scale_from_aruco: scale=%.6f persisted", project_id, scale_factor)
            if gravity_list is not None:
                cur.execute(
                    "UPDATE projects SET gravity_up_world=%s WHERE id=%s",
                    (_json.dumps(gravity_list), project_id),
                )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.warning("[%s] scale_from_aruco: DB persist failed: %s", project_id, e)

    scale_str = f"{scale_factor:.6f} m/unit" if scale_factor else "not derived"
    planes_str = f"{len(fitted_planes)} plane(s)" if fitted_planes else "none"
    progress_cb(1.0, f"Scale: {scale_str} · Planes: {planes_str}")

    # Strip internal key before serialising
    diagnostics.pop("_marker_world_pos", None)

    result = dict(prev_result)
    result.update({
        "confirmed_scale_factor": scale_factor,
        "confirmed_scale_source": scale_source,
        "gravity_up_world":       gravity_list,
        "scale_diagnostics":      diagnostics,
        "fitted_planes":          fitted_planes,
    })
    return result
