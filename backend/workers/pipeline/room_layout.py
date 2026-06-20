"""
Room layout reconstruction stage.

Derives the room's architectural planes (floor, walls, ceiling) using gravity
direction estimated from camera poses, then projects noisy wall/ceiling points
onto their fitted planes to produce clean flat surfaces.

Key assumptions (per user spec):
  - Videos are always shot right-side up — the camera Y axis (OpenCV: pointing
    down) consistently maps to the gravity-down direction across all frames.
  - Floors are horizontal, walls are perpendicular to the floor.
  - There is one primary floor (step edges treated as tolerance, not separate planes).

Algorithm:
  1. Compute gravity_down from the mean camera Y-axis direction across all poses.
  2. RANSAC for the floor: horizontal plane near the lowest camera positions.
  3. RANSAC for walls: vertical planes (dot(normal, gravity) ≈ 0).
  4. RANSAC for ceiling (optional): horizontal plane near top.
  5. For each plane, project nearby inlier points onto it.
  6. Clip each plane at its intersections with adjacent planes to avoid overlap.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Distance threshold: points within this distance of a plane get projected onto it
FLOOR_THRESHOLD_M   = 0.06   # 6cm — floors can be slightly uneven
WALL_THRESHOLD_M    = 0.08   # 8cm — walls have more reconstruction noise
CEILING_THRESHOLD_M = 0.08

# Minimum inliers for a plane to be accepted
FLOOR_MIN_INLIERS   = 2000
WALL_MIN_INLIERS    = 1000
CEILING_MIN_INLIERS = 500

# RANSAC
RANSAC_ITERATIONS = 1000
MAX_WALLS         = 6

# Void fill
FILL_GRID_SPACING_M  = 0.03   # 3cm grid on filled planes
FILL_SEARCH_RADIUS_M = 0.15   # only fill where a real point exists within 15cm (in plane coords)


def _gravity_from_cameras(cameras_json: dict) -> np.ndarray:
    """
    Estimate gravity-down direction from camera poses.

    In OpenCV camera convention, the camera Y-axis points down.
    For a right-side-up camera, Y-camera in world coords ≈ gravity-down.

    cameras_json["images"] contains per-image R (world-to-camera rotation).
    The camera Y-axis in world coords = R^T @ [0, 1, 0] = R[:, 1] transposed.
    """
    gravity_down = np.zeros(3)
    images = cameras_json.get("images", [])
    for img in images:
        R = np.array(img.get("R", [[1,0,0],[0,1,0],[0,0,1]]))
        cam_y_world = R.T @ np.array([0., 1., 0.])
        gravity_down += cam_y_world
    if np.linalg.norm(gravity_down) < 1e-6:
        logger.warning("room_layout: could not estimate gravity — using [0,1,0]")
        return np.array([0., 1., 0.])
    gravity_down /= np.linalg.norm(gravity_down)
    logger.info("room_layout: gravity_down = [%.3f, %.3f, %.3f]", *gravity_down)
    return gravity_down


def _ransac_plane(pts: np.ndarray, n_iter: int, threshold: float,
                   normal_constraint: Optional[np.ndarray] = None,
                   normal_min_dot: float = 0.85) -> Optional[tuple]:
    """
    RANSAC plane fit.

    normal_constraint: if provided, only accept planes whose normal has
    |dot(n, constraint)| >= normal_min_dot.
      - For horizontal planes (floor/ceiling): constraint = gravity_down, dot ≥ 0.85
      - For vertical planes (walls): constraint = gravity_down, dot ≤ 0.15

    Returns (normal, point_on_plane, inlier_mask) or None.
    """
    n = len(pts)
    if n < 10:
        return None

    best_mask  = None
    best_count = 0

    for _ in range(n_iter):
        idx = np.random.choice(n, 3, replace=False)
        p0, p1, p2 = pts[idx]
        v1, v2 = p1 - p0, p2 - p0
        nm = np.cross(v1, v2)
        nn = np.linalg.norm(nm)
        if nn < 1e-9:
            continue
        nm /= nn

        if normal_constraint is not None:
            dot = abs(float(np.dot(nm, normal_constraint)))
            is_horiz = dot >= normal_min_dot
            is_vert  = dot <= (1.0 - normal_min_dot)
            # Horizontal constraint
            if normal_min_dot >= 0.5 and not is_horiz:
                continue
            # Vertical constraint
            if normal_min_dot < 0.5 and not is_vert:
                continue

        dists = np.abs((pts - p0) @ nm)
        mask  = dists < threshold
        count = mask.sum()
        if count > best_count:
            best_count = count
            best_mask  = mask

    if best_mask is None or best_count < 3:
        return None

    # Refit normal via SVD on all inliers
    inl      = pts[best_mask]
    centroid = inl.mean(axis=0)
    _, _, Vt = np.linalg.svd(inl - centroid, full_matrices=False)
    normal   = Vt[-1] / (np.linalg.norm(Vt[-1]) + 1e-9)

    # Apply normal constraint (sign) for horizontal planes
    if normal_constraint is not None:
        dot = float(np.dot(normal, normal_constraint))
        if normal_min_dot >= 0.5 and dot < 0:
            normal = -normal

    # Recompute inlier mask with fitted normal
    dists      = np.abs((pts - centroid) @ normal)
    final_mask = dists < threshold

    return normal, centroid, final_mask


def _project_onto_plane(pts: np.ndarray, normal: np.ndarray,
                         point_on_plane: np.ndarray) -> np.ndarray:
    """Project points onto the plane defined by (normal, point_on_plane)."""
    dists = ((pts - point_on_plane) @ normal)[:, np.newaxis]
    return pts - dists * normal


def _fill_plane_voids(
    inlier_pts: np.ndarray,
    inlier_cols: np.ndarray,
    normal: np.ndarray,
    point_on_plane: np.ndarray,
    spacing: float,
    search_radius: float,  # kept for API compat, not used in convex-hull mode
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill voids on the plane using convex-hull containment.

    Projects inlier points into (u,v) plane coordinates, computes their 2D
    convex hull, and fills every grid cell inside the hull.  This covers large
    featureless voids (smooth walls, bare ceiling) that a fixed search-radius
    would leave empty, while naturally stopping at the surface boundary.

    Colour is sampled from the nearest real point (KD-tree, O(N log N)).
    """
    from scipy.spatial import ConvexHull, Delaunay, cKDTree

    if len(inlier_pts) < 10:
        return np.empty((0, 3)), np.empty((0, 3))

    # Orthonormal basis on the plane
    ref  = np.array([0., 0., 1.]) if abs(normal[2]) < 0.9 else np.array([1., 0., 0.])
    u_ax = np.cross(normal, ref);  u_ax /= np.linalg.norm(u_ax)
    v_ax = np.cross(normal, u_ax); v_ax /= np.linalg.norm(v_ax)

    # Project inlier points into (u, v)
    rel    = inlier_pts - point_on_plane
    u_real = rel @ u_ax
    v_real = rel @ v_ax
    uv_real = np.column_stack([u_real, v_real])

    # Build 2D convex hull; need ≥3 non-collinear points
    try:
        hull    = ConvexHull(uv_real)
        checker = Delaunay(uv_real[hull.vertices])
    except Exception:
        return np.empty((0, 3)), np.empty((0, 3))

    # Build fill grid covering the hull extent
    margin = spacing
    u_vals = np.arange(u_real.min() - margin, u_real.max() + margin, spacing)
    v_vals = np.arange(v_real.min() - margin, v_real.max() + margin, spacing)
    ug, vg = np.meshgrid(u_vals, v_vals)
    ug, vg = ug.flatten(), vg.flatten()
    uv_grid = np.column_stack([ug, vg])

    # Keep only grid points inside the convex hull
    inside = checker.find_simplex(uv_grid) >= 0
    ug, vg = ug[inside], vg[inside]
    if len(ug) == 0:
        return np.empty((0, 3)), np.empty((0, 3))

    # Colour each fill point from its nearest real inlier (KD-tree, fast)
    tree     = cKDTree(uv_real)
    _, idx   = tree.query(np.column_stack([ug, vg]), workers=-1)
    fill_pts = point_on_plane + ug[:, None] * u_ax + vg[:, None] * v_ax
    fill_cols = inlier_cols[idx]

    return fill_pts, fill_cols


def _plane_intersection_line(n1, p1, n2, p2):
    """Direction vector of the line where two planes intersect."""
    d = np.cross(n1, n2)
    if np.linalg.norm(d) < 1e-9:
        return None
    return d / np.linalg.norm(d)


async def run_room_layout(
    project_id: str,
    prev_result: dict,
    tmp: Path,
    progress_cb: Callable[[float, str], None],
) -> dict:
    import open3d as o3d
    from backend.core.storage import get_storage

    storage = get_storage()

    # ── Load cloud and cameras ────────────────────────────────────────────────
    cloud_key = prev_result.get("scaled_cloud_key") or prev_result.get("dense_cloud_key", "")
    cameras_key = prev_result.get("camera_poses_key",
                                   f"{project_id}/sfm/cameras.json")

    progress_cb(0.03, "Loading cloud and camera poses…")
    cloud_local   = tmp / "layout_cloud.ply"
    cameras_local = tmp / "layout_cameras.json"
    await storage.download(cloud_key, cloud_local)
    await storage.download(cameras_key, cameras_local)

    pcd  = o3d.io.read_point_cloud(str(cloud_local))
    pts  = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if pcd.has_colors() else np.ones((len(pts), 3)) * 0.6
    cameras_json = json.loads(cameras_local.read_text())

    # ── Step 1: gravity from cameras ─────────────────────────────────────────
    progress_cb(0.08, "Estimating gravity direction from camera poses…")
    gravity_down = _gravity_from_cameras(cameras_json)
    gravity_up   = -gravity_down

    # ── Steps 2-4: planes from marker fitted_planes (gravity-sorted) ─────────
    # fitted_planes come from scale_from_aruco.fit_planes_from_markers which
    # now uses gravity-based Y-sorting: lowest 3 markers=floor, highest 3=ceiling,
    # rest=walls. This is far more reliable than RANSAC on a cluttered cloud.
    fitted_planes = prev_result.get("fitted_planes", [])
    # Plane offsets are in SfM units (triangulated before scale was applied).
    # The cloud has been scaled by confirmed_scale_factor, so scale offsets to match.
    scale_factor = float(prev_result.get("confirmed_scale_factor") or 1.0)

    floor_normal, floor_pt       = None, None
    ceiling_normal, ceiling_pt   = None, None
    wall_planes: list[tuple]     = []

    for plane in fitted_planes:
        n   = np.array(plane["normal"], dtype=float)
        n  /= np.linalg.norm(n) + 1e-9
        pt  = n * float(plane["offset"]) * scale_factor   # ← scale to match cloud
        lbl = plane["label"]
        if lbl == "floor":
            floor_normal, floor_pt = n, pt
            logger.info("room_layout: floor from markers, normal=[%.3f,%.3f,%.3f]", *n)
        elif lbl == "ceiling":
            ceiling_normal, ceiling_pt = n, pt
            logger.info("room_layout: ceiling from markers, normal=[%.3f,%.3f,%.3f]", *n)
        elif lbl.startswith("wall"):
            wall_planes.append((n, pt, None))
            logger.info("room_layout: %s from markers, normal=[%.3f,%.3f,%.3f]", lbl, *n)

    # Floor fallback: only when we had marker-derived planes to anchor the coordinate
    # system. Without at least one marker plane, RANSAC on an unscaled cloud can
    # pick any horizontal cluster (furniture, ceiling) as the "floor", and the
    # subsequent void fill adds 100k+ synthetic points in the wrong place.
    if floor_normal is None and fitted_planes:
        progress_cb(0.20, "No marker floor — RANSAC fallback…")
        heights   = pts @ gravity_down
        floor_cands = pts[heights <= np.percentile(heights, 60)]
        fp = _ransac_plane(floor_cands, RANSAC_ITERATIONS, FLOOR_THRESHOLD_M,
                           normal_constraint=gravity_down, normal_min_dot=0.85)
        if fp and fp[2].sum() >= FLOOR_MIN_INLIERS:
            floor_normal, floor_pt, _ = fp
            logger.info("room_layout: floor via RANSAC fallback")

    progress_cb(0.30, f"Planes: floor={'✓' if floor_normal is not None else '✗'}, "
                      f"ceiling={'✓' if ceiling_normal is not None else '✗'}, "
                      f"walls={len(wall_planes)}")

    # ── Step 5: project inliers onto their planes ─────────────────────────────
    progress_cb(0.55, "Projecting inlier points onto fitted planes…")

    pts_out  = pts.copy()
    n_proj   = 0
    layout_info = {
        "gravity_down":  gravity_down.tolist(),
        "n_walls":       len(wall_planes),
        "has_floor":     floor_normal is not None,
        "has_ceiling":   ceiling_normal is not None,
    }

    fill_pts_list:  list[np.ndarray] = []
    fill_cols_list: list[np.ndarray] = []

    # Project and collect floor inliers first — used as template for wall extent
    floor_inlier_pts  = np.empty((0, 3))
    floor_inlier_cols = np.empty((0, 3))

    def _project_and_fill(normal, pt, threshold, label):
        nonlocal n_proj
        dists = np.abs((pts - pt) @ normal)
        inl   = dists < threshold
        pts_out[inl] = _project_onto_plane(pts[inl], normal, pt)
        n_proj += inl.sum()
        logger.info("room_layout: projected %d %s points", inl.sum(), label)
        if inl.sum() < 10:
            return
        fp, fc = _fill_plane_voids(pts[inl], cols[inl],
                                   normal, pt, FILL_GRID_SPACING_M, FILL_SEARCH_RADIUS_M)
        if len(fp):
            fill_pts_list.append(fp)
            fill_cols_list.append(fc)
            logger.info("room_layout: filled %d void points on %s", len(fp), label)

    if floor_normal is not None:
        _project_and_fill(floor_normal, floor_pt, FLOOR_THRESHOLD_M, "floor")

    if ceiling_normal is not None:
        _project_and_fill(ceiling_normal, ceiling_pt, CEILING_THRESHOLD_M, "ceiling")

    for wi, (w_normal, w_pt, _) in enumerate(wall_planes):
        _project_and_fill(w_normal, w_pt, WALL_THRESHOLD_M, f"wall {wi}")

    # ── Step 6: merge fill points and save ───────────────────────────────────
    n_filled = sum(len(f) for f in fill_pts_list)
    progress_cb(0.80, f"Projected {n_proj:,} pts, filled {n_filled:,} voids…")

    if fill_pts_list:
        all_fill_pts  = np.vstack(fill_pts_list)
        all_fill_cols = np.vstack(fill_cols_list)
        merged_pts    = np.vstack([pts_out, all_fill_pts])
        merged_cols   = np.vstack([cols,    all_fill_cols])
    else:
        merged_pts  = pts_out
        merged_cols = cols

    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(merged_pts)
    pcd_out.colors = o3d.utility.Vector3dVector(np.clip(merged_cols, 0, 1))

    out_key   = f"{project_id}/mvs/dense_layout.ply"
    out_local = tmp / "dense_layout.ply"
    o3d.io.write_point_cloud(str(out_local), pcd_out)
    await storage.upload(out_local, out_key)

    progress_cb(1.0, f"Room layout: {1 + len(wall_planes) + (1 if ceiling_normal is not None else 0)} planes, "
                     f"{n_proj:,} points projected")

    result = dict(prev_result)
    # Keep dense_cloud_key pointing at the original SfM-coordinate cloud so that
    # analyze_coverage (which uses HPR with SfM cameras) reads the right cloud.
    # Expose the layout output under layout_cloud_key and update scaled_cloud_key
    # so downstream exporters pick up the enriched cloud.
    result["layout_cloud_key"] = out_key
    result["scaled_cloud_key"] = out_key
    result["dense_point_count"] = int(len(merged_pts))
    layout_info["n_filled"] = int(n_filled)
    result["room_layout"]      = layout_info
    return result
