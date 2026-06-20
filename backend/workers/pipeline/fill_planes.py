"""
Plane replacement stage.

For each detected architectural plane (ceiling, walls, floor):
  1. Remove the noisy scattered points near the plane (depth-uncertain COLMAP points)
  2. Replace with a clean uniform grid ON the exact plane surface

Planes come from two sources:
  - Marker-confirmed: from scale_from_aruco fitted_planes (very accurate, ≥3 markers)
  - RANSAC-detected: found from the cloud itself for any remaining dominant planes

Only planar regions are replaced — furniture, objects, and other non-planar
content further than the threshold from any plane are kept intact.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

# Points within this distance of a fitted plane are considered inliers and replaced
PLANE_INLIER_DISTANCE_M = 0.08   # 8cm — wide enough to catch noisy wall points

# Grid spacing for the replacement surface
FILL_GRID_SPACING_M = 0.025      # 2.5cm — fine enough to look solid

# RANSAC parameters for auto-detecting additional planes from the cloud
RANSAC_MIN_INLIERS = 5000        # Minimum points to form a plane worth replacing
RANSAC_ITERATIONS  = 500
RANSAC_THRESHOLD   = 0.04        # 4cm RANSAC inlier threshold

# Minimum markers required to trust a marker-confirmed plane
MIN_MARKERS_TO_USE = 3

# Maximum number of RANSAC planes to detect (in addition to marker planes)
MAX_RANSAC_PLANES = 6


def _sample_plane_color(
    inlier_pts: np.ndarray,
    inlier_cols: np.ndarray,
) -> np.ndarray:
    """Return the median colour of existing inlier points as the replacement colour."""
    if len(inlier_cols) == 0:
        return np.array([0.90, 0.90, 0.90])
    return np.median(inlier_cols, axis=0)


def _generate_plane_grid(
    normal: np.ndarray,
    point_on_plane: np.ndarray,
    inlier_pts: np.ndarray,
    spacing: float,
) -> np.ndarray:
    """
    Generate a uniform grid of points on the plane, bounded by the convex hull
    of the inlier points projected onto the plane (with a small margin).
    """
    # Orthonormal basis on the plane
    ref = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, ref);  u /= np.linalg.norm(u)
    v = np.cross(normal, u);    v /= np.linalg.norm(v)

    # Project inlier points into (u, v) coordinates
    rel   = inlier_pts - point_on_plane
    u_pts = rel @ u
    v_pts = rel @ v

    margin = spacing * 2
    u_min, u_max = u_pts.min() - margin, u_pts.max() + margin
    v_min, v_max = v_pts.min() - margin, v_pts.max() + margin

    u_vals = np.arange(u_min, u_max, spacing)
    v_vals = np.arange(v_min, v_max, spacing)
    uu, vv = np.meshgrid(u_vals, v_vals)
    uu, vv = uu.flatten(), vv.flatten()

    pts = point_on_plane + np.outer(uu, u) + np.outer(vv, v)
    return pts


def _ransac_planes(pts: np.ndarray, n_planes: int, min_inliers: int,
                   threshold: float, n_iter: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Iteratively find dominant planes via RANSAC.
    Returns list of (normal, point_on_plane) tuples.
    Each iteration removes inliers before searching for the next plane.
    """
    remaining = pts.copy()
    planes = []

    for _ in range(n_planes):
        if len(remaining) < min_inliers:
            break

        best_normal, best_pt, best_mask = None, None, None
        best_count = 0

        for _ in range(n_iter):
            idx = np.random.choice(len(remaining), 3, replace=False)
            p0, p1, p2 = remaining[idx]
            v1, v2 = p1 - p0, p2 - p0
            n = np.cross(v1, v2)
            nn = np.linalg.norm(n)
            if nn < 1e-9:
                continue
            n /= nn
            dists = np.abs((remaining - p0) @ n)
            mask  = dists < threshold
            count = mask.sum()
            if count > best_count:
                best_count = count
                # Refit normal to all inliers via SVD
                inl = remaining[mask]
                centroid = inl.mean(axis=0)
                _, _, Vt = np.linalg.svd(inl - centroid)
                n_fit = Vt[-1]
                best_normal = n_fit / (np.linalg.norm(n_fit) + 1e-9)
                best_pt     = centroid
                best_mask   = mask

        if best_count < min_inliers:
            break

        planes.append((best_normal, best_pt))
        remaining = remaining[~best_mask]

    return planes


async def run_fill_planes(
    project_id: str,
    prev_result: dict,
    tmp: Path,
    progress_cb: Callable[[float, str], None],
) -> dict:
    import open3d as o3d
    from backend.core.storage import get_storage

    storage = get_storage()
    cloud_key = prev_result.get("dense_cloud_key", "")
    if not cloud_key:
        logger.warning("[%s] fill_planes: no dense_cloud_key — skipping", project_id)
        return prev_result

    progress_cb(0.05, "Downloading cloud for plane replacement…")
    local = tmp / "cloud_for_planes.ply"
    await storage.download(cloud_key, local)

    pcd      = o3d.io.read_point_cloud(str(local))
    pts      = np.asarray(pcd.points)
    cols     = np.asarray(pcd.colors) if pcd.has_colors() else np.ones((len(pts), 3)) * 0.5
    n_orig   = len(pts)

    # ── Collect planes ────────────────────────────────────────────────────────
    fitted_planes: list[dict] = prev_result.get("fitted_planes", [])
    marker_planes = [
        p for p in fitted_planes
        if p.get("n_markers", 0) >= MIN_MARKERS_TO_USE
    ]

    progress_cb(0.10, f"Using {len(marker_planes)} marker-confirmed planes…")

    # Build list of (normal, point_on_plane, label, source)
    plane_specs: list[tuple[np.ndarray, np.ndarray, str]] = []

    for p in marker_planes:
        normal = np.array(p["normal"], dtype=float)
        normal /= np.linalg.norm(normal) + 1e-9
        offset = float(p["offset"])
        pt_on  = normal * offset   # point on plane satisfying dot(n, x) = offset
        plane_specs.append((normal, pt_on, p["label"]))

    # RANSAC to find remaining dominant planes (walls not covered by markers)
    if len(pts) > RANSAC_MIN_INLIERS:
        progress_cb(0.15, "RANSAC fitting additional planes from cloud…")
        # Work on a coarse subset for speed (every 5th point)
        subsample = pts[::5]
        ransac_planes = _ransac_planes(
            subsample, MAX_RANSAC_PLANES,
            min_inliers=RANSAC_MIN_INLIERS // 5,
            threshold=RANSAC_THRESHOLD,
            n_iter=RANSAC_ITERATIONS,
        )
        # Deduplicate against marker planes (skip if normal is nearly parallel and close)
        for r_normal, r_pt in ransac_planes:
            is_dup = False
            for spec_n, spec_pt, _ in plane_specs:
                if abs(float(np.dot(r_normal, spec_n))) > 0.95:
                    if abs(float(np.dot(r_normal, r_pt - spec_pt))) < 0.3:
                        is_dup = True
                        break
            if not is_dup:
                plane_specs.append((r_normal, r_pt, "ransac_plane"))

        logger.info("[%s] fill_planes: %d marker + %d RANSAC planes",
                    project_id, len(marker_planes), len(ransac_planes))

    if not plane_specs:
        progress_cb(1.0, "No planes found — skipping replacement")
        return prev_result

    progress_cb(0.25, f"Replacing {len(plane_specs)} planes (threshold={PLANE_INLIER_DISTANCE_M*100:.0f}cm)…")

    # ── Replace planes one by one ─────────────────────────────────────────────
    keep_mask = np.ones(len(pts), dtype=bool)
    new_pts_list:  list[np.ndarray] = []
    new_cols_list: list[np.ndarray] = []
    total_removed  = 0
    total_added    = 0

    for i, (normal, pt_on, label) in enumerate(plane_specs):
        progress_cb(0.25 + 0.55 * i / len(plane_specs),
                    f"Replacing {label}…")

        # Find inliers: points within PLANE_INLIER_DISTANCE_M of this plane
        dists    = np.abs((pts - pt_on) @ normal)
        inl_mask = dists < PLANE_INLIER_DISTANCE_M

        # Only process if we have enough inliers to be meaningful
        n_inl = inl_mask.sum()
        if n_inl < 200:
            logger.debug("[%s] fill_planes: %s has only %d inliers — skip", project_id, label, n_inl)
            continue

        inl_pts  = pts[inl_mask]
        inl_cols = cols[inl_mask]

        # Mark inliers for removal from original cloud
        keep_mask &= ~inl_mask
        total_removed += n_inl

        # Generate clean replacement grid
        grid = _generate_plane_grid(normal, pt_on, inl_pts, FILL_GRID_SPACING_M)
        if len(grid) == 0:
            continue

        color = _sample_plane_color(inl_pts, inl_cols)
        grid_cols = np.tile(color, (len(grid), 1))

        new_pts_list.append(grid)
        new_cols_list.append(grid_cols)
        total_added += len(grid)

        logger.info("[%s] fill_planes: %s — removed %d noisy pts, added %d flat pts",
                    project_id, label, n_inl, len(grid))

    # ── Merge: kept originals + clean plane grids ─────────────────────────────
    kept_pts  = pts[keep_mask]
    kept_cols = cols[keep_mask]
    n_kept    = len(kept_pts)

    if new_pts_list:
        all_new_pts  = np.vstack(new_pts_list)
        all_new_cols = np.vstack(new_cols_list)
        merged_pts   = np.vstack([kept_pts, all_new_pts])
        merged_cols  = np.vstack([kept_cols, all_new_cols])
    else:
        merged_pts  = kept_pts
        merged_cols = kept_cols

    progress_cb(0.85, f"Removed {total_removed:,} noisy pts, added {total_added:,} flat pts "
                      f"(net: {len(merged_pts) - n_orig:+,})")

    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(merged_pts)
    pcd_out.colors = o3d.utility.Vector3dVector(np.clip(merged_cols, 0, 1))

    filled_key   = f"{project_id}/mvs/dense_filled.ply"
    filled_local = tmp / "dense_filled.ply"
    o3d.io.write_point_cloud(str(filled_local), pcd_out)
    await storage.upload(filled_local, filled_key)

    progress_cb(1.0, f"Plane replacement done: {n_orig:,} → {len(merged_pts):,} pts "
                     f"across {len(plane_specs)} surfaces")

    result = dict(prev_result)
    result["dense_cloud_key"]   = filled_key
    result["scaled_cloud_key"]  = filled_key   # override so refine_cloud uses the filled version
    result["dense_point_count"] = int(len(merged_pts))
    result["plane_fill"] = {
        "n_planes":    int(len(plane_specs)),
        "n_removed":   int(total_removed),
        "n_added":     int(total_added),
        "n_before":    int(n_orig),
        "n_after":     int(len(merged_pts)),
        "labels":      [s[2] for s in plane_specs],
        "spacing_m":   float(FILL_GRID_SPACING_M),
        "threshold_m": float(PLANE_INLIER_DISTANCE_M),
    }
    return result
