"""
Length x Breadth x Height from a gravity-aligned point cloud.

Aligns the cloud to gravity ("up" -> +Z), takes the minimum-area bounding
rectangle in the horizontal plane for L x B (robust to arbitrary yaw), and
the vertical extent for H. When no gravity is known (no camera-derived or
marker-derived "up"), detect_floor_gravity finds it from the cloud itself.
Pass scale=1.0 for a cloud already in metres
(the export pipeline's scaled_cloud_key), or a m/unit factor for a cloud
still in raw SfM units.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)


# A LiDAR floor/ground plane needs real support to trust — this is deliberately
# the same order of magnitude as room_layout.py's FLOOR_MIN_INLIERS. Depends on
# point density — a heavily downsampled cloud may need a lower value.
_FLOOR_MIN_INLIERS = 2000
_FLOOR_RANSAC_ITERATIONS = 2000
_FLOOR_DISTANCE_THRESHOLD_M = 0.02

# How many dominant planar segments to extract (iteratively, removing inliers
# each round) before picking the vertical one among them.
_MAX_PLANE_CANDIDATES = 8
# Two plane normals within this angle (mod sign) are treated as the same
# orientation cluster — merges floor+ceiling, and multiple sub-segments of
# the same wall, into one candidate axis.
_AXIS_GROUP_TOL_DEG = 12.0


def detect_floor_gravity(pcd) -> tuple[list[float] | None, int]:
    """
    Iteratively RANSAC the dominant planar segments in the cloud, group them
    by orientation, and pick the vertical axis as the group with the smallest
    full-cloud extent along itself (a room is reliably wider/longer than it
    is tall, so floor<->ceiling is the shortest of the dominant directions —
    unlike "the single largest plane", which is often a big, clean wall
    instead of a furniture-occluded floor).

    Returns (gravity_up_world, inlier_count), or (None, 0) if no plane with
    enough support was found. `inlier_count` is the combined support across
    every plane merged into the winning orientation group.
    """
    all_pts = np.asarray(pcd.points)
    remaining = pcd
    # Each group: {"normal": folded unit normal, "inliers": int, "sample_pt": mean of first plane's inliers}
    axis_groups: list[dict] = []

    for _ in range(_MAX_PLANE_CANDIDATES):
        if len(remaining.points) < _FLOOR_MIN_INLIERS:
            break
        try:
            plane_model, inliers = remaining.segment_plane(
                distance_threshold=_FLOOR_DISTANCE_THRESHOLD_M,
                ransac_n=3,
                num_iterations=_FLOOR_RANSAC_ITERATIONS,
            )
        except Exception as e:
            logger.warning("dimensions: floor RANSAC failed: %s", e)
            break

        n_inliers = len(inliers)
        if n_inliers < _FLOOR_MIN_INLIERS:
            break

        a, b, c, _d = plane_model
        normal = np.array([a, b, c], dtype=float)
        normal /= (np.linalg.norm(normal) + 1e-9)
        # Fold to a canonical sign so floor/ceiling (antiparallel normals)
        # land in the same orientation group.
        idx = int(np.argmax(np.abs(normal)))
        if normal[idx] < 0:
            normal = -normal

        remaining_pts = np.asarray(remaining.points)
        sample_pt = remaining_pts[inliers].mean(axis=0)

        merged = False
        for g in axis_groups:
            if abs(float(np.dot(g["normal"], normal))) > np.cos(np.radians(_AXIS_GROUP_TOL_DEG)):
                g["inliers"] += n_inliers
                merged = True
                break
        if not merged:
            axis_groups.append({"normal": normal, "inliers": n_inliers, "sample_pt": sample_pt})

        remaining = remaining.select_by_index(inliers, invert=True)

    candidates = [g for g in axis_groups if g["inliers"] >= _FLOOR_MIN_INLIERS]
    if not candidates:
        logger.info("dimensions: no planar segment reached %d inliers — no floor",
                    _FLOOR_MIN_INLIERS)
        return None, 0

    for g in candidates:
        proj = all_pts @ g["normal"]
        lo, hi = np.percentile(proj, [0.5, 99.5])
        g["extent"] = hi - lo

    winner = min(candidates, key=lambda g: g["extent"])
    normal = winner["normal"]

    # Orient the normal so it points from the floor toward the bulk of the
    # cloud (i.e. "up" into the room), not down into the earth below it.
    centroid_all = all_pts.mean(axis=0)
    if np.dot(centroid_all - winner["sample_pt"], normal) < 0:
        normal = -normal

    logger.info(
        "dimensions: vertical axis picked from %d orientation cluster(s) "
        "(extents=%s) — %d inliers, gravity_up=[%.3f, %.3f, %.3f]",
        len(candidates), [round(g["extent"], 2) for g in candidates],
        winner["inliers"], *normal,
    )
    return normal.tolist(), int(winner["inliers"])


def rotation_aligning(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix R such that R @ a == b, for unit vectors a, b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def compute_dimensions(
    pcd: "o3d.geometry.PointCloud",
    gravity_up: list[float],
    scale: float = 1.0,
    outlier_removal: bool = True,
    pct_clip: float = 0.5,
) -> dict:
    if outlier_removal and len(pcd.points) > 20:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    points = np.asarray(pcd.points)
    if len(points) == 0:
        raise ValueError("point cloud is empty after outlier removal")

    R = rotation_aligning(np.array(gravity_up, dtype=float), np.array([0.0, 0.0, 1.0]))
    aligned = points @ R.T

    z = aligned[:, 2]
    if pct_clip > 0:
        lo, hi = np.percentile(z, [pct_clip, 100 - pct_clip])
    else:
        lo, hi = float(z.min()), float(z.max())
    height_units = hi - lo

    xy = aligned[:, :2].astype(np.float32)
    (w_units, h_units) = cv2.minAreaRect(xy)[1]
    length_units, breadth_units = max(w_units, h_units), min(w_units, h_units)

    length_m  = length_units  * scale
    breadth_m = breadth_units * scale
    height_m  = height_units  * scale

    return {
        "length_m":      round(float(length_m), 3),
        "breadth_m":     round(float(breadth_m), 3),
        "height_m":      round(float(height_m), 3),
        "footprint_m2":  round(float(length_m * breadth_m), 2),
        "volume_m3":     round(float(length_m * breadth_m * height_m), 2),
        "n_points_used": int(len(points)),
    }


def compare_to_ground_truth(dimensions: dict, ground_truth: dict) -> dict:
    """
    Compare computed dimensions against user-supplied tape-measure ground
    truth, as a way to sanity-check the derived scale (ArUco or LiDAR-native).

    ground_truth may supply any subset of length_m/breadth_m/height_m — only
    the fields present in both dicts are compared. Returns {} if nothing
    could be compared (e.g. ground_truth is empty, or dimensions computation
    failed before this ever runs).
    """
    comparison: dict = {}
    abs_errors = []
    for field in ("length_m", "breadth_m", "height_m"):
        truth = ground_truth.get(field)
        predicted = dimensions.get(field)
        if truth is None or predicted is None or truth <= 0:
            continue
        error_m = float(predicted) - float(truth)
        error_pct = (error_m / float(truth)) * 100.0
        comparison[field] = {
            "predicted_m": round(float(predicted), 3),
            "truth_m":     round(float(truth), 3),
            "error_m":     round(error_m, 3),
            "error_pct":   round(error_pct, 2),
        }
        abs_errors.append(abs(error_pct))

    if abs_errors:
        comparison["mean_abs_error_pct"] = round(sum(abs_errors) / len(abs_errors), 2)
    return comparison
