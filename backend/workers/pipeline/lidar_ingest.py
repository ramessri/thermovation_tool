"""
LiDAR point-cloud ingestion stage.

Alternate entry point into the pipeline for scans that already come from a
depth sensor (iPhone/iPad LiDAR, dedicated scanner) as a pre-built `.ply`
point cloud, instead of video/photos. There is no SfM/MVS to run — the dense
geometry already exists — so this stage stands in for
extract_metadata..apply_known_scale in one step and hands off directly to
refine_cloud/export.

Two things the video pipeline gets "for free" from camera poses have to be
derived differently here, since there are no registered camera frames:

  - Scale: most LiDAR scanning apps already export in metres. We trust that
    by default (`lidar_scale_factor=1.0`) and multiply through only if the
    caller supplies a different value (e.g. a scanner that exports mm/cm).
    There is no ArUco cross-check available in this path.
  - Gravity / floor: instead of averaging camera Y-axes (room_layout.py's
    approach), we iteratively RANSAC the largest planar segments in the raw
    cloud and group them by orientation. Taking the single largest plane and
    calling it "the floor" fails whenever a wall has more clean, unoccluded
    surface area than the (often furniture-cluttered) floor — verified
    against a real scan where a large diagonal wall out-voted the actual
    floor by 2:1. Instead we pick the *vertical* axis among the dominant
    orientation clusters as the one with the smallest full-cloud extent along
    itself: an indoor room is reliably wider/longer than it is tall, so
    floor-to-ceiling is the shortest of the (typically mutually near-
    orthogonal) floor/wall/wall directions. Object scans have no floor
    concept (matches the existing exporter/fill_planes convention) so gravity
    detection is skipped there.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np

from backend.workers.pipeline.dimensions import detect_floor_gravity

logger = logging.getLogger(__name__)

async def run_lidar_ingest(
    project_id: str,
    ply_key: str,
    tmp: Path,
    progress_cb: Callable[[float, str], None],
    scene_type: str = "indoor_room",
    lidar_scale_factor: float = 1.0,
    ground_truth_dimensions: dict | None = None,
) -> dict:
    import open3d as o3d
    from backend.core.storage import get_storage

    storage = get_storage()

    progress_cb(0.0, "Downloading LiDAR point cloud…")
    local_ply = tmp / "lidar_input.ply"
    await storage.download(ply_key, local_ply)

    progress_cb(0.15, "Loading point cloud…")
    pcd = o3d.io.read_point_cloud(str(local_ply))
    n_pts = len(pcd.points)
    if n_pts == 0:
        raise RuntimeError(
            f"LiDAR upload at {ply_key} contains zero points — "
            "is this a valid .ply point cloud?"
        )
    logger.info("[%s] lidar_ingest: loaded %d points from %s", project_id, n_pts, ply_key)

    # ── Apply scale, if the caller says the scanner isn't already metric ────
    scale = float(lidar_scale_factor) if lidar_scale_factor else 1.0
    if abs(scale - 1.0) > 1e-9:
        progress_cb(0.30, f"Rescaling by {scale}× (declared non-metric source)…")
        pts = np.asarray(pcd.points) * scale
        pcd.points = o3d.utility.Vector3dVector(pts)
        scale_source = "lidar_manual"
    else:
        scale_source = "lidar_native"

    # ── Gravity / floor, from the cloud itself (no camera poses available) ──
    gravity_up_world = None
    floor_inliers = 0
    is_object = (scene_type == "object")
    if not is_object:
        progress_cb(0.45, "Detecting floor plane (RANSAC)…")
        gravity_up_world, floor_inliers = detect_floor_gravity(pcd)
    else:
        progress_cb(0.45, "Object scan — skipping floor/gravity detection")

    # ── Upload the (possibly rescaled) cloud as the working dense cloud ─────
    progress_cb(0.75, "Uploading ingested cloud…")
    out_key = f"{project_id}/lidar/ingested.ply"
    out_local = tmp / "ingested.ply"
    o3d.io.write_point_cloud(str(out_local), pcd)
    await storage.upload(out_local, out_key)

    progress_cb(
        1.0,
        f"Ingested {n_pts:,} LiDAR points"
        + (f", floor found ({floor_inliers:,} inliers)" if gravity_up_world else ", no floor detected"),
    )

    return {
        "scene_type": scene_type,
        "ground_truth_dimensions": ground_truth_dimensions,
        "dense_cloud_key": out_key,
        "scaled_cloud_key": out_key,
        "dense_point_count": n_pts,
        "n_points": n_pts,
        "confirmed_scale_factor": 1.0,   # cloud is already scaled on disk above
        "confirmed_scale_source": scale_source,
        "gravity_up_world": gravity_up_world,
        "lidar_ingest": {
            "source_key": ply_key,
            "applied_scale_factor": scale,
            "has_floor": gravity_up_world is not None,
            "floor_inliers": floor_inliers,
            "n_points": n_pts,
        },
    }
