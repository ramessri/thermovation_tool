"""
Point cloud refinement stage — runs after apply_scale, before coverage.

Two operations:
1. Statistical outlier removal — removes isolated noise points that COLMAP
   produces near featureless surfaces and occlusion boundaries.
2. Voxel downsampling — produces a spatially uniform cloud that scores better
   on coverage metrics and renders faster in the viewer.

Both are fast (seconds on a 1M-point cloud) and use Open3D on CPU.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


async def run_refine_cloud(
    project_id: str,
    prev_result: dict,
    tmp: Path,
    progress_cb: Callable[[float, str], None],
    scale_factor: Optional[float] = None,
    scene_type: Optional[str] = None,
) -> dict:
    """
    Refine the dense point cloud:
      - Statistical outlier removal (removes isolated noisy points)
      - Voxel downsampling (uniform spatial density, faster rendering)

    Returns prev_result with dense_cloud_key updated to the refined cloud.
    """
    import open3d as o3d
    from backend.core.storage import get_storage

    storage = get_storage()

    # Use the post-scale cloud if available, otherwise raw MVS cloud
    cloud_key = prev_result.get("scaled_cloud_key") or prev_result.get("dense_cloud_key")
    if not cloud_key:
        logger.warning("[%s] refine_cloud: no cloud key in prev_result — skipping", project_id)
        return prev_result

    progress_cb(0.05, "Downloading point cloud for refinement…")
    local_path = tmp / "cloud_to_refine.ply"
    await storage.download(cloud_key, local_path)

    progress_cb(0.15, "Loading point cloud…")
    pcd = o3d.io.read_point_cloud(str(local_path))
    n_before = len(pcd.points)
    logger.info("[%s] refine_cloud: loaded %d points from %s", project_id, n_before, cloud_key)

    if n_before == 0:
        logger.warning("[%s] refine_cloud: empty cloud — skipping", project_id)
        return prev_result

    # ── Statistical outlier removal ────────────────────────────────────────────
    # Each point must have ≥ nb_neighbors within std_ratio standard deviations
    # of the mean neighbour distance. Effective at removing COLMAP noise spikes
    # near featureless surfaces and occlusion edges.
    progress_cb(0.25, f"Statistical outlier removal ({n_before:,} points)…")
    pcd_sor, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    n_after_sor = len(pcd_sor.points)
    sor_removed = n_before - n_after_sor
    logger.info("[%s] refine_cloud: SOR removed %d points (%.1f%%)",
                project_id, sor_removed, 100 * sor_removed / max(n_before, 1))

    # ── Voxel downsampling ─────────────────────────────────────────────────────
    # Object scans need finer voxels — a 20 cm object at 3 mm voxel is already
    # 67 voxels across its longest axis, but at room-scan defaults the bbox
    # floor (10 mm) would be far too coarse for small objects.
    progress_cb(0.55, f"Voxel downsampling ({n_after_sor:,} points)…")
    is_object = (scene_type == "object")
    if scale_factor and scale_factor > 0:
        voxel_m = 0.0015 if is_object else 0.003   # 1.5 mm / 3 mm
    else:
        bbox = pcd_sor.get_axis_aligned_bounding_box()
        diag = float(np.linalg.norm(np.asarray(bbox.get_extent())))
        floor_m = 0.001 if is_object else 0.01     # 1 mm / 10 mm floor
        voxel_m = max(floor_m, diag * 0.003)       # 0.3% of diagonal

    pcd_down = pcd_sor.voxel_down_sample(voxel_size=voxel_m)
    n_final = len(pcd_down.points)
    logger.info("[%s] refine_cloud: voxel %.4f → %d points (%.1f%% of input)",
                project_id, voxel_m, n_final, 100 * n_final / max(n_before, 1))

    # ── Upload refined cloud ───────────────────────────────────────────────────
    progress_cb(0.75, f"Refined: {n_final:,} points (removed {n_before - n_final:,} noise/redundant)")
    refined_key = f"{project_id}/mvs/dense_refined.ply"
    refined_local = tmp / "dense_refined.ply"
    o3d.io.write_point_cloud(str(refined_local), pcd_down)
    await storage.upload(refined_local, refined_key)

    progress_cb(1.0, f"Refinement complete: {n_before:,} → {n_final:,} points "
                     f"({sor_removed:,} noise removed, voxel={voxel_m*1000:.1f}mm)")

    result = dict(prev_result)
    result["dense_cloud_key"]     = refined_key   # coverage uses this
    result["scaled_cloud_key"]    = refined_key   # export uses this
    result["dense_point_count"]   = n_final
    result["dense_point_count_raw"] = n_before
    result["refinement"] = {
        "sor_removed":   sor_removed,
        "voxel_m":       round(voxel_m, 4),
        "n_before":      n_before,
        "n_after":       n_final,
    }
    return result
