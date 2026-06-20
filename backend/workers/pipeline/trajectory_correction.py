"""
Corrects mis-registered "teleport" blocks flagged by SfM (sfm.py's
_detect_trajectory_jumps): a camera that briefly enters a small, mostly
featureless space (closet, stairwell nook, bathroom) and loses tracking can
have the rest of its frames re-anchored to a duplicate of nearby geometry,
offset by a rigid transform.

Strategy, per jump:
  1. Estimate a correction transform `G` from trajectory continuity — assume
     the camera's motion from the frame before the jump to the jump frame
     should resemble its motion in the preceding step.
  2. Refine `G` with ICP between the sparse points seen only by the
     "after" block and the sparse points seen only by the "before" block.
     A high ICP fitness confirms the "after" block is a duplicate of
     existing geometry, not new content.
  3. If confirmed, classify dense-cloud points that are spatially closer to
     the "after" cluster than the "before" cluster (within a small radius of
     an "after" sparse point) and apply the refined transform to them.

Downstream refine_cloud's SOR + voxel downsampling naturally merges the
now-overlapping duplicate region, so no explicit deduplication is needed here.
"""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

_ICP_FITNESS_THRESHOLD = 0.5     # below this, treat the "after" block as real content, not a duplicate
_ICP_DIST_THRESHOLD_M  = 0.5     # ICP correspondence search radius
_CLASSIFY_RADIUS_M     = 0.3     # dense points within this of an "after" sparse point are candidates


def _world_from_cam(img) -> np.ndarray:
    cw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
    R = np.asarray(cw.rotation.matrix())
    t = np.asarray(cw.translation)
    W = np.eye(4)
    W[:3, :3] = R.T
    W[:3, 3] = -R.T @ t
    return W


async def run_trajectory_correction(
    project_id: str,
    prev_result: dict,
    tmp: Path,
    progress_cb: Callable[[float, str], None],
) -> dict:
    import open3d as o3d
    import pycolmap
    from backend.core.storage import get_storage

    storage = get_storage()
    result = dict(prev_result)

    jumps = prev_result.get("sfm_trajectory_jumps", [])
    if not jumps:
        progress_cb(1.0, "No trajectory jumps detected — nothing to correct")
        result["trajectory_correction"] = {"applied": False, "reason": "no jumps detected"}
        return result

    dense_cloud_key = prev_result.get("dense_cloud_key", "")
    workspace_key   = prev_result.get("workspace_key", "")
    image_keys      = prev_result.get("image_keys", [])
    if not dense_cloud_key or not workspace_key:
        progress_cb(1.0, "Missing dense cloud or workspace — skipping correction")
        result["trajectory_correction"] = {"applied": False, "reason": "missing dense_cloud_key or workspace_key"}
        return result

    progress_cb(0.05, f"Downloading workspace + dense cloud ({len(jumps)} jump(s) to check)…")
    workspace_local = tmp / "workspace.tar.gz"
    await storage.download(workspace_key, workspace_local)
    sparse_dir = tmp / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(workspace_local, "r:gz") as tar:
        tar.extractall(tmp)

    dense_local = tmp / "dense.ply"
    await storage.download(dense_cloud_key, dense_local)

    rec = pycolmap.Reconstruction(str(sparse_dir))
    name_to_img = {img.name: img for img in rec.images.values()}

    all_names = [Path(k).name for k in image_keys]
    name_to_idx = {n: i for i, n in enumerate(all_names)}

    pcd = o3d.io.read_point_cloud(str(dense_local))
    pts = np.asarray(pcd.points)
    has_normals = pcd.has_normals()
    normals = np.asarray(pcd.normals) if has_normals else None
    has_colors = pcd.has_colors()
    colors = np.asarray(pcd.colors) if has_colors else None

    applied_jumps = []
    skipped_jumps = []

    for ji, jump in enumerate(jumps):
        before_frame = jump["before_frame"]
        after_frame  = jump["after_frame"]
        progress_cb(0.1 + 0.7 * ji / len(jumps),
                    f"Checking jump {before_frame} → {after_frame}…")

        if before_frame not in name_to_img or after_frame not in name_to_img \
                or before_frame not in name_to_idx or after_frame not in name_to_idx:
            skipped_jumps.append({**jump, "reason": "frame not found in reconstruction"})
            continue

        before_idx = name_to_idx[before_frame]
        after_idx  = name_to_idx[after_frame]

        # The "after" block runs until the next jump's before_frame, or to
        # the end of the sequence.
        block_end_idx = len(all_names) - 1
        for other in jumps[ji + 1:]:
            if other["before_frame"] in name_to_idx:
                block_end_idx = name_to_idx[other["before_frame"]]
                break

        # Need the frame two steps before the jump to estimate "expected"
        # continuation motion.
        if before_idx - 1 < 0:
            skipped_jumps.append({**jump, "reason": "no preceding frame for continuity estimate"})
            continue
        prev_name = all_names[before_idx - 1]
        if prev_name not in name_to_img:
            skipped_jumps.append({**jump, "reason": "preceding frame not registered"})
            continue

        W_prev   = _world_from_cam(name_to_img[prev_name])
        W_before = _world_from_cam(name_to_img[before_frame])
        W_after  = _world_from_cam(name_to_img[after_frame])

        # Relative motion from prev -> before, extrapolated to before -> after.
        M = np.linalg.inv(W_prev) @ W_before
        W_after_expected = W_before @ M
        G = W_after_expected @ np.linalg.inv(W_after)

        # ── Gather sparse points exclusively visible from each side ──────────
        after_ids = set()
        for i in range(after_idx, block_end_idx + 1):
            name = all_names[i]
            if name in name_to_img:
                after_ids.add(name_to_img[name].image_id)

        before_ids = {
            img.image_id for name, img in name_to_img.items()
            if name_to_idx.get(name, -1) not in range(after_idx, block_end_idx + 1)
        }

        before_pts_list, after_pts_list = [], []
        for p3d in rec.points3D.values():
            track_ids = {el.image_id for el in p3d.track.elements}
            in_after  = bool(track_ids & after_ids)
            in_before = bool(track_ids & before_ids)
            if in_after and not in_before:
                after_pts_list.append(p3d.xyz)
            elif in_before and not in_after:
                before_pts_list.append(p3d.xyz)

        if len(after_pts_list) < 50 or len(before_pts_list) < 50:
            skipped_jumps.append({**jump, "reason": "not enough exclusive sparse points to validate"})
            continue

        before_pts_sparse = np.asarray(before_pts_list)
        after_pts_sparse  = np.asarray(after_pts_list)

        # ── ICP refinement of G ───────────────────────────────────────────────
        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(after_pts_sparse)
        tgt = o3d.geometry.PointCloud()
        tgt.points = o3d.utility.Vector3dVector(before_pts_sparse)

        reg = o3d.pipelines.registration.registration_icp(
            src, tgt, _ICP_DIST_THRESHOLD_M, G,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
        )

        if reg.fitness < _ICP_FITNESS_THRESHOLD:
            skipped_jumps.append({
                **jump,
                "reason": f"ICP fitness {reg.fitness:.2f} below threshold "
                          f"{_ICP_FITNESS_THRESHOLD} — likely real new content, not a duplicate",
                "icp_fitness": round(reg.fitness, 3),
            })
            continue

        G_refined = reg.transformation

        # ── Classify dense points belonging to the "after" block ─────────────
        # A dense point is "after-block" if it's within _CLASSIFY_RADIUS_M of
        # an "after" sparse point and closer to the after cluster than to the
        # before cluster — restricting to the after cluster's bbox keeps this
        # cheap on multi-million-point clouds.
        margin = _CLASSIFY_RADIUS_M * 2
        bbox_min = after_pts_sparse.min(axis=0) - margin
        bbox_max = after_pts_sparse.max(axis=0) + margin
        in_bbox = np.all((pts >= bbox_min) & (pts <= bbox_max), axis=1)
        candidate_idx = np.where(in_bbox)[0]

        if len(candidate_idx) == 0:
            skipped_jumps.append({**jump, "reason": "no dense points in after-cluster bounding box"})
            continue

        after_tree  = o3d.geometry.KDTreeFlann(src)
        before_tree = o3d.geometry.KDTreeFlann(tgt)

        to_transform = []
        for idx in candidate_idx:
            p = pts[idx]
            _, _, d_after  = after_tree.search_knn_vector_3d(p, 1)
            _, _, d_before = before_tree.search_knn_vector_3d(p, 1)
            d_after_m  = float(np.sqrt(d_after[0]))
            d_before_m = float(np.sqrt(d_before[0]))
            if d_after_m < _CLASSIFY_RADIUS_M and d_after_m < d_before_m:
                to_transform.append(idx)

        if not to_transform:
            skipped_jumps.append({**jump, "reason": "no dense points uniquely matched to after-cluster"})
            continue

        to_transform = np.array(to_transform)
        R_g, t_g = G_refined[:3, :3], G_refined[:3, 3]
        pts[to_transform] = (R_g @ pts[to_transform].T).T + t_g
        if has_normals:
            normals[to_transform] = (R_g @ normals[to_transform].T).T

        applied_jumps.append({
            **jump,
            "icp_fitness": round(reg.fitness, 3),
            "icp_inlier_rmse": round(reg.inlier_rmse, 4),
            "n_points_corrected": int(len(to_transform)),
            "block_start": after_frame,
            "block_end": all_names[block_end_idx],
        })
        logger.info(
            "[%s] trajectory_correction: jump %s→%s — corrected %d dense points "
            "(ICP fitness=%.2f, rmse=%.3f)",
            project_id, before_frame, after_frame, len(to_transform),
            reg.fitness, reg.inlier_rmse,
        )

    if not applied_jumps:
        progress_cb(1.0, f"No corrections applied ({len(skipped_jumps)} jump(s) skipped)")
        result["trajectory_correction"] = {
            "applied": False,
            "skipped": skipped_jumps,
        }
        return result

    pcd.points = o3d.utility.Vector3dVector(pts)
    if has_normals:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    if has_colors:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    corrected_key = f"{project_id}/mvs/dense_corrected.ply"
    corrected_local = tmp / "dense_corrected.ply"
    o3d.io.write_point_cloud(str(corrected_local), pcd)
    await storage.upload(corrected_local, corrected_key)

    total_corrected = sum(j["n_points_corrected"] for j in applied_jumps)
    progress_cb(1.0, f"Corrected {len(applied_jumps)} jump(s), {total_corrected:,} points repositioned")

    result["dense_cloud_key"] = corrected_key
    result["trajectory_correction"] = {
        "applied": True,
        "corrected_cloud_key": corrected_key,
        "jumps_applied": applied_jumps,
        "jumps_skipped": skipped_jumps,
    }
    return result
