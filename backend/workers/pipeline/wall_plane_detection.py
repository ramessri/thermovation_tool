"""
HVAC stage 1 — wall-plane detection.

Image-first by design: ADE20K (SegFormer) segments "wall" pixels on a sample
of registered frames FIRST; only the 3D points those images actually
classified as wall ever reach the plane fit. RANSAC still does the final
plane-equation fit (recovering a real metric normal + offset needs 3D data —
that part cannot be image-only, same reason hvac_placement.py's distances
can't be either), but it never sees a point image segmentation didn't already
call "wall." This replaces an earlier version that RANSAC'd the whole cloud
first and only cross-checked ADE20K afterward — backwards from what the
segmentation should drive.

Closes a documented gap: room_layout.py's fit_planes_from_markers() only ever
returns a floor plane, with an explicit note that "walls/ceiling should be
detected by RANSAC on the actual point cloud (a separate future stage)". This
is that stage, scoped to walls (the HVAC placement search needs a wall, not a
ceiling).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np

from backend.core.config import settings

logger = logging.getLogger(__name__)

# Reuse room_layout.py's plane-fitting primitive rather than reimplementing RANSAC.
from backend.workers.pipeline.room_layout import (
    _ransac_plane, _gravity_from_cameras, RANSAC_ITERATIONS, WALL_THRESHOLD_M,
)

MAX_WALL_CANDIDATES = 6          # mirrors room_layout.py's MAX_WALLS
MIN_SEARCH_POINTS = 200          # stop iterating once too few points remain to matter
MIN_MASK_POINTS_PER_FRAME = 30   # ignore a frame's wall mask if it barely touches the cloud
# _ransac_plane's vertical branch (normal_min_dot < 0.5) can only bias
# SAMPLING toward dot(normal, gravity) <= (1 - normal_min_dot) — 0.49 is as
# tight as that branch gets (~30.7°), which is NOT a hard cutoff, just a
# preference during the random-sample search. The real, hard acceptance gate
# is POST_FIT_MAX_TILT_DOT below, checked explicitly against the final fitted
# normal. Confirmed on real footage this distinction matters: a 0.55 (~33.4°)
# "safety net" is actually LOOSER than the sampling bias and rejects nothing —
# it let a ~30° tilted surface (a bed's mattress top) through as a "wall".
# A 25° hard gate (sin(25°)=0.4226) makes this the binding constraint.
WALL_NORMAL_MIN_DOT = 0.49
POST_FIT_MAX_TILT_DOT = 0.4226   # hard cutoff: reject anything >25° from true vertical
DEDUP_NORMAL_DOT = 0.95          # candidates this parallel AND...
DEDUP_OFFSET_M = 0.30            # ...this close in offset are the same wall


def _is_duplicate(normal: np.ndarray, offset: float, existing: list[dict]) -> bool:
    for c in existing:
        d = float(np.dot(normal, np.array(c["normal"])))
        # Align the other plane's normal with ours before comparing offsets:
        # RANSAC returns either sign, and opposite walls of a room centred near
        # the SfM origin have equal raw offsets with flipped normals (−2.1 vs
        # +2.1 m read as −2.1 and −2.1), which used to drop one wall of every
        # opposite pair.
        if abs(d) > DEDUP_NORMAL_DOT and abs(offset - np.sign(d) * c["offset_m"]) < DEDUP_OFFSET_M:
            return True
    return False


async def run_wall_plane_detection(
    project_id: str,
    prev_result: dict,
    tmp: Path,
    progress_cb: Callable[[float, str], None],
) -> dict:
    """
    Returns prev_result augmented with:
        wall_candidates: [{normal, point_on_plane_m, offset_m, inliers,
                            meets_min_inliers, ade20k_wall_confidence,
                            ade20k_frame, ade20k_overlay_key,
                            extent_u_m, extent_v_m}, ...]
        ranked by inlier count, descending.

    dense_cloud_key at this point in the chain is the refine_cloud output,
    already metric-scaled (refine_cloud_task converges dense_cloud_key and
    scaled_cloud_key to the same file) — everything below operates directly
    in metric space, including the camera poses (loaded with scale_factor so
    R/t line up with the metric cloud; see depth_fusion_common.load_registered_cameras).
    """
    import cv2
    import open3d as o3d
    from backend.core.storage import get_storage
    from backend.workers.pipeline.depth_fusion_common import load_registered_cameras
    from backend.workers.pipeline.hvac_common import (
        download_registered_frames,
        segformer_wall_mask, points_in_mask, wall_basis,
    )

    storage = get_storage()

    cloud_key = prev_result.get("dense_cloud_key") or prev_result.get("scaled_cloud_key", "")
    cameras_key = prev_result.get("camera_poses_key", f"{project_id}/sfm/cameras.json")
    if not cloud_key:
        logger.warning("[%s] wall_plane_detection: no dense cloud key — skipping", project_id)
        return prev_result

    progress_cb(0.05, "Loading refined cloud and camera poses…")
    cloud_local = tmp / "wall_cloud.ply"
    cameras_local = tmp / "wall_cameras.json"
    await storage.download(cloud_key, cloud_local)
    await storage.download(cameras_key, cameras_local)

    pcd = o3d.io.read_point_cloud(str(cloud_local))
    pts = np.asarray(pcd.points)   # already metric — see docstring
    if len(pts) < MIN_SEARCH_POINTS:
        logger.warning("[%s] wall_plane_detection: cloud too small (%d pts) — skipping",
                       project_id, len(pts))
        return prev_result

    scale_factor = float(prev_result.get("confirmed_scale_factor") or 1.0)
    cameras_json = json.loads(cameras_local.read_text())
    cameras = load_registered_cameras(cameras_json, scale_factor=scale_factor)
    if not cameras:
        logger.warning("[%s] wall_plane_detection: no registered cameras — skipping", project_id)
        return prev_result

    gravity_down = _gravity_from_cameras(cameras_json)
    gravity_up = -gravity_down

    # ── Step 1: ADE20K wall-mask segmentation on sampled frames FIRST ─────────
    # Only points an image already classified "wall" are eligible for the
    # plane fit below — segmentation drives the geometry, not the other way
    # around.
    names = list(cameras.keys())
    max_f = settings.HVAC_MAX_WALL_SEG_FRAMES
    if max_f and len(names) > max_f:
        step = (len(names) + max_f - 1) // max_f
        names = names[::step]
    progress_cb(0.10, f"Segmenting wall pixels (ADE20K) on {len(names)} frames…")

    picked_cams = {n: cameras[n] for n in names}
    frames_dir = tmp / "wall_seg_frames"
    local_by_name = await download_registered_frames(storage, prev_result, picked_cams, frames_dir)

    wall_point_indices: set[int] = set()
    # Per contributing frame: how many cloud points it put into the wall pool,
    # and its own cached mask — reused later to render the overlay without a
    # second SegFormer pass over the same image.
    frame_contrib: dict[str, dict] = {}

    n = len(local_by_name)
    for i, (name, local) in enumerate(local_by_name.items()):
        cam = cameras[name]
        img = cv2.imread(str(local))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        wall_mask = segformer_wall_mask(img_rgb)
        if wall_mask is None:
            continue
        idx = points_in_mask(pts, cam, wall_mask)
        if len(idx) < MIN_MASK_POINTS_PER_FRAME:
            continue
        wall_point_indices.update(idx.tolist())
        frame_contrib[name] = {"n_points": len(idx), "mask": wall_mask, "img_bgr": img}

        if i % 10 == 0 or i == n - 1:
            progress_cb(0.10 + 0.35 * (i + 1) / max(n, 1), f"Segmenting: {i+1}/{n} frames…")

    if len(wall_point_indices) < MIN_SEARCH_POINTS:
        logger.warning(
            "[%s] wall_plane_detection: ADE20K found only %d wall-classified points across "
            "%d frames — skipping (no image evidence of a wall)",
            project_id, len(wall_point_indices), n,
        )
        return prev_result

    wall_pts = pts[np.array(sorted(wall_point_indices))]
    progress_cb(0.45, f"{len(wall_pts):,} points confirmed as wall by ADE20K across "
                      f"{len(frame_contrib)} frame(s) — fitting planes…")

    # ── Step 2: fit planes through ONLY the image-confirmed wall points ──────
    remaining = wall_pts.copy()
    candidates: list[dict] = []

    for i in range(MAX_WALL_CANDIDATES):
        if len(remaining) < MIN_SEARCH_POINTS:
            break
        fit = _ransac_plane(remaining, RANSAC_ITERATIONS, WALL_THRESHOLD_M,
                            normal_constraint=gravity_down, normal_min_dot=WALL_NORMAL_MIN_DOT)
        if fit is None:
            break
        normal, point_on_plane, mask = fit
        inliers = int(mask.sum())

        tilt_dot = abs(float(np.dot(normal, gravity_down)))
        if tilt_dot > POST_FIT_MAX_TILT_DOT:
            # Not actually vertical enough to trust — drop these points and keep
            # searching rather than accepting a bad candidate.
            remaining = remaining[~mask]
            continue

        offset_m = float(np.dot(normal, point_on_plane))
        if not _is_duplicate(normal, offset_m, candidates):
            # Extent in the wall's own (u, v) basis — needed by the frontend to
            # size the highlighted region; visualization only.
            u_ax, v_ax = wall_basis(normal, gravity_up)
            rel = remaining[mask] - point_on_plane
            u_vals, v_vals = rel @ u_ax, rel @ v_ax
            candidates.append({
                "normal": normal.tolist(),
                "point_on_plane_m": point_on_plane.tolist(),   # already metric
                "offset_m": offset_m,
                "inliers": inliers,
                "meets_min_inliers": inliers >= settings.HVAC_MIN_WALL_INLIERS,
                "ade20k_wall_confidence": None,   # filled in below
                "extent_u_m": [float(u_vals.min()), float(u_vals.max())],
                "extent_v_m": [float(v_vals.min()), float(v_vals.max())],
            })
        remaining = remaining[~mask]

    candidates.sort(key=lambda c: c["inliers"], reverse=True)
    progress_cb(0.75, f"Found {len(candidates)} wall candidate(s), "
                     f"{sum(c['meets_min_inliers'] for c in candidates)} above evidence floor…")

    # ── Step 3: pick each candidate's best contributing frame for the overlay ─
    # No second SegFormer pass — reuse the masks computed in step 1. "Best" =
    # the frame whose own wall mask contributed the most points to THIS
    # specific candidate's plane (not just biggest mask overall).
    for ci, c in enumerate(candidates[:3]):
        normal = np.array(c["normal"])
        point_on_plane = np.array(c["point_on_plane_m"])
        best_name, best_count = None, 0
        for name, info in frame_contrib.items():
            cam = cameras[name]
            idx = points_in_mask(pts, cam, info["mask"])
            if len(idx) == 0:
                continue
            candidate_pts = pts[idx]
            dists = np.abs((candidate_pts - point_on_plane) @ normal)
            on_this_wall = int((dists < WALL_THRESHOLD_M).sum())
            if on_this_wall > best_count:
                best_count, best_name = on_this_wall, name
        if best_name is None:
            continue
        info = frame_contrib[best_name]
        c["ade20k_wall_confidence"] = float(info["mask"].mean())
        c["ade20k_frame"] = best_name

        overlay = info["img_bgr"].copy()
        overlay[info["mask"]] = (0.5 * overlay[info["mask"]] + 0.5 * np.array([0, 200, 0])).astype(np.uint8)
        overlay_local = tmp / f"wall_ade20k_{ci}.jpg"
        cv2.imwrite(str(overlay_local), overlay)
        overlay_key = f"{project_id}/hvac/wall_ade20k_{ci}.jpg"
        await storage.upload(overlay_local, overlay_key)
        c["ade20k_overlay_key"] = overlay_key

    progress_cb(1.0, f"Wall detection complete: {len(candidates)} candidate(s)")

    result = dict(prev_result)
    result["wall_candidates"] = candidates
    return result
