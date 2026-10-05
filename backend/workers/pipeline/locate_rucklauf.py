"""
HVAC stage 3 — locate the Rücklauf (return pipe) and Vorlauf (supply pipe).

Two passes, cheapest first:
  1. Color-cue detection — convention: a small blue cap marks the Rücklauf,
     red marks the Vorlauf. Pure cv2 HSV
     thresholding + blob geometry, no model load.
  2. GDINO fallback, only if pass 1 finds nothing — but no second detection
     sweep is run here: detect_hvac_fixtures.py already prompts GDINO with
     "return pipe"/"supply pipe" as part of its one full-frame pass, storing
     results under hvac_fixtures["rucklauf_candidates"]/["vorlauf_candidates"].
     This stage just reads the highest-confidence instance from there.

If neither pass finds anything, rucklauf_position/vorlauf_position are left
None — hvac_placement.py then runs in free-wall-space mode (maximize clearance).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# HSV thresholds for the cap-color convention. Generous ranges tuned against
# typical printed/plastic cap colors under indoor lighting — may need retuning
# on real footage.
_BLUE_HSV = (np.array([95, 80, 50]), np.array([130, 255, 255]))
_RED_HSV_LO = (np.array([0, 80, 50]), np.array([10, 255, 255]))
_RED_HSV_HI = (np.array([170, 80, 50]), np.array([179, 255, 255]))

MIN_BLOB_AREA_PX = 20
MAX_BLOB_AREA_PX = 3000
MIN_CIRCULARITY = 0.55       # blob_area / enclosing_circle_area
# A round cap seen from any angle is an ellipse, which always fills π/4 ≈ 0.785
# of its minimum-area rectangle; squares/rectangles (poster cells, tiles, labels)
# fill ~1.0 — and a square also passes MIN_CIRCULARITY (2/π ≈ 0.64).
MAX_RECT_FILL = 0.88
MIN_POINTS_PER_CUE = 10
CLUSTER_EPS_M = 0.25
SNAP_TO_PIPE_RADIUS_M = 0.40  # snap a cue detection onto a nearby stage-2 pipe cluster


def _find_color_blobs(img_bgr: "np.ndarray", ranges: list[tuple]) -> list[tuple[float, float, float]]:
    """Returns [(cx_px, cy_px, radius_px), ...] for small, roughly-circular
    blobs matching any of the given HSV (lo, hi) ranges."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, lo, hi)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_BLOB_AREA_PX <= area <= MAX_BLOB_AREA_PX):
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        circle_area = np.pi * radius * radius
        if circle_area <= 0 or area / circle_area < MIN_CIRCULARITY:
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
        if rw * rh <= 0 or area / (rw * rh) > MAX_RECT_FILL:
            continue
        blobs.append((float(cx), float(cy), float(radius)))
    return blobs


def _cluster_observations(obs: list[dict]) -> Optional[dict]:
    """Pick the largest-support cluster of cue observations (mean position,
    provenance). Returns None if obs is empty."""
    import open3d as o3d

    if not obs:
        return None
    centroids = np.array([o["centroid_m"] for o in obs])
    if len(centroids) == 1:
        db_labels = np.array([0])
    else:
        cpcd = o3d.geometry.PointCloud()
        cpcd.points = o3d.utility.Vector3dVector(centroids)
        db_labels = np.array(cpcd.cluster_dbscan(eps=CLUSTER_EPS_M, min_points=1))

    best_members: list[dict] = []
    for cluster_id in sorted(set(db_labels.tolist())):
        if cluster_id < 0:
            continue
        members = [obs[i] for i in range(len(obs)) if db_labels[i] == cluster_id]
        if len(members) > len(best_members):
            best_members = members
    if not best_members:
        return None

    member_pts = np.array([m["centroid_m"] for m in best_members])
    # Largest blob (by box area) is the clearest photo to show, not just any member.
    best = max(best_members, key=lambda m: (m["box_frac"][2] - m["box_frac"][0]) * (m["box_frac"][3] - m["box_frac"][1]))
    return {
        "position_m": member_pts.mean(axis=0).tolist(),
        "n_observations": len(best_members),
        "source_frames": sorted({m["frame"] for m in best_members}),
        "method": "color_cue",
        "best_frame": best["frame"],
        "best_box_frac": best["box_frac"],
    }


def _snap_to_pipe(position: dict, pipes: list[dict]) -> dict:
    """If a nearby stage-2 pipe cluster exists, snap onto it (more representative
    of the actual pipe's extent than a single cap-blob centroid) while keeping
    the cue-based provenance for transparency."""
    if not pipes:
        return position
    pos = np.array(position["position_m"])
    dists = [float(np.linalg.norm(pos - np.array(p["position_m"]))) for p in pipes]
    j = int(np.argmin(dists))
    if dists[j] <= SNAP_TO_PIPE_RADIUS_M:
        snapped = dict(position)
        snapped["position_m"] = pipes[j]["position_m"]
        snapped["snapped_to_pipe_cluster"] = True
        return snapped
    return position


async def run_locate_rucklauf(
    project_id: str,
    prev_result: dict,
    tmp: Path,
    progress_cb: Callable[[float, str], None],
) -> dict:
    """
    Returns prev_result augmented with:
        rucklauf_position: {position_m, n_observations, source_frames, method} | None
        vorlauf_position:  same shape | None
    """
    import open3d as o3d
    from backend.core.storage import get_storage
    from backend.workers.pipeline.depth_fusion_common import load_registered_cameras
    from backend.workers.pipeline.hvac_common import (
        download_registered_frames, points_in_box,
    )
    from backend.core.config import settings

    storage = get_storage()

    cloud_key = prev_result.get("dense_cloud_key") or prev_result.get("scaled_cloud_key", "")
    cameras_key = prev_result.get("camera_poses_key", f"{project_id}/sfm/cameras.json")
    if not cloud_key:
        logger.warning("[%s] locate_rucklauf: no dense cloud key — skipping", project_id)
        return prev_result

    progress_cb(0.05, "Loading refined cloud and camera poses…")
    cloud_local = tmp / "rucklauf_cloud.ply"
    cameras_local = tmp / "rucklauf_cameras.json"
    await storage.download(cloud_key, cloud_local)
    await storage.download(cameras_key, cameras_local)

    pcd = o3d.io.read_point_cloud(str(cloud_local))
    pts = np.asarray(pcd.points)   # already metric (post refine_cloud) — see hvac_common docstring
    scale_factor = float(prev_result.get("confirmed_scale_factor") or 1.0)
    cameras_json = json.loads(cameras_local.read_text())
    cameras = load_registered_cameras(cameras_json, scale_factor=scale_factor)
    if len(pts) == 0 or not cameras:
        logger.warning("[%s] locate_rucklauf: empty cloud or no cameras — skipping", project_id)
        return prev_result

    names = list(cameras.keys())
    max_f = settings.HVAC_MAX_DETECTION_FRAMES
    if max_f and len(names) > max_f:
        step = (len(names) + max_f - 1) // max_f
        names = names[::step]

    progress_cb(0.15, f"Scanning {len(names)} frames for Rücklauf/Vorlauf color cues…")
    picked_cams = {n: cameras[n] for n in names}
    frames_dir = tmp / "rucklauf_frames"
    local_by_name = await download_registered_frames(storage, prev_result, picked_cams, frames_dir)

    blue_obs: list[dict] = []
    red_obs: list[dict] = []
    n = len(local_by_name)
    for i, (name, local) in enumerate(local_by_name.items()):
        img = cv2.imread(str(local))
        if img is None:
            continue
        cam = cameras[name]

        for ranges, obs_list in [([_BLUE_HSV], blue_obs), ([_RED_HSV_LO, _RED_HSV_HI], red_obs)]:
            for cx, cy, r in _find_color_blobs(img, ranges):
                box = [cx - r, cy - r, cx + r, cy + r]
                idx = points_in_box(pts, cam, box)
                if len(idx) < MIN_POINTS_PER_CUE:
                    continue
                centroid_m = pts[idx].mean(axis=0)   # pts already metric
                obs_list.append({
                    "centroid_m": centroid_m.tolist(),
                    "frame": name,
                    "n_points": int(len(idx)),
                    "box_frac": [box[0] / cam["width"], box[1] / cam["height"],
                                 box[2] / cam["width"], box[3] / cam["height"]],
                })
        if i % 20 == 0 or i == n - 1:
            progress_cb(0.15 + 0.55 * (i + 1) / max(n, 1), f"Color-cue scan: {i+1}/{n} frames…")

    rucklauf = _cluster_observations(blue_obs)
    vorlauf = _cluster_observations(red_obs)

    fixtures = prev_result.get("hvac_fixtures", {})
    pipes = fixtures.get("pipes", [])

    if rucklauf is None:
        progress_cb(0.75, "No Rücklauf color cue found — trying GDINO fallback…")
        candidates = fixtures.get("rucklauf_candidates", [])
        if candidates:
            best = max(candidates, key=lambda c: c["max_score"])
            rucklauf = {
                "position_m": best["position_m"],
                "n_observations": best["n_observations"],
                "source_frames": best["source_frames"],
                "method": "gdino_fallback",
                "best_frame": best.get("best_frame"),
                "best_box_frac": best.get("best_box_frac"),
            }
    else:
        rucklauf = _snap_to_pipe(rucklauf, pipes)

    if vorlauf is None:
        candidates = fixtures.get("vorlauf_candidates", [])
        if candidates:
            best = max(candidates, key=lambda c: c["max_score"])
            vorlauf = {
                "position_m": best["position_m"],
                "n_observations": best["n_observations"],
                "source_frames": best["source_frames"],
                "method": "gdino_fallback",
                "best_frame": best.get("best_frame"),
                "best_box_frac": best.get("best_box_frac"),
            }
    else:
        vorlauf = _snap_to_pipe(vorlauf, pipes)

    status = "found" if rucklauf else "not_found"
    progress_cb(1.0, f"Rücklauf: {status}" + (f" ({rucklauf['method']})" if rucklauf else ""))

    result = dict(prev_result)
    result["rucklauf_position"] = rucklauf
    result["vorlauf_position"] = vorlauf
    return result
