"""
HVAC stage 2 — GDINO(+SAM2) fixture/obstacle/window detection, lifted to 3D.

A dense, metric-scaled MVS cloud exists by this point in the chain — so 2D detections are
lifted to 3D by projecting the DENSE cloud into each detection frame and
collecting the points that fall inside the detection's box/mask, not by
blind ray-triangulation across views. This sidesteps the cross-frame
instance-correspondence problem entirely (no need to decide "is this the
same pipe as that one in another frame" before you have any 3D points to
compare) and is the same per-frame point-cloud projection formula
lingbot_fusion.py's `_fuse()` already uses to calibrate depth against COLMAP.

Same-label detections across frames are merged into instances afterward via
DBSCAN on their metric-space centroids — a much easier clustering problem
once real 3D points exist, per detection, per frame.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from backend.core.config import settings

logger = logging.getLogger(__name__)

PROMPTS = [
    "pipe", "heating pipe", "valve", "radiator", "electrical outlet", "window",
    "return pipe", "supply pipe",   # Rücklauf/Vorlauf GDINO fallback — piggybacked
    # onto this stage's single detection pass rather than a second full-frame
    # sweep in locate_rucklauf.py; that stage reads rucklauf_candidates/
    # vorlauf_candidates from here only if its own cheap color-cue pass finds nothing.
]

# Canonical fixture classes this stage reports, and the substrings of a raw
# GDINO label that map to each (checked in order — longer/more specific phrases
# before their shorter substrings, e.g. "heating pipe" and "return pipe" before "pipe").
_LABEL_MAP: list[tuple[str, str]] = [
    ("return pipe", "rucklauf_candidates"),
    ("supply pipe", "vorlauf_candidates"),
    ("heating pipe", "pipes"),
    ("pipe", "pipes"),
    ("valve", "valves"),
    ("radiator", "radiators"),
    ("electrical outlet", "electrical"),
    ("window", "windows"),
]

MIN_POINTS_PER_DETECTION = 20    # ignore detections the dense cloud barely covers
CLUSTER_EPS_M = 0.30             # merge same-label detections within this radius into one instance
BOX_THRESHOLD = 0.30


def _canonical_label(raw: str) -> str:
    raw_l = raw.lower()
    for substr, canon in _LABEL_MAP:
        if substr in raw_l:
            return canon
    return "other"


async def run_detect_hvac_fixtures(
    project_id: str,
    prev_result: dict,
    tmp: Path,
    progress_cb: Callable[[float, str], None],
) -> dict:
    """
    Returns prev_result augmented with:
        hvac_fixtures: {"pipes": [...], "valves": [...], "radiators": [...],
                         "electrical": [...], "windows": [...], "other": [...]}
        each entry: {position_m: [x,y,z], n_observations, n_points_total,
                      max_score, source_frames: [...]}
    """
    import open3d as o3d
    from backend.core.storage import get_storage
    from backend.workers.pipeline.depth_fusion_common import load_registered_cameras
    from backend.workers.pipeline.hvac_common import (
        download_registered_frames,
        detect_with_gdino, refine_box_to_mask, points_in_box, points_in_mask,
    )

    storage = get_storage()

    cloud_key = prev_result.get("dense_cloud_key") or prev_result.get("scaled_cloud_key", "")
    cameras_key = prev_result.get("camera_poses_key", f"{project_id}/sfm/cameras.json")
    if not cloud_key:
        logger.warning("[%s] detect_hvac_fixtures: no dense cloud key — skipping", project_id)
        return prev_result

    progress_cb(0.03, "Loading refined cloud and camera poses…")
    cloud_local = tmp / "fixtures_cloud.ply"
    cameras_local = tmp / "fixtures_cameras.json"
    await storage.download(cloud_key, cloud_local)
    await storage.download(cameras_key, cameras_local)

    pcd = o3d.io.read_point_cloud(str(cloud_local))
    pts = np.asarray(pcd.points)   # already metric (post refine_cloud) — see hvac_common docstring
    scale_factor = float(prev_result.get("confirmed_scale_factor") or 1.0)
    cameras_json = json.loads(cameras_local.read_text())
    cameras = load_registered_cameras(cameras_json, scale_factor=scale_factor)
    if len(pts) == 0 or not cameras:
        logger.warning("[%s] detect_hvac_fixtures: empty cloud or no cameras — skipping", project_id)
        return prev_result

    names = list(cameras.keys())
    max_f = settings.HVAC_MAX_DETECTION_FRAMES
    if max_f and len(names) > max_f:
        step = (len(names) + max_f - 1) // max_f
        names = names[::step]
        logger.info("[%s] detect_hvac_fixtures: subsampling %d -> %d frames (step %d)",
                    project_id, len(cameras), len(names), step)

    progress_cb(0.10, f"Downloading {len(names)} frames for detection…")
    picked_cams = {n: cameras[n] for n in names}
    frames_dir = tmp / "hvac_fixture_frames"
    local_by_name = await download_registered_frames(storage, prev_result, picked_cams, frames_dir)

    raw_detections: dict[str, list[dict]] = {canon: [] for _, canon in _LABEL_MAP}
    raw_detections["other"] = []

    n = len(local_by_name)
    for i, (name, local) in enumerate(local_by_name.items()):
        cam = cameras[name]
        img = cv2.imread(str(local))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            dets = detect_with_gdino(img_rgb, PROMPTS, box_threshold=BOX_THRESHOLD)
        except Exception as e:
            logger.warning("[%s] detect_hvac_fixtures: GDINO failed on %s: %s", project_id, name, e)
            continue
        if not dets:
            if i % 20 == 0 or i == n - 1:
                progress_cb(0.10 + 0.75 * (i + 1) / max(n, 1), f"Detecting: {i+1}/{n} frames…")
            continue

        for det in dets:
            label = _canonical_label(det["label"])
            box = det["box"]
            mask = refine_box_to_mask(img_rgb, box) if settings.HVAC_ENABLE_SAM2 else None
            idx = points_in_mask(pts, cam, mask) if mask is not None else points_in_box(pts, cam, box)
            if len(idx) < MIN_POINTS_PER_DETECTION:
                continue
            centroid_m = pts[idx].mean(axis=0)   # pts already metric
            x0, y0, x1, y1 = box
            raw_detections[label].append({
                "centroid_m": centroid_m.tolist(),
                "n_points": int(len(idx)),
                "score": det["score"],
                "frame": name,
                # Fractional [0,1] box so the frontend can draw it over the
                # frame image at any display size without needing width/height.
                "box_frac": [x0 / cam["width"], y0 / cam["height"], x1 / cam["width"], y1 / cam["height"]],
            })

        if i % 20 == 0 or i == n - 1:
            progress_cb(0.10 + 0.75 * (i + 1) / max(n, 1), f"Detecting: {i+1}/{n} frames…")

    progress_cb(0.90, "Clustering detections into instances…")
    fixtures: dict[str, list[dict]] = {}
    for label, dets in raw_detections.items():
        if not dets:
            fixtures[label] = []
            continue
        centroids = np.array([d["centroid_m"] for d in dets])
        if len(centroids) == 1:
            db_labels = np.array([0])
        else:
            cpcd = o3d.geometry.PointCloud()
            cpcd.points = o3d.utility.Vector3dVector(centroids)
            db_labels = np.array(cpcd.cluster_dbscan(eps=CLUSTER_EPS_M, min_points=1))

        instances = []
        for cluster_id in sorted(set(db_labels.tolist())):
            if cluster_id < 0:
                continue
            members = [dets[i] for i in range(len(dets)) if db_labels[i] == cluster_id]
            member_pts = np.array([m["centroid_m"] for m in members])
            best = max(members, key=lambda m: m["score"])
            instances.append({
                "position_m": member_pts.mean(axis=0).tolist(),
                "n_observations": len(members),
                "n_points_total": int(sum(m["n_points"] for m in members)),
                "max_score": round(max(m["score"] for m in members), 3),
                "source_frames": sorted({m["frame"] for m in members}),
                # Best-scoring observation's frame + box, for drawing the
                # detection on the actual photo in the frontend.
                "best_frame": best["frame"],
                "best_box_frac": best["box_frac"],
            })
        fixtures[label] = instances

    n_total = sum(len(v) for v in fixtures.values())
    progress_cb(1.0, f"HVAC fixtures: {n_total} instance(s) across {len(fixtures)} classes")

    result = dict(prev_result)
    result["hvac_fixtures"] = fixtures
    return result
