"""
HVAC stage 4 — wall-mount placement recommendation.

Pure numpy/open3d, no model load. Grid-searches HVAC_UNIT_SIZE_CM candidate
rectangles on each wall candidate (wall_plane_detection.py) that clears
HVAC_MIN_WALL_INLIERS, rejecting cells that overlap a detected fixture
keep-out (detect_hvac_fixtures.py) or violate HVAC_MIN_CLEARANCE_CM, then
scores survivors by distance to the Rücklauf (locate_rucklauf.py) — or, if no
Rücklauf was found, by maximizing clearance (free-wall-space mode).

The winning candidate is rendered onto whichever registered frame projects it
as a clean, fully-contained, unoccluded quadrilateral (hvac_common.quad_is_clean
+ _is_occluded) and uploaded as the overlay image the Segmentation tab shows.
Every candidate keeps corners_world_m so the 3D viewer can draw it on the
cloud/mesh.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from backend.core.config import settings
from backend.workers.pipeline.room_layout import WALL_THRESHOLD_M

logger = logging.getLogger(__name__)

OBSTACLE_KEEPOUT_RADIUS_M = 0.15   # treat each detected fixture as occupying this radius
OBSTACLE_PLANE_TOLERANCE_M = 0.50  # ignore fixtures this far from a given wall's plane
GRID_STEP_FRACTION = 0.5           # grid step = this fraction of the unit's own size
TOP_N_CANDIDATES = 5

# Occlusion test for overlay-frame selection (found necessary 2026-09-05: a
# geometrically valid, high-confidence wall candidate rendered as if floating
# on a bed/backpack, because quad_is_clean only checks the projected quad's
# OWN shape — convexity, containment, diagonal ratio — never whether real
# cloud geometry sits between the camera and that wall spot in the chosen
# photo. A small per-candidate depth test closes that gap.
OCCLUSION_GRID = 5                # 5x5 sample points across the candidate rectangle
OCCLUSION_PIXEL_RADIUS = 4        # px window used to find "something near this screen position"
OCCLUSION_DEPTH_MARGIN_M = 0.05   # ignore depth differences smaller than this (surface noise)
MAX_OCCLUDED_FRACTION = 0.15      # reject the frame if more than this fraction of samples are blocked


def _parse_unit_size_m() -> tuple[float, float]:
    w_cm, h_cm = settings.HVAC_UNIT_SIZE_CM.lower().split("x")
    return float(w_cm) / 100.0, float(h_cm) / 100.0


def _collect_obstacle_uv(
    fixtures: dict, normal: np.ndarray, point_on_plane: np.ndarray,
    u_ax: np.ndarray, v_ax: np.ndarray,
) -> list[tuple[float, float]]:
    """(u, v) positions of every fixture instance close enough to this wall's
    plane to matter for its obstacle grid — pipes/valves/radiators/electrical/
    windows all treated as keep-outs (see module docstring)."""
    obstacles_uv = []
    for label, instances in fixtures.items():
        if label in ("rucklauf_candidates", "vorlauf_candidates"):
            continue   # pipe-prompt candidates, not obstacles in their own right
        for inst in instances:
            pos = np.array(inst["position_m"])
            plane_dist = abs(float(np.dot(pos - point_on_plane, normal)))
            if plane_dist > OBSTACLE_PLANE_TOLERANCE_M:
                continue
            rel = pos - point_on_plane
            obstacles_uv.append((float(rel @ u_ax), float(rel @ v_ax)))
    return obstacles_uv


def _search_wall(
    candidate: dict, pts_metric: np.ndarray, fixtures: dict,
    gravity_up: np.ndarray, floor_pt: Optional[np.ndarray],
    rucklauf_pos: Optional[np.ndarray], unit_w: float, unit_h: float,
) -> list[dict]:
    normal = np.array(candidate["normal"])
    point_on_plane = np.array(candidate["point_on_plane_m"])

    dists = np.abs((pts_metric - point_on_plane) @ normal)
    inlier_pts = pts_metric[dists < WALL_THRESHOLD_M]
    if len(inlier_pts) < 50:
        return []

    from backend.workers.pipeline.hvac_common import wall_basis
    u_ax, v_ax = wall_basis(normal, gravity_up)
    rel = inlier_pts - point_on_plane
    u_vals, v_vals = rel @ u_ax, rel @ v_ax
    u_min, u_max = float(u_vals.min()), float(u_vals.max())
    v_min, v_max = float(v_vals.min()), float(v_vals.max())
    if (u_max - u_min) < unit_w or (v_max - v_min) < unit_h:
        return []   # wall smaller than the unit itself

    obstacles_uv = _collect_obstacle_uv(fixtures, normal, point_on_plane, u_ax, v_ax)
    cell_half_diag = float(np.hypot(unit_w, unit_h) / 2)

    step_u = max(unit_w * GRID_STEP_FRACTION, 0.05)
    step_v = max(unit_h * GRID_STEP_FRACTION, 0.05)
    u_starts = np.arange(u_min, u_max - unit_w + 1e-9, step_u)
    v_starts = np.arange(v_min, v_max - unit_h + 1e-9, step_v)
    results = []

    for u0 in u_starts:
        for v0 in v_starts:
            cu, cv = u0 + unit_w / 2, v0 + unit_h / 2
            blocked = False
            min_gap = None
            for ou, ov in obstacles_uv:
                gap = float(np.hypot(cu - ou, cv - ov)) - cell_half_diag - OBSTACLE_KEEPOUT_RADIUS_M
                if gap < 0:
                    blocked = True
                    break
                if min_gap is None or gap < min_gap:
                    min_gap = gap
            if blocked:
                continue
            clearance_cm = round((min_gap if min_gap is not None else 9.99) * 100, 1)
            if clearance_cm < settings.HVAC_MIN_CLEARANCE_CM:
                continue

            center_world = point_on_plane + cu * u_ax + cv * v_ax
            mount_height_cm = (
                round(float(np.dot(center_world - floor_pt, gravity_up)) * 100, 1)
                if floor_pt is not None else None
            )
            distance_to_rucklauf_cm = (
                round(float(np.linalg.norm(center_world - rucklauf_pos)) * 100, 1)
                if rucklauf_pos is not None else None
            )
            corners_uv = [(u0, v0), (u0 + unit_w, v0), (u0 + unit_w, v0 + unit_h), (u0, v0 + unit_h)]
            corners_world = [point_on_plane + uu * u_ax + vv * v_ax for uu, vv in corners_uv]
            results.append({
                "wall_inliers": candidate["inliers"],
                "clearance_cm": clearance_cm,
                "mount_height_cm": mount_height_cm,
                "distance_to_rucklauf_cm": distance_to_rucklauf_cm,
                "center_world_m": center_world.tolist(),
                "corners_world_m": [c.tolist() for c in corners_world],
                "score": distance_to_rucklauf_cm if distance_to_rucklauf_cm is not None else -clearance_cm,
            })
    return results


def _sample_quad_grid(corners_world_m: list[np.ndarray], n: int) -> list[np.ndarray]:
    """n x n bilinear grid of points across the rectangle C0,C1,C2,C3 (that
    winding order — C0, C1=C0+u_axis, C2=C0+u_axis+v_axis, C3=C0+v_axis)."""
    c0, c1, c2, c3 = corners_world_m
    pts = []
    for i in range(n):
        s = i / (n - 1)
        for j in range(n):
            t = j / (n - 1)
            pts.append((1 - s) * (1 - t) * c0 + s * (1 - t) * c1 + s * t * c2 + (1 - s) * t * c3)
    return pts


def _is_occluded(corners_world_m: list[np.ndarray], cam: dict, pts_metric: np.ndarray) -> bool:
    """
    True if real cloud geometry sits between this camera and enough of the
    candidate rectangle's surface — furniture in front of a wall spot that is
    itself geometrically valid. Projects the whole cloud into this frame once
    (vectorized), then for each of a small grid of points sampled across the
    candidate rectangle, checks whether any cloud point lands near the same
    screen position but at meaningfully shallower depth.
    """
    from backend.workers.pipeline.hvac_common import project_cloud_to_frame_depth

    K, R, t = cam["K"], cam["R"], cam["t"]
    u_all, v_all, z_all, inb_all = project_cloud_to_frame_depth(pts_metric, cam)

    samples = _sample_quad_grid(corners_world_m, OCCLUSION_GRID)
    occluded = 0
    for p in samples:
        p_cam = R @ p + t
        if p_cam[2] <= 1e-6:
            occluded += 1
            continue
        target_depth = p_cam[2]
        u_px = K[0, 0] * p_cam[0] / target_depth + K[0, 2]
        v_px = K[1, 1] * p_cam[1] / target_depth + K[1, 2]
        near = inb_all & (np.abs(u_all - u_px) < OCCLUSION_PIXEL_RADIUS) & (np.abs(v_all - v_px) < OCCLUSION_PIXEL_RADIUS)
        if near.any() and float(z_all[near].min()) < target_depth - OCCLUSION_DEPTH_MARGIN_M:
            occluded += 1
    return (occluded / len(samples)) > MAX_OCCLUDED_FRACTION


def _pick_overlay_frame(
    corners_world_m: list[list[float]], cameras: dict, pts_metric: np.ndarray,
) -> Optional[tuple[str, list[tuple[float, float]]]]:
    """Best registered frame that projects the 4 corners as a clean, fully-
    contained quad (hvac_common.quad_is_clean — convexity, containment,
    diagonal ratio) AND isn't blocked by real foreground geometry in that
    specific photo (_is_occluded — neither check catches what the other does:
    a skewed-but-unobstructed quad passes quad_is_clean and fails occlusion
    only if something is genuinely in front of it, and vice versa)."""
    from backend.workers.pipeline.hvac_common import project_point_to_pixel, quad_is_clean

    best_name, best_area, best_px = None, -1.0, None
    corners = [np.array(c) for c in corners_world_m]
    for name, cam in cameras.items():
        px = [project_point_to_pixel(c, cam["K"], cam["R"], cam["t"]) for c in corners]
        if not quad_is_clean(px, cam["width"], cam["height"]):
            continue
        if _is_occluded(corners, cam, pts_metric):
            continue
        pts = np.array(px)
        x, y = pts[:, 0], pts[:, 1]
        area = abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2  # shoelace formula
        if area > best_area:
            best_name, best_area, best_px = name, area, px
    if best_name is None:
        return None
    return best_name, best_px


async def run_hvac_placement(
    project_id: str,
    prev_result: dict,
    tmp: Path,
    progress_cb: Callable[[float, str], None],
) -> dict:
    """
    Returns prev_result augmented with:
        hvac_placement: {status, candidates: [...], overlay_image_key}
    """
    import open3d as o3d
    from backend.core.storage import get_storage
    from backend.workers.pipeline.depth_fusion_common import load_registered_cameras
    from backend.workers.pipeline.room_layout import _gravity_from_cameras

    storage = get_storage()

    def _passes_ade20k(c: dict) -> bool:
        # None means this candidate was never checked (only the top 3 by
        # inliers get an ADE20K pass in wall_plane_detection.py) — give it the
        # benefit of the doubt. A checked-and-failed confidence is a real,
        # specific signal from image segmentation that this is not a wall
        # (confirmed on real footage: a bed's mattress edge cleared the
        # inlier floor and the 25° tilt gate, but ADE20K had flagged it at
        # 3% wall-pixel confidence) — exclude it outright, don't just down-rank.
        conf = c.get("ade20k_wall_confidence")
        return conf is None or conf >= settings.HVAC_MIN_ADE20K_CONFIDENCE

    wall_candidates = [c for c in prev_result.get("wall_candidates", [])
                       if c.get("meets_min_inliers") and _passes_ade20k(c)]
    if not wall_candidates:
        logger.info("[%s] hvac_placement: no wall candidate meets the evidence floor", project_id)
        result = dict(prev_result)
        result["hvac_placement"] = {"status": "no_wall_evidence", "candidates": [], "overlay_image_key": None}
        return result

    cloud_key = prev_result.get("dense_cloud_key") or prev_result.get("scaled_cloud_key", "")
    cameras_key = prev_result.get("camera_poses_key", f"{project_id}/sfm/cameras.json")
    progress_cb(0.05, "Loading refined cloud and camera poses…")
    cloud_local = tmp / "placement_cloud.ply"
    cameras_local = tmp / "placement_cameras.json"
    await storage.download(cloud_key, cloud_local)
    await storage.download(cameras_key, cameras_local)

    pcd = o3d.io.read_point_cloud(str(cloud_local))
    scale_factor = float(prev_result.get("confirmed_scale_factor") or 1.0)
    pts_metric = np.asarray(pcd.points)   # already metric (post refine_cloud)

    cameras_json = json.loads(cameras_local.read_text())
    cameras = load_registered_cameras(cameras_json, scale_factor=scale_factor)
    gravity_up = -_gravity_from_cameras(cameras_json)

    # fitted_planes' offset is genuinely in SfM units (triangulated by
    # scale_from_aruco.py before scale was known — see room_layout.py's own
    # identical conversion), unlike the already-metric dense cloud above.
    floor_entry = next((p for p in prev_result.get("fitted_planes", []) if p.get("label") == "floor"), None)
    floor_pt = None
    if floor_entry:
        fn = np.array(floor_entry["normal"], dtype=float)
        fn /= np.linalg.norm(fn) + 1e-9
        floor_pt = fn * float(floor_entry["offset"]) * scale_factor
    else:
        # No marker floor plane (grid-marker scans, or /reprocess — fitted_planes
        # isn't persisted): take the floor as the low end of the cloud along
        # gravity, percentile-clipped like dimensions.py's height.
        floor_pt = gravity_up * float(np.percentile(pts_metric @ gravity_up, 0.5))

    rucklauf = prev_result.get("rucklauf_position")
    rucklauf_pos = np.array(rucklauf["position_m"]) if rucklauf else None
    fixtures = prev_result.get("hvac_fixtures", {})
    unit_w, unit_h = _parse_unit_size_m()

    progress_cb(0.2, f"Searching {len(wall_candidates)} wall(s) for a {settings.HVAC_UNIT_SIZE_CM}cm mounting spot…")
    all_results: list[dict] = []
    for candidate in sorted(wall_candidates, key=lambda c: c["inliers"], reverse=True):
        all_results.extend(_search_wall(
            candidate, pts_metric, fixtures, gravity_up, floor_pt, rucklauf_pos, unit_w, unit_h,
        ))

    if not all_results:
        status = "no_rucklauf_free_mode" if rucklauf_pos is None else "no_fit"
        progress_cb(1.0, f"No valid mounting spot found ({status})")
        result = dict(prev_result)
        result["hvac_placement"] = {"status": status, "candidates": [], "overlay_image_key": None}
        return result

    all_results.sort(key=lambda r: r["score"])
    top = all_results[:TOP_N_CANDIDATES]
    for rank, r in enumerate(top, start=1):
        r["rank"] = rank

    progress_cb(0.7, "Rendering overlay for rank-1 candidate…")
    overlay_key = None
    winner = top[0]
    picked = _pick_overlay_frame(winner["corners_world_m"], cameras, pts_metric)
    if picked is not None:
        name, px = picked
        image_keys = prev_result.get("image_keys", prev_result.get("frame_keys", []))
        key_by_name = {Path(k).name: k for k in image_keys}
        storage_key = key_by_name.get(name)
        if storage_key:
            local = tmp / name
            await storage.download(storage_key, local)
            img = cv2.imread(str(local))
            if img is not None:
                pts_px = np.array(px, dtype=np.int32)
                cv2.polylines(img, [pts_px], isClosed=True, color=(0, 220, 0), thickness=3)
                label = f"#1  d(Rucklauf)={winner['distance_to_rucklauf_cm']}cm  clearance={winner['clearance_cm']}cm"
                cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
                ts = int(time.time())
                overlay_local = tmp / "placement_overlay.jpg"
                cv2.imwrite(str(overlay_local), img)
                overlay_key = f"{project_id}/hvac/placement_{ts}.jpg"
                await storage.upload(overlay_local, overlay_key)

    status = "placed" if rucklauf_pos is not None else "no_rucklauf_free_mode"
    progress_cb(1.0, f"HVAC placement: {status}, {len(top)} candidate(s) ranked")

    # Keep corners_world_m (4 points x 3 floats, negligible size) so the 3D
    # viewer can draw each candidate on the cloud/mesh — everything else is
    # trimmed, the frontend only needs the summary fields.
    slim_candidates = [{
        "rank": r["rank"], "wall_inliers": r["wall_inliers"],
        "clearance_cm": r["clearance_cm"], "mount_height_cm": r["mount_height_cm"],
        "distance_to_rucklauf_cm": r["distance_to_rucklauf_cm"],
        "corners_world_m": r["corners_world_m"],
    } for r in top]

    result = dict(prev_result)
    result["hvac_placement"] = {
        "status": status,
        "candidates": slim_candidates,
        "overlay_image_key": overlay_key,
    }
    return result
