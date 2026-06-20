"""
Stage 6: Coverage Analysis

Scores every point in the cloud by:
  - Number of camera frustums it is visible from (via open3d HiddenPointRemoval)
  - Average cosine similarity between point normal and camera ray
Colors the cloud red→yellow→green and generates shot suggestions via DBSCAN.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np

from backend.core.storage import get_storage

logger = logging.getLogger(__name__)

# Subsample threshold for the visibility pass (O(N log N) per camera).
# Keep well below what caused OOM kills. 75K pts × 25 cams ≈ 1.9M HPR ops,
# comfortably within the 12 GB VRAM / 30 GB RAM budget.
_SUBSAMPLE_THRESHOLD = 500_000
_SUBSAMPLE_TARGET    = 500_000

# Suggested camera standoff distance from a low-coverage surface, in metres.
# The cloud is metric-scaled, so this is a real physical distance — do not
# confuse with `scene_scale` (the bbox diagonal), which is for HPR only.
_SUGGESTION_STANDOFF_M = 1.5
# Max cameras used for HPR scoring.
# HPR camera count: use ~35% of registered cameras, min 25, max 150.
# Hardcoding 25 was correct for ~130-frame videos but badly wrong for 350-frame 8K runs.
def _hpr_camera_count(n_cameras: int) -> int:
    return max(25, min(150, int(n_cameras * 0.35)))


# ── Internal helpers ──────────────────────────────────────────────────────────

def _score_to_rgb(s: np.ndarray) -> np.ndarray:
    """Map per-point scores in [0,1] to red→yellow→green RGB colours."""
    r = np.clip(2 * (1 - s), 0, 1)
    g = np.clip(2 * s, 0, 1)
    b = np.zeros_like(s)
    return np.stack([r, g, b], axis=-1)


def _object_surface_label(
    cluster_centroid: np.ndarray,
    object_center: np.ndarray,
    gravity_up: np.ndarray | None,
) -> str:
    """
    Return a human-readable label for which part of the object a cluster is on,
    relative to the object center and the gravity vector.

    Examples: "top", "bottom", "front-right side", "left side".
    """
    direction = cluster_centroid - object_center
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return "center"
    direction = direction / norm

    if gravity_up is not None:
        up = np.asarray(gravity_up, dtype=float)
        up_norm = np.linalg.norm(up)
        if up_norm > 1e-9:
            up = up / up_norm
            vert = float(np.dot(direction, up))
            if vert > 0.65:
                return "top"
            if vert < -0.65:
                return "bottom"
            # Horizontal component
            horiz = direction - vert * up
            horiz_norm = np.linalg.norm(horiz)
            if horiz_norm < 1e-9:
                return "side"
            horiz = horiz / horiz_norm

            # Build a consistent horizontal basis (arbitrary "forward" as the
            # axis most orthogonal to up; "right" = forward × up).
            # Pick whichever world axis is most orthogonal to up as forward.
            candidates = np.eye(3)
            fwd = candidates[np.argmin(np.abs(candidates @ up))]
            right = np.cross(fwd, up)
            right /= np.linalg.norm(right) + 1e-9

            f_dot = float(np.dot(horiz, fwd))
            r_dot = float(np.dot(horiz, right))

            # Threshold: label as combined only when both components are strong
            h_labels = []
            if abs(f_dot) > 0.45:
                h_labels.append("front" if f_dot > 0 else "back")
            if abs(r_dot) > 0.45:
                h_labels.append("right" if r_dot > 0 else "left")
            horiz_label = "-".join(h_labels) if h_labels else "side"
            prefix = "upper " if vert > 0.25 else ("lower " if vert < -0.25 else "")
            return f"{prefix}{horiz_label}"

    # No gravity — fall back to dominant axis of direction vector
    abs_dir = np.abs(direction)
    axis = int(np.argmax(abs_dir))
    sign = "+" if direction[axis] > 0 else "-"
    return f"{sign}{'XYZ'[axis]} side"


def _parse_camera(cam_entry: dict, cameras_list: list[dict]) -> dict | None:
    """
    Return a dict with intrinsics + extrinsics for one image entry.

    cameras.json stores cameras under a 'cameras' list with 'params'
    holding [f, cx, cy, k] for SIMPLE_RADIAL (or [fx, fy, cx, cy] for
    PINHOLE).  We handle both.
    """
    cam_id = cam_entry.get("camera_id")
    cam = next((c for c in cameras_list if c.get("camera_id") == cam_id), None)
    if cam is None:
        return None

    model = cam.get("model", "SIMPLE_RADIAL")
    params = cam.get("params", [])

    if model in ("SIMPLE_RADIAL", "SIMPLE_PINHOLE") and len(params) >= 3:
        f, cx, cy = params[0], params[1], params[2]
        fx, fy = f, f
    elif model in ("PINHOLE",) and len(params) >= 4:
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    elif len(params) >= 3:
        # Best guess: treat first param as focal
        f = params[0]; cx, cy = params[1], params[2]
        fx, fy = f, f
    else:
        return None

    width = cam.get("width", 0)
    height = cam.get("height", 0)

    # COLMAP MVS runs at max_image_size but cameras.json stores original
    # intrinsics. Scale fx/fy/cx/cy/width/height to match the MVS resolution
    # so frustum culling works correctly for high-res inputs (e.g. 8K).
    _MVS_MAX = 1920
    if width > 0 and height > 0:
        orig_max = max(width, height)
        if orig_max > _MVS_MAX:
            scale = _MVS_MAX / orig_max
            fx *= scale; fy *= scale
            cx *= scale; cy *= scale
            width = int(round(width * scale))
            height = int(round(height * scale))

    # cam_from_world is the 3×4 [R | t] matrix (world → camera)
    mat = cam_entry.get("cam_from_world", {}).get("matrix_3x4")
    if mat is None:
        return None

    Rt = np.array(mat, dtype=np.float64)   # (3, 4)
    R = Rt[:, :3]                           # (3, 3)
    t = Rt[:, 3]                            # (3,)

    # Camera centre in world space: C = -R^T t
    cam_loc = (-R.T @ t).astype(np.float64)

    return {
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "width": width, "height": height,
        "Rt": Rt, "cam_loc": cam_loc,
    }


# ── Public entry point ────────────────────────────────────────────────────────

def _camera_orbit_hull_mask(pts: np.ndarray, images_list: list, cameras_list: list) -> np.ndarray:
    """
    Return a boolean mask of points that fall inside the convex hull of camera
    positions.  Used for object scans to exclude background/floor — the object
    being photographed sits inside the orbit, so only those points matter.

    Falls back to an axis-aligned bounding box with a 10 % margin if the hull
    is degenerate (e.g. all cameras co-planar or too few positions).
    """
    from scipy.spatial import ConvexHull, Delaunay  # type: ignore

    cam_positions = []
    for img in images_list:
        cam = _parse_camera(img, cameras_list)
        if cam is not None:
            cam_positions.append(cam["cam_loc"])

    if len(cam_positions) < 4:
        logger.warning("[coverage] Too few camera positions (%d) for hull — using bbox fallback", len(cam_positions))
        cam_arr = np.array(cam_positions) if cam_positions else pts
        mn, mx = cam_arr.min(axis=0), cam_arr.max(axis=0)
        margin = (mx - mn) * 0.10
        return np.all((pts >= mn - margin) & (pts <= mx + margin), axis=1)

    cam_arr = np.array(cam_positions, dtype=np.float64)  # (K, 3)

    # Check 3-D spread: if all points are nearly co-planar the hull volume is
    # ~0 and Delaunay.find_simplex() will miss all points.
    extents = cam_arr.max(axis=0) - cam_arr.min(axis=0)
    min_extent = float(extents.min())
    if min_extent < 1e-3 * float(extents.max()):
        logger.warning("[coverage] Camera orbit is near-planar (min extent %.4f) — using bbox fallback", min_extent)
        mn, mx = cam_arr.min(axis=0), cam_arr.max(axis=0)
        margin = (mx - mn) * 0.10
        return np.all((pts >= mn - margin) & (pts <= mx + margin), axis=1)

    try:
        hull = ConvexHull(cam_arr)
        tri = Delaunay(cam_arr[hull.vertices])
        mask = tri.find_simplex(pts) >= 0
        logger.info("[coverage] Object hull: %d/%d pts inside camera orbit hull (%d hull vertices)",
                    int(mask.sum()), len(pts), len(hull.vertices))
        return mask
    except Exception as e:
        logger.warning("[coverage] ConvexHull failed (%s) — using bbox fallback", e)
        mn, mx = cam_arr.min(axis=0), cam_arr.max(axis=0)
        margin = (mx - mn) * 0.10
        return np.all((pts >= mn - margin) & (pts <= mx + margin), axis=1)


def _outdoor_footprint_mask(
    pts: np.ndarray,
    images_list: list,
    cameras_list: list,
    gravity_up_world: list | None,
) -> np.ndarray:
    """
    Return a boolean mask for outdoor scene boundary filtering.

    Two independent filters are AND-ed together:

    1. **Horizontal footprint** — project camera positions onto the horizontal
       plane (orthogonal to gravity), compute their 2D convex hull, and keep
       only cloud points whose horizontal projection falls inside that hull.
       This is the "surveyed area" footprint: if you didn't walk near it,
       it's background.

    2. **Height band** — keep only points between a terrain floor estimate
       (5th percentile of camera heights minus a 0.5 m margin for grass /
       ground-level objects below camera) and a ceiling estimate (highest
       camera position plus a 5 m headroom for trees, facades, etc.).
       The floor estimate degrades gracefully on non-flat terrain because it
       is derived from where the cameras actually went, not from RANSAC on
       the cloud.

    Falls back to a bbox-only filter if fewer than 3 cameras are registered.
    """
    from scipy.spatial import ConvexHull, Delaunay  # type: ignore

    cam_positions = []
    for img in images_list:
        cam = _parse_camera(img, cameras_list)
        if cam is not None:
            cam_positions.append(cam["cam_loc"])

    if len(cam_positions) < 3:
        logger.warning("[coverage] Too few camera positions (%d) for outdoor footprint", len(cam_positions))
        cam_arr = np.array(cam_positions) if cam_positions else pts
        mn, mx = cam_arr.min(axis=0), cam_arr.max(axis=0)
        margin = (mx - mn) * 0.15
        return np.all((pts >= mn - margin) & (pts <= mx + margin), axis=1)

    cam_arr = np.array(cam_positions, dtype=np.float64)  # (K, 3)

    # Determine up axis from gravity vector, falling back to the world axis
    # with the smallest camera-position variance (most likely vertical).
    if gravity_up_world is not None:
        up = np.array(gravity_up_world, dtype=float)
        up_norm = np.linalg.norm(up)
        up = up / up_norm if up_norm > 1e-9 else np.array([0.0, 1.0, 0.0])
    else:
        variances = cam_arr.var(axis=0)
        up_axis_idx = int(np.argmin(variances))
        up = np.eye(3)[up_axis_idx]

    # Build an orthonormal horizontal basis (h0, h1) perpendicular to up.
    # Pick an arbitrary vector not parallel to up for the cross product.
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(up, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    h0 = np.cross(up, ref); h0 /= np.linalg.norm(h0)
    h1 = np.cross(up, h0); h1 /= np.linalg.norm(h1)

    # Project cameras + cloud points onto the horizontal plane.
    cam_h = cam_arr @ np.stack([h0, h1], axis=1)   # (K, 2)
    pts_h = pts     @ np.stack([h0, h1], axis=1)   # (N, 2)

    # ── 1. Horizontal footprint ───────────────────────────────────────────────
    try:
        hull2d = ConvexHull(cam_h)
        tri2d  = Delaunay(cam_h[hull2d.vertices])
        footprint_mask = tri2d.find_simplex(pts_h) >= 0
    except Exception as e:
        logger.warning("[coverage] 2D footprint hull failed (%s) — using bbox fallback", e)
        mn2, mx2 = cam_h.min(axis=0), cam_h.max(axis=0)
        margin2  = (mx2 - mn2) * 0.10
        footprint_mask = np.all((pts_h >= mn2 - margin2) & (pts_h <= mx2 + margin2), axis=1)

    # ── 2. Height band ────────────────────────────────────────────────────────
    # Camera heights along the up axis.
    cam_heights = cam_arr @ up        # (K,)
    pts_heights = pts     @ up        # (N,)

    terrain_z  = float(np.percentile(cam_heights, 5))  - 0.5   # 0.5 m below lowest camera
    ceiling_z  = float(cam_heights.max())              + 5.0   # 5 m above highest camera

    height_mask = (pts_heights >= terrain_z) & (pts_heights <= ceiling_z)

    combined = footprint_mask & height_mask
    logger.info(
        "[coverage] Outdoor clip: footprint %d/%d pts, height-band %d/%d pts, combined %d/%d pts "
        "(terrain_z=%.2f, ceiling_z=%.2f)",
        int(footprint_mask.sum()), len(pts),
        int(height_mask.sum()),    len(pts),
        int(combined.sum()),       len(pts),
        terrain_z, ceiling_z,
    )
    return combined


async def run_coverage_analysis(
    project_id: str,
    scaled_cloud_key: str,
    camera_poses_key: str,
    tmp: Path,
    progress_cb: Callable[[float, str], None],
    source_object: str | None = None,
    gravity_up_world: list | None = None,
    scene_type: str | None = None,
) -> dict:
    import open3d as o3d

    progress_cb(0.0, "Downloading inputs…")
    storage = get_storage()

    # ── 1. Download inputs ────────────────────────────────────────────────────
    cloud_local = tmp / "cloud.ply"
    await storage.download(scaled_cloud_key, cloud_local)
    progress_cb(0.05, "Downloaded point cloud")

    # camera_poses_key may be empty when called from Phase 2 (apply_known_scale uses .si()
    # so it doesn't carry prev_result). Derive from the standard SfM output path.
    if not camera_poses_key:
        camera_poses_key = f"{project_id}/sfm/cameras.json"
        logger.info("[%s] camera_poses_key was empty; using default: %s", project_id, camera_poses_key)

    cameras_local = tmp / "cameras.json"
    await storage.download(camera_poses_key, cameras_local)
    cams_data = json.loads(cameras_local.read_text())
    images_list = cams_data.get("images", [])
    cameras_list = cams_data.get("cameras", [])
    progress_cb(0.08, f"Loaded {len(images_list)} cameras")

    # ── 2. Load point cloud + voxel downsample immediately ───────────────────
    # Load then immediately voxel-downsample so the full-resolution array is
    # never held in RAM alongside the working arrays. This is more memory-
    # efficient than loading all N points then random-sampling.
    pcd_full = o3d.io.read_point_cloud(str(cloud_local))
    n_pts_full = len(pcd_full.points)
    if n_pts_full == 0:
        raise RuntimeError(f"Point cloud at {scaled_cloud_key} contains zero points.")

    if n_pts_full > _SUBSAMPLE_THRESHOLD:
        # Points sit on 2D surfaces, so density ∝ 1/voxel_size² (not cubed).
        # voxel_size = bbox_diag / sqrt(target) gives ≈ target points.
        bbox_diag = np.linalg.norm(
            np.asarray(pcd_full.get_max_bound()) - np.asarray(pcd_full.get_min_bound())
        )
        voxel_size = max(bbox_diag / (_SUBSAMPLE_TARGET ** 0.5), 0.001)
        pcd = pcd_full.voxel_down_sample(voxel_size)
        del pcd_full  # release full-res cloud before further processing
        import gc; gc.collect()
        progress_cb(0.10, f"Loaded {n_pts_full:,} pts → voxel-downsampled to {len(pcd.points):,}")
    else:
        pcd = pcd_full
        progress_cb(0.10, f"Loaded point cloud: {n_pts_full:,} points")

    n_pts = len(pcd.points)

    # ── 3. Estimate normals ───────────────────────────────────────────────────
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    progress_cb(0.15, "Estimated normals")

    pts = np.asarray(pcd.points)      # (N, 3)
    normals = np.asarray(pcd.normals) # (N, 3)

    # ── 4. Scene-type boundary clipping ──────────────────────────────────────
    # Object: keep only points inside the 3D convex hull of camera positions
    # (the orbited object sits inside the camera shell).
    # Outdoor: keep only points inside the 2D horizontal footprint of the
    # camera walk AND within a sensible height band above the terrain.
    if scene_type == "object":
        hull_mask = _camera_orbit_hull_mask(pts, images_list, cameras_list)
        if hull_mask.sum() == 0:
            logger.warning("[%s] Hull clipping removed all points — skipping clip", project_id)
        else:
            keep_idx = np.where(hull_mask)[0]
            pcd = pcd.select_by_index(keep_idx.tolist())
            pts = np.asarray(pcd.points)
            normals = np.asarray(pcd.normals)
            n_pts = len(pts)
            progress_cb(0.16, f"Object hull clip: {n_pts:,} pts inside camera orbit")

    elif scene_type == "outdoor":
        fp_mask = _outdoor_footprint_mask(pts, images_list, cameras_list, gravity_up_world)
        if fp_mask.sum() == 0:
            logger.warning("[%s] Outdoor footprint clip removed all points — skipping clip", project_id)
        else:
            keep_idx = np.where(fp_mask)[0]
            pcd = pcd.select_by_index(keep_idx.tolist())
            pts = np.asarray(pcd.points)
            normals = np.asarray(pcd.normals)
            n_pts = len(pts)
            progress_cb(0.16, f"Outdoor footprint clip: {n_pts:,} pts inside survey area")

    # Orient normals toward the viewpoint centroid.
    # For room scans the world origin is a reasonable proxy.  For object scans
    # the object is inside the orbit so we use the mean camera position instead
    # — orienting toward the world origin would flip normals on half the object.
    _all_cam_locs = [
        c["cam_loc"] for img in images_list
        if (c := _parse_camera(img, cameras_list)) is not None
    ]
    if scene_type == "object" and _all_cam_locs:
        _orient_anchor = np.mean(_all_cam_locs, axis=0)
    else:
        _orient_anchor = np.array([0.0, 0.0, 0.0])
    pcd.orient_normals_towards_camera_location(_orient_anchor)
    normals = np.asarray(pcd.normals)  # refresh after orientation

    # ── 5. Scene scale for HPR radius ────────────────────────────────────────
    scene_scale = float(np.linalg.norm(
        np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())
    ))
    hpr_radius = scene_scale * 100.0
    progress_cb(0.18, f"Scene scale={scene_scale:.4f}, HPR radius={hpr_radius:.2f}")

    # ── 6. Per-point visibility scoring ──────────────────────────────────────
    view_counts = np.zeros(n_pts, dtype=np.float64)
    angle_cos_sum = np.zeros(n_pts, dtype=np.float64)

    # Sample cameras evenly — HPR is O(N log N) per camera.
    # Use ~35% of registered cameras (min 25, max 150) so score scales correctly
    # with video length rather than collapsing to 7% on 350-camera 8K runs.
    max_hpr = _hpr_camera_count(len(images_list))
    if len(images_list) > max_hpr:
        step = len(images_list) / max_hpr
        images_list = [images_list[int(round(step * i))] for i in range(max_hpr)]
    n_cameras = len(images_list)
    for cam_idx, img_entry in enumerate(images_list):
        cam = _parse_camera(img_entry, cameras_list)
        if cam is None:
            continue

        cam_loc = cam["cam_loc"]
        Rt = cam["Rt"]
        fx, fy = cam["fx"], cam["fy"]
        cx, cy = cam["cx"], cam["cy"]
        width, height = cam["width"], cam["height"]

        # Project all vis-pass points into this camera
        # Xc = R * X + t  (homogeneous)
        pts_h = np.hstack([pts, np.ones((n_pts, 1))])  # (N, 4)
        Xc = (Rt @ pts_h.T).T                               # (N, 3)
        depth = Xc[:, 2]

        # In-frustum mask (before occlusion)
        u = (fx * Xc[:, 0] / np.maximum(depth, 1e-9) + cx)
        v = (fy * Xc[:, 1] / np.maximum(depth, 1e-9) + cy)
        frustum_mask = (
            (depth > 0) &
            (u >= 0) & (u < width) &
            (v >= 0) & (v < height)
        )

        frustum_idx = np.where(frustum_mask)[0]
        if len(frustum_idx) == 0:
            continue

        # Occlusion culling via HPR on the frustum-visible subset
        pcd_frustum = pcd.select_by_index(frustum_idx.tolist())
        try:
            _, hpr_pt_map = pcd_frustum.hidden_point_removal(cam_loc, hpr_radius)
            # hpr_pt_map: indices into pcd_frustum that are visible
            visible_local = np.array(hpr_pt_map, dtype=np.int64)
        except Exception as e:
            logger.warning(f"[coverage] HPR failed for camera {cam_idx}: {e}; using frustum only")
            visible_local = np.arange(len(frustum_idx))

        # Map back to global vis indices
        visible_global = frustum_idx[visible_local]

        # Accumulate view count
        view_counts[visible_global] += 1

        # Angle between normal and camera→point ray
        # ray direction: point - camera_location (unnormalized)
        rays = pts[visible_global] - cam_loc                  # (V, 3)
        ray_norms = np.linalg.norm(rays, axis=1, keepdims=True)
        safe_ray_norms = np.maximum(ray_norms, 1e-9)
        rays_unit = rays / safe_ray_norms                         # (V, 3)

        nrms = normals[visible_global]                        # (V, 3)
        nrm_norms = np.linalg.norm(nrms, axis=1, keepdims=True)
        safe_nrm_norms = np.maximum(nrm_norms, 1e-9)
        nrms_unit = nrms / safe_nrm_norms                         # (V, 3)

        # cos(angle) between -ray and normal (1 = camera looking straight at surface)
        cos_angles = np.clip(
            np.einsum("ij,ij->i", -rays_unit, nrms_unit), 0.0, 1.0
        )
        angle_cos_sum[visible_global] += cos_angles

        if (cam_idx + 1) % max(1, n_cameras // 20) == 0 or cam_idx + 1 == n_cameras:
            progress_cb(
                0.18 + 0.60 * (cam_idx + 1) / n_cameras,
                f"Processed camera {cam_idx+1}/{n_cameras}",
            )

    progress_cb(0.78, "Computing per-point scores…")

    # ── 7. Compute scores ─────────────────────────────────────────────────────
    p95 = np.percentile(view_counts, 95)
    if p95 < 1e-9:
        p95 = max(float(view_counts.max()), 1.0)

    view_count_score = np.clip(view_counts / p95, 0.0, 1.0)
    # Suppress divide-by-zero warning: np.where evaluates both branches before masking.
    with np.errstate(divide='ignore', invalid='ignore'):
        angle_score = np.where(view_counts > 0, angle_cos_sum / view_counts, 0.0)
    score_vis = 0.6 * view_count_score + 0.4 * angle_score

    score = score_vis

    coverage_score = float(np.mean(score))
    progress_cb(0.80, f"Coverage score: {coverage_score:.3f}")

    # ── 8. Colorise the full cloud ────────────────────────────────────────────
    pcd.colors = o3d.utility.Vector3dVector(_score_to_rgb(score))
    progress_cb(0.82, "Colorised point cloud")

    # ── 9. DBSCAN shot suggestions ────────────────────────────────────────────
    suggestions: list[dict] = []
    low_idx = np.where(score < 0.4)[0]

    if len(low_idx) >= 10:
        from sklearn.cluster import DBSCAN

        scene_diag = float(np.linalg.norm(
            np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())
        ))

        # Object scans: use a larger eps (5% of diag) — we want a few large,
        # meaningful angle-based clusters, not many tiny overlapping patches.
        # Rooms/outdoor: adaptive eps, shrink until we get ≥3 clusters.
        if scene_type == "object":
            eps = scene_diag * 0.05
        else:
            eps = scene_diag * 0.02
            for _eps_attempt in range(3):
                _db_test = DBSCAN(eps=eps, min_samples=5).fit(pts[low_idx])
                _n_test = len(set(_db_test.labels_)) - (1 if -1 in _db_test.labels_ else 0)
                if _n_test >= 3 or eps < scene_diag * 0.005:
                    break
                eps *= 0.6

        db = DBSCAN(eps=eps, min_samples=5).fit(pts[low_idx])
        labels = db.labels_
        logger.info("[coverage] DBSCAN eps=%.4f → %d clusters",
                    eps, len(set(labels)) - (1 if -1 in labels else 0))

        # Collect clusters (exclude noise label=-1), sorted by size (largest first)
        unique_labels = [lb for lb in set(labels) if lb != -1]
        cluster_sizes = [(lb, int(np.sum(labels == lb))) for lb in unique_labels]
        cluster_sizes.sort(key=lambda x: -x[1])

        # Keep all clusters with ≥0.5% of total low-coverage points (no top-5 cap)
        min_cluster_pts = max(5, int(len(low_idx) * 0.005))
        cluster_sizes = [(lb, sz) for lb, sz in cluster_sizes if sz >= min_cluster_pts]

        raw_suggestions = []

        # If spatial clustering produced only 1 cluster (the whole room is one
        # connected blob), fall back to surface-normal bucketing instead.
        # Group low-coverage points by their dominant normal direction to get
        # per-surface suggestions: floor, ceiling, each wall quadrant.
        if len(cluster_sizes) <= 1 and len(low_idx) > 100:
            low_normals = normals[low_idx]
            low_pts_arr = pts[low_idx]
            # Project normals onto the 6 principal directions
            principal = np.array([
                [1, 0, 0], [-1, 0, 0],   # +X / -X walls
                [0, 1, 0], [0, -1, 0],   # +Y / -Y
                [0, 0, 1], [0, 0, -1],   # +Z / -Z walls
            ], dtype=float)
            dots = low_normals @ principal.T  # (N, 6)
            bucket = dots.argmax(axis=1)      # dominant direction per point

            # Only label as floor/ceiling if gravity_up_world is known.
            # Without gravity alignment, +Y/-Y could be anything — mislabelling
            # confuses users (e.g. white ceiling gets called "floor").
            if gravity_up_world is not None:
                gup = np.array(gravity_up_world, dtype=float)
                gup /= np.linalg.norm(gup) + 1e-9
                # Find which principal direction is most aligned with gravity-up
                gravity_dots = [float(np.dot(gup, p)) for p in principal]
                up_idx   = int(np.argmax(gravity_dots))    # most aligned with up
                down_idx = int(np.argmin(gravity_dots))    # most opposed = down
                surface_names_base = ["+X wall", "-X wall", "+Y surface", "-Y surface", "+Z wall", "-Z wall"]
                # Outdoor terrain is never a flat "floor"; overhead is open sky
                # not a "ceiling". Use terrain/upper labels for outdoor.
                if scene_type == "outdoor":
                    surface_names_base[up_idx]   = "upper/overhead surface"
                    surface_names_base[down_idx] = "ground/terrain surface"
                else:
                    surface_names_base[up_idx]   = "ceiling"
                    surface_names_base[down_idx] = "floor"
                surface_names = surface_names_base
            else:
                # No gravity — use axis labels for horizontal surfaces, avoid floor/ceiling
                surface_names = ["+X wall", "-X wall", "horizontal surface", "horizontal surface", "+Z wall", "-Z wall"]
            for b_idx, s_name in enumerate(surface_names):
                mask = bucket == b_idx
                if mask.sum() < min_cluster_pts:
                    continue
                b_pts = low_pts_arr[mask]
                b_normals = low_normals[mask]
                centroid = b_pts.mean(axis=0)
                avg_normal = b_normals.mean(axis=0)
                n_norm = np.linalg.norm(avg_normal)
                avg_normal = avg_normal / n_norm if n_norm > 1e-9 else principal[b_idx]
                cam_pos = centroid + _SUGGESTION_STANDOFF_M * avg_normal
                raw_suggestions.append({
                    "centroid": centroid,
                    "cam_pos": cam_pos,
                    "direction": avg_normal,
                    "cluster_size": int(mask.sum()),
                    "pct_low": mask.sum() / max(1, len(low_idx)),
                    "surface": s_name,
                    "cluster_pts_idx": low_idx[mask],
                })
            logger.info("[coverage] Normal-bucket fallback: %d surface suggestions", len(raw_suggestions))
        else:
            for lb, cluster_size in cluster_sizes:
                cluster_mask = labels == lb
                cluster_pts_idx = low_idx[cluster_mask]
                cluster_pts = pts[cluster_pts_idx]
                centroid = cluster_pts.mean(axis=0)

                cluster_normals = normals[cluster_pts_idx]
                avg_normal = cluster_normals.mean(axis=0)
                n_norm = np.linalg.norm(avg_normal)
                avg_normal = avg_normal / n_norm if n_norm > 1e-9 else np.array([0.0, 0.0, 1.0])

                cam_pos = centroid + _SUGGESTION_STANDOFF_M * avg_normal
                raw_suggestions.append({
                    "centroid": centroid,
                    "cam_pos": cam_pos,
                    "direction": avg_normal,
                    "cluster_size": cluster_size,
                    "pct_low": cluster_size / max(1, len(low_idx)),
                    "cluster_pts_idx": cluster_pts_idx,
                })

        # For object scans, derive the object centroid now (after hull clip)
        # so each suggestion can be labelled by its position on the object.
        _object_center = pts.mean(axis=0) if scene_type == "object" else None
        _gup = (np.array(gravity_up_world, dtype=float) if gravity_up_world else None)

        # Angular deduplication for object mode: drop suggestions whose viewing
        # direction from the object center is within MIN_ANGLE_DEG of a larger one.
        if scene_type == "object" and _object_center is not None and len(raw_suggestions) > 1:
            MIN_ANGLE_DEG = 35.0
            accepted: list[dict] = []
            for s in raw_suggestions:  # already sorted largest-first
                d = s["centroid"] - _object_center
                norm = np.linalg.norm(d)
                if norm < 1e-9:
                    accepted.append(s)
                    continue
                d_unit = d / norm
                too_close = False
                for kept in accepted:
                    k = kept["centroid"] - _object_center
                    k_norm = np.linalg.norm(k)
                    if k_norm < 1e-9:
                        continue
                    k_unit = k / k_norm
                    cos_a = float(np.clip(np.dot(d_unit, k_unit), -1.0, 1.0))
                    if np.degrees(np.arccos(cos_a)) < MIN_ANGLE_DEG:
                        too_close = True
                        break
                if not too_close:
                    accepted.append(s)
            dropped = len(raw_suggestions) - len(accepted)
            if dropped:
                logger.info("[coverage] Angular dedup: dropped %d duplicate-angle suggestions", dropped)
            raw_suggestions = accepted

        # Cap suggestions (objects: 4, rooms: 6, outdoor: 5)
        _max_suggestions = {"object": 4, "outdoor": 5}.get(scene_type, 6)
        raw_suggestions = raw_suggestions[:_max_suggestions]

        # Order as a walking path via nearest-neighbour greedy TSP so one
        # continuous video pass covers all clusters without backtracking.
        if len(raw_suggestions) > 1:
            remaining_idx = list(range(1, len(raw_suggestions)))
            ordered = [raw_suggestions[0]]
            while remaining_idx:
                last = ordered[-1]["cam_pos"]
                best_i = min(remaining_idx, key=lambda i: np.linalg.norm(raw_suggestions[i]["cam_pos"] - last))
                ordered.append(raw_suggestions[best_i])
                remaining_idx.remove(best_i)
            raw_suggestions = ordered

        for idx, s in enumerate(raw_suggestions):
            cam_px, cam_py, cam_pz = (float(v) for v in s["cam_pos"])
            nx_s, ny_s, nz_s = (float(v) for v in s["direction"])
            cluster_size = s["cluster_size"]
            pct = s["pct_low"] * 100

            # Distinguish "never photographed" from "photographed but poor quality"
            # by checking mean view_count in this cluster.
            cluster_pts_idx = s.get("cluster_pts_idx")
            if cluster_pts_idx is not None and len(cluster_pts_idx) > 0:
                mean_views = float(view_counts[cluster_pts_idx].mean())
            else:
                mean_views = 0.0
            quality_issue = mean_views >= 1.0  # seen but poorly covered

            if scene_type == "object" and _object_center is not None:
                surface_label = _object_surface_label(s["centroid"], _object_center, _gup)
                if quality_issue:
                    msg = (
                        f"{surface_label.capitalize()} of object ({cluster_size:,} pts, "
                        f"{pct:.0f}% of under-covered) — this side was filmed but coverage is poor. "
                        f"Shoot closer, slower, or with better lighting from this angle."
                    )
                else:
                    msg = (
                        f"{surface_label.capitalize()} of object ({cluster_size:,} pts, "
                        f"{pct:.0f}% of under-covered) — little to no footage from this angle. "
                        f"Orbit to photograph this side."
                    )
            else:
                surface = s.get("surface", "")
                surface_label = f" ({surface})" if surface else ""
                if quality_issue:
                    msg = (
                        f"Area {idx + 1}{surface_label}: {cluster_size:,} points ({pct:.0f}% of under-covered) — "
                        f"filmed but coverage is low. Shoot more slowly or from closer."
                    )
                else:
                    msg = (
                        f"Area {idx + 1}{surface_label}: {cluster_size:,} points ({pct:.0f}% of under-covered) — "
                        f"aim at this spot from the suggested position."
                    )

            suggestions.append({
                "position": [cam_px, cam_py, cam_pz],
                "direction": [nx_s, ny_s, nz_s],
                "message": msg,
                "cluster_size": cluster_size,
                "area_index": idx + 1,
            })

    n_low = int(len(low_idx))
    progress_cb(0.90, f"Generated {len(suggestions)} shot suggestion(s); {n_low:,} low-coverage points")

    # Anchor reminder — tell the user to include the scale reference object
    anchor_reminder = None
    if source_object and source_object.lower() not in ("manual_1m", "manual"):
        objects = [o.strip() for o in source_object.split(",") if o.strip()]
        if len(objects) == 1:
            anchor_reminder = (
                f"Make sure to include the {objects[0]} in your re-shoot — "
                f"it will be used to confirm the metric scale."
            )
        else:
            # Show top 3, summarise the rest
            top = objects[:3]
            rest = len(objects) - 3
            listed = ", ".join(top)
            suffix = f" (and {rest} other recognised objects)" if rest > 0 else ""
            anchor_reminder = (
                f"Include at least one of these in your re-shoot: {listed}{suffix}. "
                f"Any visible anchor object will confirm the metric scale."
            )

    # ── 10. Upload coloured cloud (timestamped + latest alias) ───────────────
    import time as _time
    run_ts = int(_time.time())
    colored_local = tmp / "cloud_colored.ply"
    o3d.io.write_point_cloud(str(colored_local), pcd)

    # Timestamped copy for history
    ts_key = f"{project_id}/coverage/run_{run_ts}_cloud.ply"
    await storage.upload(colored_local, ts_key)
    # Latest alias (overwrites previous)
    coverage_key = f"{project_id}/coverage/cloud_colored.ply"
    await storage.upload(colored_local, coverage_key)
    progress_cb(1.0, f"Coverage score: {coverage_score:.1%}  Uploaded {coverage_key}")

    # Point density: pts per cubic metre (if scale factor available, else SfM units)
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent())
    volume = float(np.prod(extent)) if float(np.prod(extent)) > 1e-6 else None
    point_density = round(n_pts / volume, 1) if volume else None

    return {
        "coverage_cloud_key": coverage_key,
        "coverage_ts_key": ts_key,
        "coverage_run_ts": run_ts,
        "coverage_score": coverage_score,
        "anchor_reminder": anchor_reminder,
        "suggestions": suggestions,
        "n_points": int(n_pts),
        "n_low_coverage": n_low,
        "point_density": point_density,   # pts per unit³ (m³ if scale applied)
        "scene_volume": round(volume, 2) if volume else None,
    }
