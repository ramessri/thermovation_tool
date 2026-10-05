"""
Stage 7: Export final deliverables (.ply, .obj, .las)

Produces:
  PLY        — coverage-heatmap-coloured cloud.
  OBJ        — Poisson surface reconstruction (depth=9).
  LAS        — laspy v1.4 point-format-2 with 16-bit RGB colour.
  confidence_cloud.ply — same points, colours = per-point confidence heatmap.
  semantic_cloud.ply   — same points, colours = per-object semantic labels.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from backend.core.storage import get_storage
from backend.core.config import settings

logger = logging.getLogger(__name__)

_CONFIDENCE_BASE = 0.70   # MVS points have this baseline confidence


def _confidence_color(score: float) -> tuple[float, float, float]:
    """Map [0,1] confidence to RGB: red=low, yellow=mid, green=high."""
    r = max(0.0, min(1.0, 2.0 * (1.0 - score)))
    g = max(0.0, min(1.0, 2.0 * score))
    b = 0.0
    return r, g, b


def _build_confidence_colors(
    pts: np.ndarray,
    coverage_colors: Optional[np.ndarray],
    object_labels: list[dict],
) -> np.ndarray:
    """Derive per-point confidence and return RGB color array (N, 3)."""
    n = len(pts)
    conf = np.full(n, _CONFIDENCE_BASE, dtype=np.float32)

    # Coverage signal: green channel of heatmap ≈ coverage level
    if coverage_colors is not None and len(coverage_colors) == n:
        cov_signal = coverage_colors[:, 1].astype(np.float32)   # green channel
        conf = np.clip(conf * 0.5 + cov_signal * 0.5, 0.0, 1.0)

    # Override with semantic confidence for labeled regions
    for region in object_labels:
        try:
            bmin = np.array(region["bbox_3d_min"])
            bmax = np.array(region["bbox_3d_max"])
            span = bmax - bmin
            bmin2 = bmin - span * 0.05
            bmax2 = bmax + span * 0.05
            mask = np.all((pts >= bmin2) & (pts <= bmax2), axis=1)
            if mask.any():
                sem_conf = float(region.get("semantic_confidence", _CONFIDENCE_BASE))
                conf[mask] = np.clip(sem_conf, 0.0, 1.0)
        except Exception:
            pass

    colors = np.array([_confidence_color(c) for c in conf], dtype=np.float64)
    return colors


def _semantic_color_palette(n_objects: int) -> list[tuple[float, float, float]]:
    """Generate n distinct HSV-derived colours for semantic objects."""
    colors = []
    for i in range(n_objects):
        hue = (i * 0.618033988749895) % 1.0   # golden ratio spread
        # Convert HSV (h, 0.8, 0.9) → RGB
        h6 = hue * 6.0
        f = h6 - int(h6)
        p, q, t = 0.9 * 0.2, 0.9 * (1 - 0.8 * f), 0.9 * (1 - 0.8 * (1 - f))
        sector = int(h6) % 6
        if sector == 0:   rgb = (0.9, t, p)
        elif sector == 1: rgb = (q, 0.9, p)
        elif sector == 2: rgb = (p, 0.9, t)
        elif sector == 3: rgb = (p, q, 0.9)
        elif sector == 4: rgb = (t, p, 0.9)
        else:             rgb = (0.9, p, q)
        colors.append(rgb)
    return colors


def _build_semantic_colors(
    pts: np.ndarray,
    object_labels: list[dict],
) -> np.ndarray:
    """Color each labeled region distinctly; unlabeled = medium gray."""
    n = len(pts)
    colors = np.full((n, 3), 0.45, dtype=np.float64)   # gray for unlabeled

    palette = _semantic_color_palette(len(object_labels))
    for idx, region in enumerate(object_labels):
        try:
            bmin = np.array(region["bbox_3d_min"])
            bmax = np.array(region["bbox_3d_max"])
            span = bmax - bmin
            mask = np.all((pts >= bmin - span * 0.05) & (pts <= bmax + span * 0.05), axis=1)
            if mask.any():
                colors[mask] = palette[idx]
        except Exception:
            pass

    return colors


async def run_export(
    project_id: str,
    coverage_cloud_key: str,
    tmp: Path,
    progress_cb: Callable[[float, str], None],
    object_labels: Optional[list[dict]] = None,
    scene_type: str = "indoor_room",
    gravity_up_world: Optional[list[float]] = None,
    ground_truth_dimensions: Optional[dict] = None,
    cloud_is_metric: bool = False,
) -> dict:
    import open3d as o3d

    is_object = scene_type == "object"
    progress_cb(0.0, "Downloading coverage cloud…")
    storage = get_storage()

    # ── 1. Download the coverage-coloured cloud ───────────────────────────────
    cloud_local = tmp / "cloud.ply"
    await storage.download(coverage_cloud_key, cloud_local)

    pcd = o3d.io.read_point_cloud(str(cloud_local))
    n_pts = len(pcd.points)
    logger.info(f"[exporter] Loaded cloud: {n_pts:,} points")
    progress_cb(0.05, f"Loaded point cloud: {n_pts:,} points")

    exports = []

    # ── 1b. Real-world L x B x H ───────────────────────────────────────────────
    # Only when the cloud is in metres (scale derived from markers, or LiDAR);
    # an unscaled SfM cloud would report arbitrary units as metres. "Up" comes
    # from the ArUco floor marker or LiDAR ingest when available, otherwise
    # from the cloud's own floor plane. Object scans have no floor — skip.
    dimensions = None
    if cloud_is_metric and not is_object:
        try:
            from backend.workers.pipeline.dimensions import (
                compare_to_ground_truth, compute_dimensions, detect_floor_gravity,
            )
            if not gravity_up_world:
                progress_cb(0.055, "Detecting floor plane for dimensions…")
                gravity_up_world, _ = detect_floor_gravity(pcd)
            if gravity_up_world:
                dimensions = compute_dimensions(pcd, gravity_up_world, scale=1.0)
                progress_cb(0.06, f"Dimensions: {dimensions['length_m']}m x "
                                  f"{dimensions['breadth_m']}m x {dimensions['height_m']}m")
                if ground_truth_dimensions:
                    gt_check = compare_to_ground_truth(dimensions, ground_truth_dimensions)
                    if gt_check:
                        dimensions["ground_truth_check"] = gt_check
                        mae = gt_check.get("mean_abs_error_pct")
                        if mae is not None:
                            progress_cb(0.065, f"Ground-truth check: {mae:.1f}% mean abs error")
            else:
                logger.warning("[exporter] no floor plane found — dimensions unavailable")
        except Exception as e:
            logger.warning("[exporter] dimension computation failed: %s", e)

    # ── 2. PLY export ─────────────────────────────────────────────────────────
    # For object scans: export the scaled MVS cloud (real RGB, clean geometry).
    # For rooms/outdoor: export the coverage-coloured cloud (heatmap is useful).
    ply_key = f"{project_id}/exports/output.ply"
    if is_object:
        mvs_ply_key = f"{project_id}/mvs/dense.ply"
        mvs_ply_local = tmp / "mvs_export.ply"
        try:
            await storage.download(mvs_ply_key, mvs_ply_local)
            mvs_pcd_ply = o3d.io.read_point_cloud(str(mvs_ply_local))

            # Clip to camera orbit sphere (in SfM units, before scaling)
            try:
                import json as _json2
                from backend.workers.pipeline.coverage import _parse_camera
                _cam_local = tmp / "cameras_ply.json"
                await storage.download(f"{project_id}/sfm/cameras.json", _cam_local)
                _cams = _json2.loads(_cam_local.read_text())
                _cam_pos = [_parse_camera(img, _cams.get("cameras", []))["cam_loc"]
                            for img in _cams.get("images", [])
                            if _parse_camera(img, _cams.get("cameras", [])) is not None]
                if len(_cam_pos) >= 3:
                    _ca = np.array(_cam_pos, dtype=np.float64)
                    _cen = _ca.mean(axis=0)
                    _r = float(np.max(np.linalg.norm(_ca - _cen, axis=1)))
                    _raw = np.asarray(mvs_pcd_ply.points)
                    _mask = np.linalg.norm(_raw - _cen, axis=1) <= _r * 0.9
                    mvs_pcd_ply = mvs_pcd_ply.select_by_index(np.where(_mask)[0].tolist())
                    logger.info("[exporter] PLY orbit clip: %d → %d pts (r=%.3f)", len(_raw), len(mvs_pcd_ply.points), _r)
            except Exception as _e:
                logger.warning("[exporter] PLY orbit clip failed (%s)", _e)

            # Apply metric scale
            try:
                import psycopg2 as _pg2
                _c2 = _pg2.connect(settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgres://"))
                _cur2 = _c2.cursor()
                _cur2.execute("SELECT confirmed_scale_factor FROM projects WHERE id=%s", (project_id,))
                _row2 = _cur2.fetchone()
                _c2.close()
                ply_scale = float(_row2[0]) if _row2 and _row2[0] else 1.0
            except Exception:
                ply_scale = 1.0
            if ply_scale != 1.0:
                mvs_pts = np.asarray(mvs_pcd_ply.points) * ply_scale
                mvs_pcd_ply.points = o3d.utility.Vector3dVector(mvs_pts)
            o3d.io.write_point_cloud(str(mvs_ply_local), mvs_pcd_ply)
            await storage.upload(mvs_ply_local, ply_key)
            logger.info("[exporter] PLY: %d pts, scale=%.5f", len(mvs_pcd_ply.points), ply_scale)
        except Exception as e:
            logger.warning("[exporter] MVS PLY unavailable (%s), using coverage cloud", e)
            await storage.upload(cloud_local, ply_key)
    else:
        await storage.upload(cloud_local, ply_key)
    exports.append({
        "label": "Colored Point Cloud (.ply)",
        "key": ply_key,
        "mime_type": "application/octet-stream",
    })
    progress_cb(0.20, "PLY export complete")

    # ── 2b. Confidence + semantic colour PLYs ────────────────────────────────
    if object_labels:
        pts_arr = np.asarray(pcd.points, dtype=np.float64)
        coverage_colors = np.asarray(pcd.colors) if pcd.has_colors() else None

        try:
            conf_colors = _build_confidence_colors(pts_arr, coverage_colors, object_labels)
            conf_pcd = o3d.geometry.PointCloud()
            conf_pcd.points = pcd.points
            conf_pcd.colors = o3d.utility.Vector3dVector(conf_colors)
            conf_local = tmp / "confidence_cloud.ply"
            o3d.io.write_point_cloud(str(conf_local), conf_pcd)
            conf_key = f"{project_id}/exports/confidence_cloud.ply"
            await storage.upload(conf_local, conf_key)
            exports.append({
                "label": "Confidence Heatmap (.ply)",
                "key": conf_key,
                "mime_type": "application/octet-stream",
            })
            logger.info("[exporter] confidence_cloud.ply written (%d pts)", n_pts)
        except Exception as e:
            logger.warning("[exporter] confidence PLY failed: %s", e)

        try:
            sem_colors = _build_semantic_colors(pts_arr, object_labels)
            sem_pcd = o3d.geometry.PointCloud()
            sem_pcd.points = pcd.points
            sem_pcd.colors = o3d.utility.Vector3dVector(sem_colors)
            sem_local = tmp / "semantic_cloud.ply"
            o3d.io.write_point_cloud(str(sem_local), sem_pcd)
            sem_key = f"{project_id}/exports/semantic_cloud.ply"
            await storage.upload(sem_local, sem_key)
            exports.append({
                "label": "Semantic Labels (.ply)",
                "key": sem_key,
                "mime_type": "application/octet-stream",
            })
            logger.info("[exporter] semantic_cloud.ply written (%d pts, %d objects)",
                        n_pts, len(object_labels))
        except Exception as e:
            logger.warning("[exporter] semantic PLY failed: %s", e)

    progress_cb(0.22, "Confidence + semantic PLYs complete")

    # ── 3. OBJ — scene-aware surface reconstruction ───────────────────────────
    obj_path = tmp / "output.obj"
    obj_key = f"{project_id}/exports/output.obj"

    try:
        if n_pts < 100:
            raise ValueError(f"Too few points for reconstruction: {n_pts}")

        pts_arr = np.asarray(pcd.points)

        # Estimate normals if not present
        if not pcd.has_normals():
            radius = max(0.02, float(np.percentile(
                np.linalg.norm(pts_arr - pts_arr.mean(axis=0), axis=1), 5
            )) * 0.1)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=max(radius, 0.05), max_nn=50
                )
            )

        # Orient normals: for objects use the cloud centroid (camera orbits the object);
        # for rooms/outdoor use scene origin (camera walks through the space).
        centroid = pts_arr.mean(axis=0) if is_object else np.array([0.0, 0.0, 0.0])
        pcd.orient_normals_towards_camera_location(centroid)

        if is_object:
            # For object scans, the COLMAP MVS dense cloud is much cleaner than
            # the GS-extracted cloud (fewer artifacts, real RGB from photos).
            # Load it directly and apply the metric scale factor.
            progress_cb(0.25, "Loading MVS dense cloud for mesh…")
            mvs_key = f"{project_id}/mvs/dense.ply"
            mvs_local = tmp / "mvs_dense.ply"
            mvs_pcd = None
            try:
                await storage.download(mvs_key, mvs_local)
                mvs_pcd = o3d.io.read_point_cloud(str(mvs_local))

                # Clip to camera orbit sphere BEFORE scaling (cameras.json is in SfM units).
                # We use a sphere (centroid + radius) rather than the strict convex hull,
                # because partial orbits produce thin wedge hulls that exclude the object.
                try:
                    import json as _json
                    from backend.workers.pipeline.coverage import _parse_camera
                    cameras_local = tmp / "cameras_for_clip.json"
                    await storage.download(f"{project_id}/sfm/cameras.json", cameras_local)
                    cams_data = _json.loads(cameras_local.read_text())
                    cam_positions = []
                    for img in cams_data.get("images", []):
                        cam = _parse_camera(img, cams_data.get("cameras", []))
                        if cam is not None:
                            cam_positions.append(cam["cam_loc"])
                    if len(cam_positions) >= 3:
                        cam_arr = np.array(cam_positions, dtype=np.float64)
                        centroid = cam_arr.mean(axis=0)
                        orbit_radius = float(np.max(np.linalg.norm(cam_arr - centroid, axis=1)))
                        # Keep points within 1.1× the orbit radius of the centroid
                        raw_pts = np.asarray(mvs_pcd.points)
                        dist = np.linalg.norm(raw_pts - centroid, axis=1)
                        mask = dist <= orbit_radius * 0.9
                        mvs_pcd = mvs_pcd.select_by_index(np.where(mask)[0].tolist())
                        logger.info("[exporter] Orbit sphere clip: %d → %d pts (r=%.3f)",
                                    len(raw_pts), len(mvs_pcd.points), orbit_radius)
                except Exception as e:
                    logger.warning("[exporter] Orbit sphere clip failed (%s) — using full MVS cloud", e)

                # Apply metric scale from DB
                try:
                    import psycopg2 as _pg
                    _c = _pg.connect(settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgres://"))
                    _cur = _c.cursor()
                    _cur.execute("SELECT confirmed_scale_factor FROM projects WHERE id=%s", (project_id,))
                    _row = _cur.fetchone()
                    _c.close()
                    scale = float(_row[0]) if _row and _row[0] else 1.0
                except Exception:
                    scale = 1.0

                if scale != 1.0:
                    mvs_pts = np.asarray(mvs_pcd.points) * scale
                    mvs_pcd.points = o3d.utility.Vector3dVector(mvs_pts)

                logger.info("[exporter] MVS cloud: %d pts, scale=%.5f", len(mvs_pcd.points), scale)
            except Exception as e:
                logger.warning("[exporter] Could not load MVS cloud (%s), falling back to export cloud", e)
                mvs_pcd = None

            mesh_pcd = mvs_pcd if mvs_pcd and len(mvs_pcd.points) > 100 else pcd
            mesh_pts = np.asarray(mesh_pcd.points)

            # Estimate normals oriented toward the object centroid
            centroid = mesh_pts.mean(axis=0)
            if not mesh_pcd.has_normals():
                mesh_pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50)
                )
            mesh_pcd.orient_normals_towards_camera_location(centroid)

            # BPA radius from KNN on a sample
            sample_idx = np.random.choice(len(mesh_pts), min(1000, len(mesh_pts)), replace=False)
            tree = o3d.geometry.KDTreeFlann(mesh_pcd)
            nn_dists = []
            for i in sample_idx:
                pt = mesh_pts[i]
                _, _, dist2 = tree.search_knn_vector_3d(pt, 2)
                if len(dist2) > 1:
                    nn_dists.append(float(dist2[1]) ** 0.5)
            avg_nn = float(np.median(nn_dists)) if nn_dists else 0.01
            radii = [avg_nn * 1.5, avg_nn * 3, avg_nn * 6]

            progress_cb(0.35, f"Running Ball Pivoting on MVS cloud ({len(mesh_pts):,} pts, r~{avg_nn*1000:.1f}mm)…")
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                mesh_pcd, o3d.utility.DoubleVector(radii)
            )
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()

            # ── Post-processing ───────────────────────────────────────────────
            n_before = len(mesh.triangles)

            # 1. Remove tiny disconnected fragments — keep any component with at
            #    least 1% of the triangles of the largest component.  This drops
            #    floating background dust while preserving wheels, handles, etc.
            triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)
            if len(cluster_n_triangles) > 1:
                max_component = int(cluster_n_triangles.max())
                min_keep = max(10, int(max_component * 0.01))  # keep ≥1% of largest
                small_mask = np.array([cluster_n_triangles[c] < min_keep
                                       for c in triangle_clusters])
                if small_mask.any():
                    mesh.remove_triangles_by_mask(small_mask)
                    mesh.remove_unreferenced_vertices()

                # Object scans only: drop components whose centroid is more than
                # 0.6× the orbit radius from the mesh centroid — removes background
                # slabs that survived the size threshold.  Outdoor/indoor scenes are
                # too large for a radius-based exclusion zone.
                if is_object:
                    mesh_verts = np.asarray(mesh.vertices)
                    object_centroid = mesh_verts.mean(axis=0)
                    tri_clusters2, cluster_sizes2, _ = mesh.cluster_connected_triangles()
                    tri_clusters2 = np.asarray(tri_clusters2)
                    tris2 = np.asarray(mesh.triangles)
                    far_mask = np.zeros(len(tris2), dtype=bool)
                    _orbit_r = locals().get('orbit_radius', 1.0)
                    _mesh_scale = locals().get('scale', 1.0)
                    for cid in np.unique(tri_clusters2):
                        c_tri_idx = np.where(tri_clusters2 == cid)[0]
                        c_vert_idx = np.unique(tris2[c_tri_idx])
                        c_centroid = np.asarray(mesh.vertices)[c_vert_idx].mean(axis=0)
                        if float(np.linalg.norm(c_centroid - object_centroid)) > _orbit_r * _mesh_scale * 0.6:
                            far_mask[c_tri_idx] = True
                    if far_mask.any():
                        mesh.remove_triangles_by_mask(far_mask)
                        mesh.remove_unreferenced_vertices()
                        logger.info("[exporter] Centroid-distance filter: removed %d stray triangles",
                                    int(far_mask.sum()))

                logger.info("[exporter] Component filter: %d → %d triangles (threshold=%d tris)",
                            n_before, len(mesh.triangles), min_keep)

            # 2. Remove long-edge triangles — "tent" triangles that span between
            #    the object surface and distant background points.
            if len(mesh.triangles) > 0:
                verts = np.asarray(mesh.vertices)
                tris  = np.asarray(mesh.triangles)
                edge_a = np.linalg.norm(verts[tris[:, 1]] - verts[tris[:, 0]], axis=1)
                edge_b = np.linalg.norm(verts[tris[:, 2]] - verts[tris[:, 1]], axis=1)
                edge_c = np.linalg.norm(verts[tris[:, 0]] - verts[tris[:, 2]], axis=1)
                max_edge = np.maximum(np.maximum(edge_a, edge_b), edge_c)
                # Threshold: 10× median edge length
                edge_threshold = float(np.median(max_edge)) * 10
                long_mask = max_edge > edge_threshold
                if long_mask.any():
                    mesh.remove_triangles_by_mask(long_mask)
                    mesh.remove_unreferenced_vertices()
                    logger.info("[exporter] Long-edge filter: removed %d triangles (threshold=%.4fm)",
                                int(long_mask.sum()), edge_threshold)

            # 3. Hole filling via trimesh.
            if len(mesh.triangles) > 0:
                try:
                    import trimesh as _tm
                    _verts = np.asarray(mesh.vertices)
                    _faces = np.asarray(mesh.triangles)
                    _tm_mesh = _tm.Trimesh(vertices=_verts, faces=_faces, process=False)
                    _tm.repair.fill_holes(_tm_mesh)
                    n_filled = len(_tm_mesh.faces) - len(_faces)
                    if n_filled > 0:
                        logger.info("[exporter] Hole filling: +%d triangles", n_filled)
                        # Re-transfer vertex colors from the source cloud
                        filled = o3d.geometry.TriangleMesh()
                        filled.vertices = o3d.utility.Vector3dVector(_tm_mesh.vertices)
                        filled.triangles = o3d.utility.Vector3iVector(_tm_mesh.faces)
                        _color_src = mesh_pcd if is_object else pcd
                        if _color_src.has_colors():
                            pcd_tree2 = o3d.geometry.KDTreeFlann(_color_src)
                            new_verts = _tm_mesh.vertices
                            new_colors = np.zeros((len(new_verts), 3), dtype=np.float64)
                            for i, v in enumerate(new_verts):
                                _, idx2, _ = pcd_tree2.search_knn_vector_3d(v, 1)
                                new_colors[i] = np.asarray(_color_src.colors)[idx2[0]]
                            filled.vertex_colors = o3d.utility.Vector3dVector(new_colors)
                        mesh = filled
                except Exception as e:
                    logger.warning("[exporter] Hole filling failed (%s)", e)

            # 4. Taubin smoothing — reduces surface noise without shrinking the mesh.
            if len(mesh.triangles) > 0:
                mesh = mesh.filter_smooth_taubin(number_of_iterations=30)

            mesh.compute_vertex_normals()
            method_label = f"Ball Pivoting (MVS {len(mesh_pts):,} pts)"
        elif scene_type == "outdoor":
            # Outdoor: also use BPA — Poisson fills open space with hallucinated
            # geometry (terrible for swing sets, trees, thin structures).
            # Downsample the coverage cloud, orient normals toward cameras, run BPA.
            progress_cb(0.25, "Downsampling for outdoor BPA…")
            TARGET_OUTDOOR = 300_000
            n_pts_out = len(pcd.points)
            if n_pts_out > TARGET_OUTDOOR:
                # Estimate actual point spacing from a KNN sample, then scale by
                # the downsampling ratio — avoids bounding-box volume errors.
                _samp = np.random.choice(n_pts_out, min(2000, n_pts_out), replace=False)
                _kd = o3d.geometry.KDTreeFlann(pcd)
                _knn_d = []
                for _si in _samp:
                    _, _, _d2 = _kd.search_knn_vector_3d(np.asarray(pcd.points)[_si], 2)
                    if len(_d2) > 1:
                        _knn_d.append(float(_d2[1]) ** 0.5)
                _avg_spacing = float(np.median(_knn_d)) if _knn_d else 0.05
                # voxel = spacing × (ratio)^(1/3) to achieve target count
                _ratio = n_pts_out / TARGET_OUTDOOR
                _voxel = _avg_spacing * (_ratio ** (1 / 3))
                pcd_out = pcd.voxel_down_sample(max(_voxel, _avg_spacing * 1.1))
                logger.info("[exporter] Outdoor downsample: %d → %d pts (voxel=%.4f, spacing=%.4f)",
                            n_pts_out, len(pcd_out.points), _voxel, _avg_spacing)
            else:
                pcd_out = pcd
            out_pts = np.asarray(pcd_out.points)

            # Orient normals toward camera positions (better than centroid for walk-through scenes)
            pcd_out.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=50)
            )
            try:
                import json as _j
                _cam_file = tmp / "cameras_outdoor.json"
                await storage.download(f"{project_id}/sfm/cameras.json", _cam_file)
                _cams_raw = _j.loads(_cam_file.read_text())
                from backend.workers.pipeline.coverage import _parse_camera
                _cam_locs = [_parse_camera(img, _cams_raw.get("cameras", []))["cam_loc"]
                             for img in _cams_raw.get("images", [])
                             if _parse_camera(img, _cams_raw.get("cameras", [])) is not None]
                if _cam_locs:
                    # Orient normals toward the centroid of camera positions
                    _cam_arr = np.array(_cam_locs, dtype=np.float64)
                    _cam_centroid = _cam_arr.mean(axis=0)
                    pcd_out.orient_normals_towards_camera_location(_cam_centroid)
            except Exception as _ne:
                logger.warning("[exporter] Outdoor normal orient failed (%s), using consistent", _ne)
                pcd_out.orient_normals_consistent_tangent_plane(30)

            # BPA radius from KNN sample
            _sample = np.random.choice(len(out_pts), min(1000, len(out_pts)), replace=False)
            _tree_out = o3d.geometry.KDTreeFlann(pcd_out)
            _nn_out = []
            for _si in _sample:
                _, _, _d2 = _tree_out.search_knn_vector_3d(out_pts[_si], 2)
                if len(_d2) > 1:
                    _nn_out.append(float(_d2[1]) ** 0.5)
            _avg_nn_out = float(np.median(_nn_out)) if _nn_out else 0.05
            _radii_out = [_avg_nn_out * 2, _avg_nn_out * 4, _avg_nn_out * 8]

            progress_cb(0.35, f"Running Ball Pivoting on outdoor cloud ({len(out_pts):,} pts, r~{_avg_nn_out*100:.1f}cm)…")
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd_out, o3d.utility.DoubleVector(_radii_out)
            )
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()
            mesh.compute_vertex_normals()
            method_label = f"Ball Pivoting outdoor ({len(out_pts):,} pts)"
        else:
            # Poisson for indoor rooms — fills enclosed spaces well
            progress_cb(0.25, "Running Poisson surface reconstruction (depth=9)…")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9
            )
            densities = np.asarray(densities)
            density_threshold = float(np.quantile(densities, 0.05))
            mesh.remove_vertices_by_mask(densities < density_threshold)
            mesh.compute_vertex_normals()
            method_label = "Poisson"

        # Transfer vertex colours — use the MVS cloud for objects (real RGB),
        # the coverage cloud for rooms (coverage heatmap)
        color_pcd = mesh_pcd if is_object and 'mesh_pcd' in dir() else pcd
        if color_pcd.has_colors() and len(mesh.vertices) > 0:
            pcd_tree = o3d.geometry.KDTreeFlann(color_pcd)
            vert_arr = np.asarray(mesh.vertices)
            pcd_colors = np.asarray(color_pcd.colors)
            vert_colors = np.zeros((len(vert_arr), 3), dtype=np.float64)
            for i, v in enumerate(vert_arr):
                _, idx, _ = pcd_tree.search_knn_vector_3d(v, 1)
                vert_colors[i] = pcd_colors[idx[0]]
            mesh.vertex_colors = o3d.utility.Vector3dVector(vert_colors)

        o3d.io.write_triangle_mesh(str(obj_path), mesh)
        n_tris = len(mesh.triangles)
        progress_cb(0.65, f"{method_label} mesh: {n_tris:,} triangles")

    except Exception as exc:
        logger.warning(
            f"[exporter] Mesh reconstruction failed ({exc}); "
            "falling back to point-only .obj"
        )
        with open(obj_path, "w") as fh:
            for p in np.asarray(pcd.points):
                fh.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        progress_cb(0.65, f"Fallback point-only OBJ ({n_pts:,} vertices)")

    await storage.upload(obj_path, obj_key)
    exports.append({
        "label": "Mesh (.obj)",
        "key": obj_key,
        "mime_type": "model/obj",
    })
    progress_cb(0.70, "OBJ export complete")

    # ── 4. LAS — laspy point-format 2 with RGB ────────────────────────────────
    import laspy

    las_path = tmp / "output.las"
    las_key = f"{project_id}/exports/output.las"

    header = laspy.LasHeader(point_format=2, version="1.4")
    las = laspy.LasData(header=header)

    pts_arr = np.asarray(pcd.points)
    las.x = pts_arr[:, 0]
    las.y = pts_arr[:, 1]
    las.z = pts_arr[:, 2]

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)  # float64 in [0, 1]
        las.red   = (colors[:, 0] * 65535).astype(np.uint16)
        las.green = (colors[:, 1] * 65535).astype(np.uint16)
        las.blue  = (colors[:, 2] * 65535).astype(np.uint16)
    else:
        las.red   = np.zeros(n_pts, dtype=np.uint16)
        las.green = np.zeros(n_pts, dtype=np.uint16)
        las.blue  = np.zeros(n_pts, dtype=np.uint16)

    las.write(str(las_path))
    progress_cb(0.90, f"LAS written: {n_pts:,} points")

    await storage.upload(las_path, las_key)
    exports.append({
        "label": "LiDAR Exchange (.las)",
        "key": las_key,
        "mime_type": "application/octet-stream",
    })
    progress_cb(1.0, "All exports complete.")

    return {"exports": exports, "dimensions": dimensions}
