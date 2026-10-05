"""
MetricAnything depth-fusion stage — optional densification, runs after
refine_cloud. Same role as lingbot_fusion.py (a second, alternative
densifier — a project can enable either, both, or neither; each produces
its own separate artifact), but architecturally simpler: MetricAnything's
own requirements.txt pins `torch<2.5.0`, which the main worker's pinned
torch==2.4.1 already satisfies — confirmed before writing this, not assumed
— so it runs in-process here with no isolated venv and no subprocess, unlike
LingBot-Map's torch-2.8 requirement.

Pipeline:
  1. For each sampled frame: predict per-pixel depth directly in this process
     (vendored MetricAnything student_depthmap model), using the REAL
     COLMAP-calibrated focal length as f_px — MetricAnything's own metric
     scale is literally parametrized by f_px (inverse_depth = canonical *
     width/f_px), so a true calibrated focal length is a strictly better
     anchor than the model card's own documented fallback (image width).
  2. Per-frame robust AFFINE calibration (scale + offset) against the COLMAP
     dense cloud projected into each view — same discipline lingbot_fusion.py
     uses (backend/workers/pipeline/depth_fusion_common.py's robust_affine):
     never trust raw monocular depth un-anchored, calibrated focal length or not.
     Frames whose depth doesn't correlate with COLMAP's are skipped.
  3. TSDF fusion of the calibrated depth maps using COLMAP's real per-frame
     poses (from cameras.json, SfM units) -> a single fused surface.
  4. DBSCAN floater removal; void-gated merge with COLMAP's own dense cloud
     (COLMAP detail kept, MetricAnything only fills genuine gaps).
  5. Scale to metric via confirmed_scale_factor; upload as a SEPARATE artifact
     (additive — does not alter dense_cloud_key).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from backend.core.config import settings
from backend.workers.pipeline.depth_fusion_common import load_registered_cameras, robust_affine

logger = logging.getLogger(__name__)

# The worker image clones the repo to /opt/metric-anything (outside the
# ./backend bind mount); METRICANYTHING_VENDOR_DIR points there.
_VENDOR_STUDENT_DEPTHMAP_DIR = Path(settings.METRICANYTHING_VENDOR_DIR) / "models" / "student_depthmap"

_model = None
_device = None

# A frame's calibrated depth must actually track COLMAP's depths. The trimmed
# fit alone can't tell: on an uninformative depth map it collapses to a ≈ 0,
# b ≈ mean depth with every point an "inlier", which would fuse a phantom
# flat surface.
_MIN_CALIBRATION_CORR = 0.5


def _load_model():
    """
    Lazy-load the vendored MetricAnything student_depthmap model (once per
    worker process). Two quirks of the vendored repo:
    (1) the importable module lives at models/student_depthmap/depth_model.py,
    not the repo root, so that subdirectory must go on sys.path;
    (2) vit_factory.py calls torch.hub.load("network", ...) with a path
    resolved from the process CWD, so the CWD must be that subdirectory for
    the from_pretrained call.
    """
    global _model, _device
    if _model is not None:
        return _model, _device

    import torch

    if not _VENDOR_STUDENT_DEPTHMAP_DIR.exists():
        raise RuntimeError(
            f"MetricAnything not found at {_VENDOR_STUDENT_DEPTHMAP_DIR} — clone "
            f"https://github.com/metric-anything/metric-anything to "
            f"METRICANYTHING_VENDOR_DIR (the worker-gpu image does this at /opt/metric-anything)"
        )

    vendor_dir = _VENDOR_STUDENT_DEPTHMAP_DIR
    if str(vendor_dir) not in sys.path:
        sys.path.insert(0, str(vendor_dir))
    from depth_model import MetricAnythingDepthMap  # type: ignore

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    cwd = os.getcwd()
    try:
        os.chdir(vendor_dir)
        _model = MetricAnythingDepthMap.from_pretrained(
            settings.METRICANYTHING_CHECKPOINT_REPO,
            filename=settings.METRICANYTHING_CHECKPOINT_FILE,
            cache_dir=settings.METRICANYTHING_CACHE_DIR,
        ).to(_device).eval()
    finally:
        os.chdir(cwd)
    logger.info("[metricanything] student_depthmap loaded on %s", _device)
    return _model, _device


def _predict_depth(image_rgb: np.ndarray, model, device: str, focal_px: float) -> np.ndarray:
    """
    Returns metric depth in metres, shape (H, W). The raw model(x)/forward()
    requires an exact img_size x img_size input and returns scale-ambiguous
    canonical inverse depth — model.infer(x, f_px=...) is the real entry point
    (resizes internally, converts to metric via inverse_depth = canonical *
    width/f_px). Applies the official ImageNet preprocessing (infer.py's).
    """
    import torch
    from torchvision.transforms import v2

    transform = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    x = transform(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        depth_m = model.infer(x, f_px=focal_px)["depth"]
    return depth_m.cpu().numpy().astype(np.float32)


def _fuse(
    frames_dir: Path, cameras_json_path: Path, dense_ply: Path,
    progress_cb: Callable[[float, str], None], scene_type: str = "object",
):
    """Affine-calibrated TSDF fusion. Returns (pcd, mesh, metrics) in the SfM frame."""
    import open3d as o3d
    from scipy.spatial import cKDTree

    model, device = _load_model()

    cameras_json = json.loads(cameras_json_path.read_text())
    cameras = load_registered_cameras(cameras_json)  # SfM units — scaled to metric at the end
    if not cameras:
        raise RuntimeError("no registered cameras in cameras.json")

    cm = o3d.io.read_point_cloud(str(dense_ply))
    cm, _ = cm.remove_statistical_outlier(20, 2.0)
    cmpts = np.asarray(cm.points)
    if len(cmpts) < 100:
        raise RuntimeError("COLMAP dense cloud too small for calibration")
    tree = cKDTree(cmpts)
    samp = cmpts[np.random.default_rng(0).choice(len(cmpts), min(20000, len(cmpts)), replace=False)]
    res = float(np.median(tree.query(samp, k=2)[0][:, 1]))      # median NN spacing
    center = np.median(cmpts, 0)
    rdist = np.linalg.norm(cmpts - center, axis=1)
    max_depth = float(np.percentile(rdist, 99) * 2.5)
    # Adaptive voxel — same reasoning as lingbot_fusion.py's _fuse(): a fine
    # fixed voxel on a large room produces 10M+ pts and OOMs mesh extraction.
    core = cmpts[rdist < np.percentile(rdist, 98)]
    diag = float(np.linalg.norm(core.max(0) - core.min(0)))
    voxel_len = max(res * 2.0, diag / 700.0)
    logger.info("[metricanything] fusion: frames=%d NN_res=%.4f diag=%.2f voxel=%.4f max_depth=%.2f",
                len(cameras), res, diag, voxel_len, max_depth)

    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_len, sdf_trunc=voxel_len * 3,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    names = sorted(n for n in cameras if (frames_dir / n).exists())
    S = len(names)
    scales, used, skipped = [], 0, 0

    for i, name in enumerate(names):
        cam = cameras[name]
        img_bgr = cv2.imread(str(frames_dir / name))
        if img_bgr is None:
            skipped += 1
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]
        K, R, t = cam["K"], cam["R"], cam["t"]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        try:
            depth_raw = _predict_depth(img_rgb, model, device, focal_px=fx)
        except Exception as e:
            logger.warning("[metricanything] depth prediction failed on %s: %s", name, e)
            skipped += 1
            continue
        if depth_raw.shape != (H, W):
            depth_raw = cv2.resize(depth_raw, (W, H), interpolation=cv2.INTER_LINEAR)

        # Calibrate against COLMAP's own triangulated points projected into
        # this same frame — never trust raw monocular depth un-anchored, a
        # real calibrated f_px above is a better prior, not a substitute.
        pc = cmpts @ R.T + t
        z = pc[:, 2]
        front = z > 1e-3
        u = fx * pc[:, 0] / np.where(front, z, 1) + cx
        v = fy * pc[:, 1] / np.where(front, z, 1) + cy
        inb = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        a_i = b_i = None
        frac = corr = 0.0
        if inb.sum() > 50:
            dl = depth_raw[v[inb].astype(int), u[inb].astype(int)]
            zc = z[inb]
            ok = dl > 1e-3
            if ok.sum() > 50:
                a_i, b_i, frac = robust_affine(dl[ok], zc[ok])
                if a_i > 0:
                    corr = float(np.corrcoef(dl[ok], zc[ok])[0, 1])
        if (a_i is None or not np.isfinite(a_i) or frac < 0.4
                or not corr >= _MIN_CALIBRATION_CORR):
            skipped += 1
            continue

        scales.append(a_i)
        d = (depth_raw * a_i + b_i).astype(np.float32)
        d[d <= 0] = 0.0
        color = o3d.geometry.Image(np.ascontiguousarray(img_rgb))
        depth_img = o3d.geometry.Image(np.ascontiguousarray(d))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth_img, depth_scale=1.0, depth_trunc=max_depth,
            convert_rgb_to_intensity=False)
        intrins = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        vol.integrate(rgbd, intrins, T)
        used += 1
        if used % 20 == 0:
            progress_cb(0.2 + 0.6 * (i + 1) / max(S, 1), f"Fusing frame {used}/{S}…")

    if used == 0:
        raise RuntimeError("no frames could be calibrated against COLMAP")
    scale_std = float(np.std(scales))
    logger.info("[metricanything] integrated %d frames (skipped %d) | scale med=%.3f std=%.3f",
                used, skipped, float(np.median(scales)), scale_std)

    pcd = vol.extract_point_cloud()
    n0 = len(pcd.points)
    if n0 > 5_000_000:
        pcd = pcd.voxel_down_sample(voxel_len * 1.5)
        logger.info("[metricanything] pre-cluster downsample %d -> %d pts", n0, len(pcd.points))
    labels = np.array(pcd.cluster_dbscan(eps=voxel_len * 5, min_points=20))
    floaters = 0
    if labels.max() >= 0:
        counts = np.bincount(labels[labels >= 0])
        big = np.where(counts > 0.02 * counts.max())[0]
        keep = np.isin(labels, big)
        pcd = pcd.select_by_index(np.where(keep)[0])
        floaters = n0 - len(pcd.points)
    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    logger.info("[metricanything] floater removal: %d -> %d pts; mesh %d tris",
                n0, len(pcd.points), len(mesh.triangles))

    # Void-gated merge — keep COLMAP's sharp MVS detail, add MetricAnything
    # points ONLY where COLMAP is empty (same reasoning as lingbot_fusion.py).
    ma_pts = np.asarray(pcd.points)
    void_thr = res * 3.0
    if len(ma_pts):
        d_near, _ = tree.query(ma_pts, k=1, workers=-1)
        fill_mask = d_near > void_thr
    else:
        fill_mask = np.zeros(0, bool)
    fill = pcd.select_by_index(np.where(fill_mask)[0])
    combined = cm + fill
    logger.info("[metricanything] void-gated merge: COLMAP %d + fill %d (of %d) = %d",
                len(cm.points), len(fill.points), len(ma_pts), len(combined.points))

    n_pre_clip = len(combined.points)
    cam_C = np.array([-cameras[n]["R"].T @ cameras[n]["t"] for n in names])

    # Orbit-sphere clip — object only (same as lingbot_fusion.py).
    if scene_type == "object" and len(cam_C) >= 4:
        ctr = cam_C.mean(0)
        cam_r = float(np.percentile(np.linalg.norm(cam_C - ctr, axis=1), 95))
        clip_r = cam_r * 0.95
        cpts = np.asarray(combined.points)
        keep = np.linalg.norm(cpts - ctr, axis=1) <= clip_r
        combined = combined.select_by_index(np.where(keep)[0])
        logger.info("[metricanything] orbit-sphere clip (r=%.3f): %d -> %d pts",
                    clip_r, n_pre_clip, len(combined.points))

    # Floor-down cut — object + outdoor (same as lingbot_fusion.py).
    if scene_type != "indoor_room" and len(cam_C) >= 4:
        try:
            n_pre_floor = len(combined.points)
            plane, _ = combined.segment_plane(distance_threshold=res * 3,
                                              ransac_n=3, num_iterations=1000)
            nrm = np.array(plane[:3])
            L = np.linalg.norm(nrm) + 1e-9
            sd = (np.asarray(combined.points) @ nrm + plane[3]) / L
            cam_sd = (cam_C @ nrm + plane[3]) / L
            if np.median(cam_sd) < 0:
                sd = -sd
            keep_f = sd >= -res * 5
            combined = combined.select_by_index(np.where(keep_f)[0])
            logger.info("[metricanything] floor-down cut: %d -> %d pts (removed %d)",
                        n_pre_floor, len(combined.points), n_pre_floor - len(combined.points))
        except Exception as e:
            logger.warning("[metricanything] floor cut skipped (%s)", e)

    metrics = {
        "fused_point_count": len(combined.points),
        "colmap_points": len(cm.points),
        "fill_points": len(fill.points),
        "clipped_points": n_pre_clip - len(combined.points),
        "fused_mesh_tris": len(mesh.triangles),
        "frames_used": used,
        "frames_skipped": skipped,
        "scale_std": round(scale_std, 4),
        "floaters_removed": floaters,
    }
    return combined, mesh, metrics


async def run_metricanything_fusion(
    project_id: str,
    prev_result: dict,
    tmp: Path,
    progress_cb: Callable[[float, str], None],
    scale_factor: Optional[float] = None,
    scene_type: Optional[str] = None,
) -> dict:
    """Densify the reconstruction with MetricAnything depth fusion (additive artifact)."""
    import open3d as o3d
    from backend.core.storage import get_storage

    storage = get_storage()

    frame_keys = prev_result.get("frame_keys") or []
    camera_poses_key = prev_result.get("camera_poses_key")
    if not frame_keys or not camera_poses_key:
        logger.warning("[%s] metricanything_fusion: missing frames/camera poses — skipping", project_id)
        return prev_result

    progress_cb(0.05, "Preparing frames…")
    max_f = settings.METRICANYTHING_MAX_FRAMES
    if max_f and len(frame_keys) > max_f:
        step = (len(frame_keys) + max_f - 1) // max_f
        kept = frame_keys[::step]
        logger.info("[%s] metricanything_fusion: subsampling %d -> %d frames (step %d)",
                    project_id, len(frame_keys), len(kept), step)
        frame_keys = kept
    frames_dir = tmp / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for k in frame_keys:
        await storage.download(k, frames_dir / Path(k).name)

    cameras_json = tmp / "cameras.json"
    await storage.download(camera_poses_key, cameras_json)
    dense_ply = tmp / "dense.ply"
    await storage.download(f"{project_id}/mvs/dense.ply", dense_ply)

    progress_cb(0.15, "Fusing MetricAnything depth (calibrated against COLMAP)…")
    try:
        # Missing scene_type means the project default (indoor_room) — never
        # "object", whose orbit-sphere clip deletes a whole room.
        pcd, mesh, metrics = _fuse(frames_dir, cameras_json, dense_ply, progress_cb,
                                   scene_type=scene_type or "indoor_room")
    finally:
        # Free GPU memory between pipeline runs — this stage isn't in a
        # subprocess like LingBot, so it shares the worker's lifetime memory.
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    if len(pcd.points) == 0:
        raise RuntimeError("fused cloud is empty after clipping — nothing to upload")

    if scale_factor and scale_factor > 0:
        pcd.scale(scale_factor, center=np.zeros(3))
        mesh.scale(scale_factor, center=np.zeros(3))
        metrics["scaled_metric"] = True

    progress_cb(0.9, "Uploading densified cloud + mesh…")
    cloud_key = f"{project_id}/metricanything/fused.ply"
    mesh_key = f"{project_id}/metricanything/fused_mesh.obj"
    cloud_local = tmp / "ma_fused.ply"
    mesh_local = tmp / "ma_fused_mesh.obj"
    o3d.io.write_point_cloud(str(cloud_local), pcd)
    o3d.io.write_triangle_mesh(str(mesh_local), mesh)
    await storage.upload(cloud_local, cloud_key)
    await storage.upload(mesh_local, mesh_key)

    progress_cb(1.0, f"Densified: {metrics['fused_point_count']:,} pts, "
                     f"{metrics['fused_mesh_tris']:,} tris from {metrics['frames_used']} frames")

    result = dict(prev_result)
    result["metricanything_cloud_key"] = cloud_key
    result["metricanything_mesh_key"] = mesh_key
    result["metricanything_fusion"] = metrics
    return result
