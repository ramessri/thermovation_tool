"""
LingBot-Map depth-fusion stage — optional densification, runs after refine_cloud.

Pipeline:
  1. Subprocess into the isolated lingbot venv (torch 2.8) to run depth inference
     on the extracted frames -> per-frame npz {depth, intrinsic, conf, rgb, names}.
  2. Per-frame robust AFFINE depth calibration (scale + offset) against the COLMAP
     MVS cloud projected into each view (skip poorly-calibrated frames).
  3. TSDF fusion of the calibrated depth maps using COLMAP's per-frame poses
     (from cameras.json) -> a single fused surface (cloud + mesh) in the SfM frame.
  4. DBSCAN floater removal (drop small disconnected clusters).
  5. Scale to metric via confirmed_scale_factor; upload as a SEPARATE artifact.

This module runs in the MAIN worker env (torch 2.4) and uses only numpy/open3d/
scipy — it must NEVER import the `lingbot` package (torch 2.8 ABI clash). All
torch-2.8 work happens in the subprocess.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from backend.core.config import settings
from backend.workers.pipeline.depth_fusion_common import robust_affine

logger = logging.getLogger(__name__)



def _ensure_checkpoint() -> Path:
    """Lazy-download the LingBot checkpoint to the persistent models volume."""
    ckpt = Path(settings.LINGBOT_CHECKPOINT)
    if ckpt.exists():
        return ckpt
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[lingbot] checkpoint missing — downloading %s/%s …",
                settings.LINGBOT_CHECKPOINT_REPO, settings.LINGBOT_CHECKPOINT_FILE)
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id=settings.LINGBOT_CHECKPOINT_REPO,
        filename=settings.LINGBOT_CHECKPOINT_FILE,
        local_dir=str(ckpt.parent),
    )
    # hf may nest under subfolders; symlink/copy to the configured path if needed
    p = Path(path)
    if p != ckpt and not ckpt.exists():
        ckpt.symlink_to(p)
    return ckpt


def _run_inference(frames_dir: Path, ckpt: Path, out_npz: Path,
                   progress_cb: Callable[[float, str], None], mask_sky: bool = False):
    """Invoke the depth-inference script in the isolated lingbot venv."""
    cmd = [
        settings.LINGBOT_VENV_PYTHON, settings.LINGBOT_INFER_SCRIPT,
        "--image_folder", str(frames_dir),
        "--model_path", str(ckpt),
        "--out_npz", str(out_npz),
        "--num_scale_frames", str(settings.LINGBOT_NUM_SCALE_FRAMES),
        "--camera_num_iterations", str(settings.LINGBOT_CAMERA_ITERS),
        "--windowed_threshold", str(settings.LINGBOT_WINDOWED_THRESHOLD),
        "--window_size", str(settings.LINGBOT_WINDOW_SIZE),
    ]
    if mask_sky:
        cmd.append("--mask_sky")
    import os
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logger.info("[lingbot] inference subprocess: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, text=True, env=env)
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            logger.info("[lingbot:infer] %s", line)
            if "Inference" in line:
                progress_cb(0.45, "Depth inference complete")
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"lingbot depth inference failed (exit {proc.returncode})")
    if not out_npz.exists():
        raise RuntimeError("lingbot inference produced no npz output")


def _fuse(frames_npz: Path, cameras_json: Path, dense_ply: Path,
          progress_cb: Callable[[float, str], None], scene_type: str = "object"):
    """Affine-calibrated TSDF fusion. Returns (pcd, mesh, metrics) in the SfM frame."""
    import open3d as o3d
    from scipy.spatial import cKDTree

    fr = np.load(frames_npz, allow_pickle=True)
    depth, intr, conf, rgb = fr["depth"], fr["intrinsic"], fr["conf"], fr["rgb"]
    names = list(fr["names"])
    S, H, W = depth.shape

    cj = json.load(open(cameras_json))
    w2c = {}
    for im in cj["images"]:
        M = np.array(im["cam_from_world"]["matrix_3x4"])
        T = np.eye(4); T[:3, :4] = M
        w2c[im["name"]] = T

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
    cthr = float(np.percentile(conf, settings.LINGBOT_CONF_PERCENTILE))
    # Adaptive voxel: large rooms at a fine voxel produce 10M+ pts / 20M+ tris and
    # OOM the worker during mesh extraction + merge. Floor at res*2 (objects), but
    # coarsen for big scenes by the core (outlier-robust) extent so point/tri count
    # and RAM stay bounded.
    core = cmpts[rdist < np.percentile(rdist, 98)]
    diag = float(np.linalg.norm(core.max(0) - core.min(0)))
    voxel_len = max(res * 2.0, diag / 700.0)
    logger.info("[lingbot] fusion: frames=%d NN_res=%.4f diag=%.2f voxel=%.4f max_depth=%.2f conf>=%.2f",
                S, res, diag, voxel_len, max_depth, cthr)

    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_len, sdf_trunc=voxel_len * 3,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    scales, used, skipped = [], 0, 0
    for i in range(S):
        n = names[i]
        if n not in w2c:
            continue
        Ki = intr[i]; fx, fy, cx, cy = Ki[0, 0], Ki[1, 1], Ki[0, 2], Ki[1, 2]
        T = w2c[n]; R, t = T[:3, :3], T[:3, 3]
        pc = cmpts @ R.T + t
        z = pc[:, 2]; front = z > 1e-3
        u = fx * pc[:, 0] / np.where(front, z, 1) + cx
        v = fy * pc[:, 1] / np.where(front, z, 1) + cy
        inb = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        a_i = b_i = None
        frac = 0.0
        if inb.sum() > 50:
            dl = depth[i][v[inb].astype(int), u[inb].astype(int)]
            zc = z[inb]; ok = dl > 1e-3
            if ok.sum() > 50:
                a_i, b_i, frac = robust_affine(dl[ok], zc[ok])
        if a_i is None or not np.isfinite(a_i) or frac < 0.4:
            skipped += 1
            continue
        scales.append(a_i)
        d = (depth[i] * a_i + b_i).astype(np.float32)
        d[conf[i] < cthr] = 0.0
        d[d <= 0] = 0.0
        color = o3d.geometry.Image(np.ascontiguousarray(rgb[i]))
        depth_img = o3d.geometry.Image(np.ascontiguousarray(d))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth_img, depth_scale=1.0, depth_trunc=max_depth,
            convert_rgb_to_intensity=False)
        intrins = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        vol.integrate(rgbd, intrins, T)
        used += 1
        if used % 20 == 0:
            progress_cb(0.5 + 0.3 * used / max(S, 1), f"Fusing frame {used}/{S}…")

    if used == 0:
        raise RuntimeError("no frames could be calibrated against COLMAP")
    scale_std = float(np.std(scales))
    logger.info("[lingbot] integrated %d frames (skipped %d) | scale med=%.3f std=%.3f",
                used, skipped, float(np.median(scales)), scale_std)

    pcd = vol.extract_point_cloud()
    n0 = len(pcd.points)
    # Safety net: if still very large, downsample before DBSCAN (near-quadratic).
    if n0 > 5_000_000:
        pcd = pcd.voxel_down_sample(voxel_len * 1.5)
        logger.info("[lingbot] pre-cluster downsample %d -> %d pts", n0, len(pcd.points))
    labels = np.array(pcd.cluster_dbscan(eps=voxel_len * 5, min_points=20))
    floaters = 0
    if labels.max() >= 0:
        counts = np.bincount(labels[labels >= 0])
        big = np.where(counts > 0.02 * counts.max())[0]
        keep = np.isin(labels, big)
        pcd = pcd.select_by_index(np.where(keep)[0])
        floaters = n0 - len(pcd.points)
    mesh = vol.extract_triangle_mesh(); mesh.compute_vertex_normals()
    logger.info("[lingbot] floater removal: %d -> %d pts; mesh %d tris",
                n0, len(pcd.points), len(mesh.triangles))

    # Void-gated merge: keep COLMAP's sharp MVS detail, add LingBot points ONLY
    # where COLMAP is empty. Replacing COLMAP with the smooth TSDF surface loses
    # fine detail (fuzzier); this augments instead of replacing.
    lb_pts = np.asarray(pcd.points)
    void_thr = res * 3.0
    if len(lb_pts):
        d_near, _ = tree.query(lb_pts, k=1, workers=-1)
        fill_mask = d_near > void_thr
    else:
        fill_mask = np.zeros(0, bool)
    fill = pcd.select_by_index(np.where(fill_mask)[0])
    combined = cm + fill          # COLMAP detail + LingBot gap-fill (SfM frame)
    logger.info("[lingbot] void-gated merge: COLMAP %d + fill %d (of %d LingBot) = %d",
                len(cm.points), len(fill.points), len(lb_pts), len(combined.points))

    n_pre_clip = len(combined.points)
    cam_C = np.array([(-np.array(im["cam_from_world"]["matrix_3x4"])[:, :3].T
                       @ np.array(im["cam_from_world"]["matrix_3x4"])[:, 3])
                      for im in cj["images"]])

    # Orbit-sphere clip — OBJECT ONLY. The object sits inside the camera orbit, so
    # a sphere about the camera centroid bounds it cleanly. Outdoor gets NO sphere
    # — it keeps the same spatial extent as the COLMAP pipeline (no bound); indoor
    # likewise unbounded.
    if scene_type == "object" and len(cam_C) >= 4:
        ctr = cam_C.mean(0)
        cam_r = float(np.percentile(np.linalg.norm(cam_C - ctr, axis=1), 95))
        clip_r = cam_r * 0.95
        cpts = np.asarray(combined.points)
        keep = np.linalg.norm(cpts - ctr, axis=1) <= clip_r
        combined = combined.select_by_index(np.where(keep)[0])
        logger.info("[lingbot] orbit-sphere clip (r=%.3f): %d -> %d pts",
                    clip_r, n_pre_clip, len(combined.points))

    # Floor-down cut — object + outdoor. Removes LingBot's hallucinated sub-ground
    # points (COLMAP never produces geometry below the surface), keeping the fused
    # cloud consistent with the pipeline. Skipped for indoor_room.
    if scene_type != "indoor_room" and len(cam_C) >= 4:
        try:
            n_pre_floor = len(combined.points)
            plane, _ = combined.segment_plane(distance_threshold=res * 3,
                                              ransac_n=3, num_iterations=1000)
            nrm = np.array(plane[:3]); L = np.linalg.norm(nrm) + 1e-9
            sd = (np.asarray(combined.points) @ nrm + plane[3]) / L     # signed dist
            cam_sd = (cam_C @ nrm + plane[3]) / L
            if np.median(cam_sd) < 0:                                    # cameras must be above
                sd = -sd
            keep_f = sd >= -res * 5                                      # allow floor thickness
            combined = combined.select_by_index(np.where(keep_f)[0])
            logger.info("[lingbot] floor-down cut: %d -> %d pts (removed %d below floor)",
                        n_pre_floor, len(combined.points), n_pre_floor - len(combined.points))
        except Exception as e:
            logger.warning("[lingbot] floor cut skipped (%s)", e)

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


async def run_lingbot_fusion(
    project_id: str,
    prev_result: dict,
    tmp: Path,
    progress_cb: Callable[[float, str], None],
    scale_factor: Optional[float] = None,
    scene_type: Optional[str] = None,
) -> dict:
    """Densify the reconstruction with LingBot depth fusion (additive artifact)."""
    import open3d as o3d
    from backend.core.storage import get_storage

    storage = get_storage()

    frame_keys = prev_result.get("frame_keys") or []
    camera_poses_key = prev_result.get("camera_poses_key")
    if not frame_keys or not camera_poses_key:
        logger.warning("[%s] lingbot_fusion: missing frames/camera poses — skipping", project_id)
        return prev_result

    # 1. Download frames (preserve basenames so they match cameras.json names).
    # Cap the count: long walkthroughs OOM depth inference even in windowed mode,
    # and indoor coverage is highly redundant — evenly subsample above the cap.
    progress_cb(0.05, "Preparing frames for depth inference…")
    max_f = settings.LINGBOT_MAX_FRAMES
    if max_f and len(frame_keys) > max_f:
        step = (len(frame_keys) + max_f - 1) // max_f
        kept = frame_keys[::step]
        logger.info("[%s] lingbot_fusion: subsampling %d -> %d frames (step %d)",
                    project_id, len(frame_keys), len(kept), step)
        frame_keys = kept
    frames_dir = tmp / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for k in frame_keys:
        await storage.download(k, frames_dir / Path(k).name)

    # COLMAP artifacts (SfM frame): cameras.json + raw MVS dense cloud
    cameras_json = tmp / "cameras.json"
    await storage.download(camera_poses_key, cameras_json)
    dense_ply = tmp / "dense.ply"
    await storage.download(f"{project_id}/mvs/dense.ply", dense_ply)

    # 2. Depth inference in the isolated lingbot venv
    progress_cb(0.15, "Running LingBot depth inference (isolated venv)…")
    ckpt = _ensure_checkpoint()
    frames_npz = tmp / "frames.npz"
    _run_inference(frames_dir, ckpt, frames_npz, progress_cb,
                   mask_sky=(scene_type == "outdoor"))

    # 3-5. Affine calibration + TSDF fusion + floater removal (main env)
    progress_cb(0.5, "Calibrating depth and fusing (TSDF)…")
    pcd, mesh, metrics = _fuse(frames_npz, cameras_json, dense_ply, progress_cb,
                               scene_type=scene_type or "indoor_room")  # project default; "object" would orbit-clip a room away

    # 6. Scale to metric (same convention as apply_known_scale: metric = sfm * factor)
    if scale_factor and scale_factor > 0:
        pcd.scale(scale_factor, center=np.zeros(3))
        mesh.scale(scale_factor, center=np.zeros(3))
        metrics["scaled_metric"] = True

    # 7. Upload separate densified artifacts
    progress_cb(0.9, "Uploading densified cloud + mesh…")
    cloud_key = f"{project_id}/lingbot/fused.ply"
    mesh_key = f"{project_id}/lingbot/fused_mesh.obj"
    cloud_local = tmp / "fused.ply"
    mesh_local = tmp / "fused_mesh.obj"
    o3d.io.write_point_cloud(str(cloud_local), pcd)
    o3d.io.write_triangle_mesh(str(mesh_local), mesh)
    await storage.upload(cloud_local, cloud_key)
    await storage.upload(mesh_local, mesh_key)

    progress_cb(1.0, f"Densified: {metrics['fused_point_count']:,} pts, "
                     f"{metrics['fused_mesh_tris']:,} tris from {metrics['frames_used']} frames")

    result = dict(prev_result)
    result["lingbot_cloud_key"] = cloud_key
    result["lingbot_mesh_key"] = mesh_key
    result["lingbot_fusion"] = metrics
    return result
