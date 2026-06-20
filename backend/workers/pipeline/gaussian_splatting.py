"""
3D Gaussian Splatting pipeline stage using nerfstudio splatfacto.

Replaces COLMAP MVS. Takes SfM output (camera poses + registered frames)
and trains a 3DGS model that:
  - Covers featureless surfaces (walls, ceilings, floors) that MVS cannot
  - Produces a textured mesh as the primary deliverable
  - Outputs Gaussian centres as a point cloud for coverage analysis
  - Exports a .splat file for real-time WebGL rendering in the viewer

Pipeline position: after SfM, replacing run_mvs_openmvs.

Input:  workspace.tar.gz (COLMAP sparse), registered frame image keys
Output: {
    "splat_key": "{project_id}/splat/scene.splat",       # WebGL renderer
    "ply_key":   "{project_id}/splat/point_cloud.ply",   # coverage analysis
    "mesh_key":  "{project_id}/splat/mesh.obj",          # final deliverable
    "n_gaussians": int,
    "runtime_seconds": float,
}
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np

from backend.core.storage import get_storage

logger = logging.getLogger(__name__)

# Training iterations — 15K is fast (~15 min) and sufficient for room-scale scenes.
# 30K gives marginally better quality but doubles training time.
DEFAULT_ITERATIONS = int(os.environ.get("GSPLAT_ITERATIONS", "15000"))
MAX_IMAGE_SIZE = int(os.environ.get("GSPLAT_MAX_IMAGE_SIZE", "1920"))


def _write_colmap_cameras_txt(cameras_json: dict, out_path: Path, scale: float = 1.0) -> None:
    """
    Write COLMAP cameras.txt from our cameras.json format.
    Each camera: CAMERA_ID MODEL WIDTH HEIGHT params...

    `scale` is applied to width/height and the focal/principal-point entries of
    `params` so the intrinsics match pre-resized images. Distortion coefficients
    (k1, k2, p1, p2) are unitless and unchanged.
    """
    cameras_list = cameras_json.get("cameras", [])
    if isinstance(cameras_list, dict):
        cameras_list = list(cameras_list.values())

    # Number of leading params (focal + principal point) to scale per model.
    # Anything beyond is distortion (k1, k2, p1, p2, ...) and is unitless.
    _SCALABLE_PARAM_COUNT = {
        "SIMPLE_PINHOLE":  3,   # f, cx, cy
        "PINHOLE":         4,   # fx, fy, cx, cy
        "SIMPLE_RADIAL":   3,   # f, cx, cy   (k1 unitless)
        "RADIAL":          3,   # f, cx, cy   (k1, k2 unitless)
        "OPENCV":          4,   # fx, fy, cx, cy   (k1, k2, p1, p2 unitless)
        "OPENCV_FISHEYE":  4,   # fx, fy, cx, cy
        "FULL_OPENCV":     4,
        "FOV":             4,
    }

    with open(out_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cam in cameras_list:
            cam_id = cam["camera_id"]
            model = cam.get("model", "SIMPLE_RADIAL")
            w = int(round(cam["width"]  * scale))
            h = int(round(cam["height"] * scale))
            raw_params = list(cam["params"])
            n_scalable = _SCALABLE_PARAM_COUNT.get(model, 3)
            scaled_params = [
                p * scale if i < n_scalable else p
                for i, p in enumerate(raw_params)
            ]

            # gaussian-splatting only supports PINHOLE; convert from SIMPLE_RADIAL/PINHOLE.
            # SIMPLE_RADIAL params: f, cx, cy, k1  → PINHOLE: fx, fy, cx, cy
            # SIMPLE_PINHOLE params: f, cx, cy      → PINHOLE: fx, fy, cx, cy
            # Ignoring k (small, ~0.018 for modern lenses); undistortion error is negligible.
            if model in ("SIMPLE_RADIAL", "RADIAL", "SIMPLE_PINHOLE"):
                f_val, cx, cy = scaled_params[0], scaled_params[1], scaled_params[2]
                model = "PINHOLE"
                scaled_params = [f_val, f_val, cx, cy]
            # PINHOLE already correct; OPENCV has fx,fy,cx,cy as first 4 — use those directly
            elif model == "OPENCV":
                model = "PINHOLE"
                scaled_params = scaled_params[:4]

            params = " ".join(repr(p) for p in scaled_params)
            f.write(f"{cam_id} {model} {w} {h} {params}\n")


def _write_colmap_images_txt(cameras_json: dict, out_path: Path) -> None:
    """
    Write COLMAP images.txt from our cameras.json format.
    Each image: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    """
    images_list = cameras_json.get("images", [])

    with open(out_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, img in enumerate(images_list):
            img_id = i + 1
            cam_id = img["camera_id"]
            name = img["name"]
            # cam_from_world: R (3×3) + t (3,) in our format
            cfw_raw = img.get("cam_from_world", {})
            mat = cfw_raw.get("matrix_3x4") if isinstance(cfw_raw, dict) else cfw_raw
            if mat is None:
                continue
            m = np.array(mat, dtype=np.float64)
            R = m[:, :3]
            t = m[:, 3]
            # Convert rotation matrix to quaternion (COLMAP convention: QW QX QY QZ)
            tr = R[0, 0] + R[1, 1] + R[2, 2]
            if tr > 0:
                s = 0.5 / np.sqrt(tr + 1.0)
                qw = 0.25 / s
                qx = (R[2, 1] - R[1, 2]) * s
                qy = (R[0, 2] - R[2, 0]) * s
                qz = (R[1, 0] - R[0, 1]) * s
            else:
                if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                    s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                    qw = (R[2, 1] - R[1, 2]) / s
                    qx = 0.25 * s
                    qy = (R[0, 1] + R[1, 0]) / s
                    qz = (R[0, 2] + R[2, 0]) / s
                elif R[1, 1] > R[2, 2]:
                    s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                    qw = (R[0, 2] - R[2, 0]) / s
                    qx = (R[0, 1] + R[1, 0]) / s
                    qy = 0.25 * s
                    qz = (R[1, 2] + R[2, 1]) / s
                else:
                    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                    qw = (R[1, 0] - R[0, 1]) / s
                    qx = (R[0, 2] + R[2, 0]) / s
                    qy = (R[1, 2] + R[2, 1]) / s
                    qz = 0.25 * s
            tx, ty, tz = t
            f.write(f"{img_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                    f"{tx:.9f} {ty:.9f} {tz:.9f} {cam_id} {name}\n")
            f.write("\n")  # empty points line


def _write_colmap_points3d_txt(sparse_ply_path: Path, out_path: Path) -> None:
    """
    Convert sparse.ply point cloud to COLMAP points3D.txt.
    Format: POINT3D_ID X Y Z R G B ERROR TRACK[]
    """
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(sparse_ply_path))
    pts = np.asarray(pcd.points)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else np.full((len(pts), 3), 128, dtype=np.uint8)

    with open(out_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for i, (pt, col) in enumerate(zip(pts, colors)):
            f.write(f"{i+1} {pt[0]:.9f} {pt[1]:.9f} {pt[2]:.9f} "
                    f"{col[0]} {col[1]} {col[2]} 0.5\n")


def _extract_point_cloud_from_checkpoint(output_dir: Path, out_ply: Path) -> int:
    """
    Extract Gaussian centres directly from the nerfstudio checkpoint.
    Bypasses ns-export which has PyTorch serialization issues.
    The checkpoint stores Gaussian means as 'gauss_params.means'.
    """
    import open3d as o3d
    import torch

    # Find checkpoint file
    ckpt_files = sorted(output_dir.rglob("step-*.ckpt"), key=lambda f: f.stat().st_mtime)
    if not ckpt_files:
        return 0

    ckpt = ckpt_files[-1]
    try:
        # Load with weights_only=False (nerfstudio uses custom classes)
        state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        # Navigate to model state dict
        pipeline = state.get("pipeline", {})
        # Try several key paths used across nerfstudio versions
        means = None
        for key_path in [
            ["_model", "gauss_params", "means"],
            ["_model", "means"],
            ["gauss_params", "means"],
        ]:
            d = pipeline
            for k in key_path:
                d = d.get(k) if isinstance(d, dict) else getattr(d, k, None)
                if d is None:
                    break
            if d is not None:
                means = d
                break

        if means is None:
            logger.warning("[3dgs] could not find means in checkpoint keys: %s", list(pipeline.keys())[:10])
            return 0

        positions = means.numpy() if hasattr(means, 'numpy') else np.array(means)
        n = len(positions)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        pcd.colors = o3d.utility.Vector3dVector(np.full((n, 3), 0.7))  # default grey
        o3d.io.write_point_cloud(str(out_ply), pcd)
        logger.info("[3dgs] extracted %d Gaussian centres from checkpoint", n)
        return n
    except Exception as e:
        logger.warning("[3dgs] checkpoint extraction failed: %s", e)
        return 0


def _ply_to_splat(plydata, out_path: Path) -> None:
    """
    Convert a gaussian-splatting output PLY to the packed 32-byte-per-Gaussian
    .splat format consumed by @mkkellogg/gaussian-splats-3d WebGL viewer.

    Layout (32 bytes):  xyz(3×f32) | scale(3×u8) | rgba(4×u8) | rot(4×u8)
    """
    import struct
    v = plydata["vertex"]
    n = len(v)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def sh0_to_color(f):
        # SH degree-0 coefficient → linear colour → clamp to [0,1]
        return np.clip(f * 0.28209 + 0.5, 0, 1)

    xs = np.array(v["x"], dtype=np.float32)
    ys = np.array(v["y"], dtype=np.float32)
    zs = np.array(v["z"], dtype=np.float32)

    # Scales are log-space in the PLY
    sx = np.exp(np.array(v["scale_0"], dtype=np.float32))
    sy = np.exp(np.array(v["scale_1"], dtype=np.float32))
    sz = np.exp(np.array(v["scale_2"], dtype=np.float32))

    opacity = sigmoid(np.array(v["opacity"], dtype=np.float32))

    r = sh0_to_color(np.array(v["f_dc_0"], dtype=np.float32))
    g = sh0_to_color(np.array(v["f_dc_1"], dtype=np.float32))
    b = sh0_to_color(np.array(v["f_dc_2"], dtype=np.float32))

    # Rotations stored as w,x,y,z quaternion
    rw = np.array(v["rot_0"], dtype=np.float32)
    rx = np.array(v["rot_1"], dtype=np.float32)
    ry = np.array(v["rot_2"], dtype=np.float32)
    rz = np.array(v["rot_3"], dtype=np.float32)
    rnorm = np.sqrt(rw**2 + rx**2 + ry**2 + rz**2) + 1e-8
    rw /= rnorm; rx /= rnorm; ry /= rnorm; rz /= rnorm

    def f2u8(arr):
        return (np.clip(arr, 0, 1) * 255).astype(np.uint8)

    def scale_to_u8(s):
        # log-encode scale to [0,255]
        return (np.clip((np.log(np.clip(s, 1e-6, None)) + 10) / 20, 0, 1) * 255).astype(np.uint8)

    buf = bytearray(n * 32)
    for i in range(n):
        offset = i * 32
        struct.pack_into("<fff", buf, offset, xs[i], ys[i], zs[i])
        buf[offset+12] = scale_to_u8(np.array([sx[i]]))[0]
        buf[offset+13] = scale_to_u8(np.array([sy[i]]))[0]
        buf[offset+14] = scale_to_u8(np.array([sz[i]]))[0]
        buf[offset+15] = f2u8(np.array([r[i]]))[0]
        buf[offset+16] = f2u8(np.array([g[i]]))[0]
        buf[offset+17] = f2u8(np.array([b[i]]))[0]
        buf[offset+18] = f2u8(np.array([opacity[i]]))[0]
        buf[offset+19] = 0  # pad
        buf[offset+20] = f2u8(np.array([(rw[i]+1)/2]))[0]
        buf[offset+21] = f2u8(np.array([(rx[i]+1)/2]))[0]
        buf[offset+22] = f2u8(np.array([(ry[i]+1)/2]))[0]
        buf[offset+23] = f2u8(np.array([(rz[i]+1)/2]))[0]
        # remaining 8 bytes unused
    out_path.write_bytes(bytes(buf))
    logger.info("[3dgs] wrote %d Gaussians to %s (%d bytes)", n, out_path.name, len(buf))


async def run_gaussian_splatting(
    project_id: str,
    workspace_key: str,
    camera_poses_key: str,
    sparse_cloud_key: str,
    frame_keys: list[str],
    tmp: Path,
    progress_cb: Callable[[float, str], None],
    n_iterations: int = DEFAULT_ITERATIONS,
) -> dict:
    """
    Train a 3DGS model using nerfstudio splatfacto and export all deliverables.
    """
    t0 = time.time()
    storage = get_storage()

    progress_cb(0.0, "3DGS: downloading inputs…")

    # ── 1. Download SfM outputs ───────────────────────────────────────────────
    ws_tar = tmp / "workspace.tar.gz"
    await storage.download(workspace_key, ws_tar)

    cameras_local = tmp / "cameras.json"
    await storage.download(camera_poses_key, cameras_local)
    cameras_json = json.loads(cameras_local.read_text())

    sparse_ply = tmp / "sparse.ply"
    await storage.download(sparse_cloud_key, sparse_ply)

    # ── 2. Write COLMAP text format that nerfstudio understands ───────────────
    progress_cb(0.05, "3DGS: preparing COLMAP input format…")
    colmap_dir = tmp / "colmap" / "sparse" / "0"
    colmap_dir.mkdir(parents=True, exist_ok=True)
    images_dir = tmp / "colmap" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Subsample frames to keep memory bounded. Splatfacto caches all images as
    # float32 tensors — 350 frames at 1920x1080 is ~8.7GB before model + grads.
    # 30GB host with no swap is OOM territory.
    max_frames = int(os.environ.get("GSPLAT_MAX_FRAMES", "100"))
    if len(frame_keys) > max_frames:
        stride = len(frame_keys) / max_frames
        idxs = sorted({int(i * stride) for i in range(max_frames)})
        frame_keys = [frame_keys[i] for i in idxs if i < len(frame_keys)]
        logger.info("[3dgs] subsampled frames %d → %d (GSPLAT_MAX_FRAMES)", len(idxs), len(frame_keys))

    # Filter cameras_json.images to the kept subset so images.txt matches what's on disk.
    kept_names = {Path(k).name for k in frame_keys}
    all_images = cameras_json.get("images", [])
    cameras_json = {**cameras_json, "images": [img for img in all_images if img.get("name") in kept_names]}
    logger.info("[3dgs] images.txt entries: %d (of %d registered)", len(cameras_json["images"]), len(all_images))

    # Compute resize scale up-front so intrinsics match the resized images.
    cam_list_pre = cameras_json.get("cameras", [])
    if isinstance(cam_list_pre, dict):
        cam_list_pre = list(cam_list_pre.values())
    orig_max_dim = max(cam_list_pre[0]["width"], cam_list_pre[0]["height"]) if cam_list_pre else MAX_IMAGE_SIZE
    intrinsics_scale = (MAX_IMAGE_SIZE / orig_max_dim) if orig_max_dim > MAX_IMAGE_SIZE else 1.0
    logger.info("[3dgs] orig camera dim=%s, intrinsics_scale=%.4f", orig_max_dim, intrinsics_scale)

    _write_colmap_cameras_txt(cameras_json, colmap_dir / "cameras.txt", scale=intrinsics_scale)
    _write_colmap_images_txt(cameras_json, colmap_dir / "images.txt")
    _write_colmap_points3d_txt(sparse_ply, colmap_dir / "points3D.txt")

    # ── 3. Download + pre-resize registered frames ───────────────────────────
    # Nerfstudio asks interactively before downscaling; pre-resize here so
    # we can always pass --downscale-factor 1 and avoid the prompt.
    from PIL import Image as _PIL_Image
    n_frames = len(frame_keys)
    progress_cb(0.08, f"3DGS: downloading {n_frames} frames…")
    for i, key in enumerate(frame_keys):
        fname = Path(key).name
        local = images_dir / fname
        await storage.download(key, local)
        # Resize if larger than MAX_IMAGE_SIZE
        try:
            img = _PIL_Image.open(local)
            if max(img.size) > MAX_IMAGE_SIZE:
                scale = MAX_IMAGE_SIZE / max(img.size)
                new_w = int(img.size[0] * scale)
                new_h = int(img.size[1] * scale)
                img = img.resize((new_w, new_h), _PIL_Image.LANCZOS)
                img.save(local, "JPEG", quality=90)
        except Exception as _resize_err:
            logger.warning("3DGS: could not resize %s: %s", fname, _resize_err)
        if i % 30 == 0:
            progress_cb(0.08 + 0.10 * i / n_frames, f"3DGS: frame {i+1}/{n_frames}")

    # ── 4. Train with original gaussian-splatting repo ───────────────────────
    # Uses diff-gaussian-rasterization (not gsplat/nerfstudio) — simpler CUDA
    # kernels that don't trigger driver panics on NVIDIA 595.x.
    output_dir = tmp / "gs_output"
    output_dir.mkdir(exist_ok=True)
    progress_cb(0.20, f"3DGS: training ({n_iterations} iterations)…")

    gs_script = os.environ.get(
        "GAUSSIAN_SPLATTING_SCRIPT",
        "/opt/gaussian-splatting/train.py",
    )
    train_cmd = [
        "python", gs_script,
        "-s", str(tmp / "colmap"),
        "--model_path", str(output_dir),
        "--iterations", str(n_iterations),
        "--save_iterations", str(n_iterations),
        "--test_iterations", "-1",   # skip eval to save time
        "--quiet",
    ]

    env = {
        **os.environ,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "4"),
    }

    proc = subprocess.Popen(
        train_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, text=True, env=env,
        cwd=str(tmp / "colmap"),
    )

    step = 0
    stdout_lines = []
    for line in proc.stdout:
        line_s = line.strip()
        stdout_lines.append(line_s)
        # tqdm writes lines like "Training progress:  10%|... [1500/15000 ...]"
        # also plain "[ITER 1500]" style
        try:
            if "Training progress" in line_s and "/" in line_s:
                nums = [t for t in line_s.split() if "/" in t]
                if nums:
                    step = int(nums[0].split("/")[0].lstrip("[").strip())
            elif line_s.startswith("[") and "]" in line_s:
                token = line_s[1:line_s.index("]")]
                step = int(token.strip())
        except Exception:
            pass
        if step and step % 500 == 0:
            frac = 0.20 + 0.60 * min(step / n_iterations, 1.0)
            progress_cb(frac, f"3DGS training: step {step}/{n_iterations}")

    proc.wait()
    train_log = tmp / "train.log"
    train_log.write_text("\n".join(stdout_lines))
    logger.info("[3dgs] train rc=%s, step=%s, lines=%d", proc.returncode, step, len(stdout_lines))
    logger.info("[3dgs] last 20 lines:\n%s", "\n".join(stdout_lines[-20:]))

    if proc.returncode != 0 or step < max(50, n_iterations // 100):
        tail = "\n".join(stdout_lines[-40:])
        raise RuntimeError(
            f"gaussian-splatting training failed (rc={proc.returncode}, step={step}/{n_iterations})\n"
            f"--- last 40 lines ---\n{tail[-2000:]}"
        )
    progress_cb(0.82, "3DGS: training complete, collecting outputs…")

    # ── 5. Collect output PLY + convert to .splat ─────────────────────────────
    # train.py saves: <model_path>/point_cloud/iteration_N/point_cloud.ply
    ply_out = output_dir / "point_cloud" / f"iteration_{n_iterations}" / "point_cloud.ply"
    if not ply_out.exists():
        # Fallback: find any point_cloud.ply
        candidates = sorted(output_dir.rglob("point_cloud.ply"))
        ply_out = candidates[-1] if candidates else ply_out

    n_gaussians = 0
    ply_key = None
    splat_key = None

    if ply_out.exists():
        from plyfile import PlyData
        plydata = PlyData.read(str(ply_out))
        n_gaussians = len(plydata["vertex"])
        logger.info("[3dgs] output PLY: %d Gaussians at %s", n_gaussians, ply_out)

        # Upload the raw 3DGS PLY (viewer can load this directly)
        ply_key = f"{project_id}/splat/point_cloud.ply"
        await storage.upload(ply_out, ply_key)

        # Convert to packed .splat format for WebGL viewer
        splat_out = tmp / "scene.splat"
        try:
            _ply_to_splat(plydata, splat_out)
            splat_key = f"{project_id}/splat/scene.splat"
            await storage.upload(splat_out, splat_key)
            logger.info("[3dgs] .splat uploaded: %s", splat_key)
        except Exception as e:
            logger.warning("[3dgs] .splat conversion failed: %s", e)
    else:
        logger.error("[3dgs] no point_cloud.ply found in %s", output_dir)

    # Mesh extraction via Open3D Poisson
    mesh_key = None
    if ply_out.exists() and n_gaussians > 0:
        try:
            import open3d as o3d
            # Use only positions from the Gaussian PLY for Poisson
            from plyfile import PlyData as _PD
            _pd = _PD.read(str(ply_out))
            _v = _pd["vertex"]
            pts = np.column_stack([np.array(_v["x"]), np.array(_v["y"]), np.array(_v["z"])])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
            mesh_out = tmp / "mesh.obj"
            o3d.io.write_triangle_mesh(str(mesh_out), mesh)
            mesh_key = f"{project_id}/splat/mesh.obj"
            await storage.upload(mesh_out, mesh_key)
            logger.info("[3dgs] mesh: %d triangles", len(mesh.triangles))
        except Exception as e:
            logger.warning("[3dgs] mesh extraction failed: %s", e)

    runtime = time.time() - t0
    progress_cb(1.0, f"3DGS done: {n_gaussians:,} Gaussians in {runtime:.0f}s")

    return {
        "dense_cloud_key": ply_key or sparse_cloud_key,
        "splat_key": splat_key,
        "ply_key": ply_key,
        "mesh_key": mesh_key,
        "dense_point_count": n_gaussians,
        "n_gaussians": n_gaussians,
        "runtime_seconds": runtime,
    }
