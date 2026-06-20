"""
Stage 4: Multi-View Stereo dense reconstruction using COLMAP via pycolmap.

Pipeline:
  1. Download the SfM workspace tarball (database.db + sparse/0/*.bin) and
     all frame images that were inputs to SfM.
  2. Extract the workspace to a local scratch directory.
  3. pycolmap.undistort_images() — writes an undistorted COLMAP workspace
     (dense/ with images/, sparse/, stereo/).
  4. pycolmap.patch_match_stereo() — GPU patch-match. We cap max_image_size
     to keep the 4070 Ti's 12 GB happy (default is -1 = no resize).
  5. pycolmap.stereo_fusion() — fuses per-view depth maps into a dense PLY.
  6. Upload dense.ply to storage.

Returns:
    {"dense_cloud_key": "..."}

Workspace persistence choice
----------------------------
The Phase 2 sfm.py was extended to also upload a tar.gz of the COLMAP
workspace (database.db + sparse/0/*.bin) under
  `{project_id}/sfm/workspace.tar.gz`
We download that tarball here rather than re-running SfM, and rather than
asking the smoke harness to keep a local cache. This makes the production
pipeline (which runs each stage as an independent Celery task with its own
ephemeral tmp dir) work without any harness gymnastics.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Callable

from backend.core.storage import get_storage


# ── Tunables ──────────────────────────────────────────────────────────────────
#
# Conservative settings for 12 GB VRAM (RTX 4070 Ti).
# 1200 px cap leaves ~3-4 GB headroom vs 1600 px. COLMAP's default
# cache_size is 32 GB which OOMs immediately; cap it to 8 GB.
PATCH_MATCH_MAX_IMAGE_SIZE = 1200
PATCH_MATCH_CACHE_SIZE_GB  = 8

# The pycolmap wheels on PyPI are CPU-only — they bind against a COLMAP built
# without CUDA, so pycolmap.patch_match_stereo() raises
# "Dense stereo reconstruction requires CUDA". To unblock dense reconstruction
# we shell out to the system `colmap` binary, which we build with CUDA in the
# worker-gpu Docker image. Everything else (undistort, IO) still goes through
# pycolmap for typing/error messages.
COLMAP_BIN = os.environ.get("COLMAP_BIN", "/usr/local/bin/colmap")


def _extract_workspace(tar_path: Path, dest: Path) -> None:
    """Extract the SfM workspace tarball into `dest`."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(dest)


def _count_ply_points(ply_path: Path) -> int:
    """Read the binary/ascii PLY header for the vertex count without loading data."""
    with open(ply_path, "rb") as fh:
        for _ in range(50):
            line = fh.readline()
            if not line:
                break
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("element vertex"):
                try:
                    return int(text.split()[-1])
                except (ValueError, IndexError):
                    return -1
            if text == "end_header":
                break
    return -1


async def run_mvs_openmvs(
    project_id: str,
    sparse_cloud_key: str,
    camera_poses_key: str,
    tmp: Path,
    progress_cb: Callable[[float, str], None],
    mvs_params_override: dict | None = None,
) -> dict:
    """
    Note: the function name is preserved for tasks.py compatibility, even
    though we now use COLMAP rather than OpenMVS.
    """
    import pycolmap

    storage = get_storage()
    t0 = time.time()
    progress_cb(0.0, "Starting dense reconstruction (COLMAP patch_match_stereo)…")

    # ── 1. Resolve manifest of upstream stage to locate the workspace + frames
    # We rely on the convention used elsewhere: cameras_json holds the list of
    # registered images; the workspace tarball lives next to it. The chain
    # in tasks.py also passes prev_result so we have other keys, but we keep
    # this stage's signature unchanged and derive the workspace key from the
    # camera_poses_key prefix.
    sfm_prefix = camera_poses_key.rsplit("/", 1)[0]  # e.g. "smoke/sfm"
    workspace_key = f"{sfm_prefix}/workspace.tar.gz"

    # Download cameras.json so we know which frames are registered.
    cams_path = tmp / "cameras.json"
    await storage.download(camera_poses_key, cams_path)
    cams = json.loads(cams_path.read_text())
    registered_names = [img["name"] for img in cams.get("images", [])]
    if not registered_names:
        raise RuntimeError(
            "MVS: cameras.json has no registered images — SfM produced an empty reconstruction."
        )
    progress_cb(0.05, f"Loaded {len(registered_names)} registered camera poses")

    # Download workspace tarball.
    ws_tar = tmp / "workspace.tar.gz"
    await storage.download(workspace_key, ws_tar)
    workspace_in = tmp / "ws_in"
    _extract_workspace(ws_tar, workspace_in)
    db_path = workspace_in / "database.db"
    sparse_in = workspace_in / "sparse" / "0"
    if not db_path.exists() or not sparse_in.exists():
        raise RuntimeError(
            f"MVS: workspace tarball at {workspace_key} is missing database.db or sparse/0/"
        )
    progress_cb(0.08, "Workspace extracted")

    # ── 2. Download the frame images that SfM registered ──────────────────────
    # All frames live under {project_id}/frames/<name>.
    image_dir = tmp / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    n_frames = len(registered_names)
    for i, name in enumerate(registered_names):
        frame_key = f"{project_id}/frames/{name}"
        await storage.download(frame_key, image_dir / name)
        if i % 25 == 0 or i + 1 == n_frames:
            progress_cb(
                0.08 + 0.12 * (i + 1) / n_frames,
                f"Downloaded frame {i+1}/{n_frames}",
            )

    # ── 3. Undistort ──────────────────────────────────────────────────────────
    dense_dir = tmp / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)
    progress_cb(0.22, "Running pycolmap.undistort_images…")
    pycolmap.undistort_images(
        output_path=str(dense_dir),
        input_path=str(sparse_in),
        image_path=str(image_dir),
    )
    progress_cb(0.35, "Undistortion complete")

    # ── 4. Patch-match stereo (GPU via colmap CLI) ────────────────────────────
    if not Path(COLMAP_BIN).exists():
        raise RuntimeError(
            f"MVS: required CUDA-enabled colmap binary not found at {COLMAP_BIN}. "
            f"Set $COLMAP_BIN or rebuild the worker-gpu image."
        )
    params = mvs_params_override or {}
    max_image_size = int(params.get("max_image_size", PATCH_MATCH_MAX_IMAGE_SIZE))
    geom_consistency = "true" if params.get("geom_consistency", True) else "false"
    progress_cb(
        0.36,
        f"Running colmap patch_match_stereo "
        f"(max_image_size={max_image_size})…",
    )
    pm_cmd = [
        COLMAP_BIN, "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.max_image_size", str(max_image_size),
        "--PatchMatchStereo.geom_consistency", geom_consistency,
        "--PatchMatchStereo.cache_size", str(PATCH_MATCH_CACHE_SIZE_GB),
    ]
    if "num_samples" in params:
        pm_cmd += ["--PatchMatchStereo.num_samples", str(int(params["num_samples"]))]
    # The COLMAP flag is filter_min_num_consistent, not min_num_consistent.
    if "min_num_consistent" in params:
        pm_cmd += ["--PatchMatchStereo.filter_min_num_consistent", str(int(params["min_num_consistent"]))]
    pm_log = tmp / "patch_match.log"
    pm_last_lines: list[str] = []
    pm_t0 = time.time()
    with open(pm_log, "w") as log_fh:
        proc = subprocess.Popen(pm_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log_fh.write(line)
            pm_last_lines.append(line)
            if len(pm_last_lines) > 50:
                pm_last_lines.pop(0)
            m = re.search(r'Processing view (\d+) / (\d+)', line)
            if m:
                done, total = int(m.group(1)), int(m.group(2))
                elapsed = time.time() - pm_t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = f" (~{int((total - done) / rate / 60)} min left)" if rate > 0 and done > 0 else ""
                progress_cb(
                    0.36 + 0.44 * done / max(total, 1),
                    f"Patch-match stereo: {done}/{total}{eta}",
                )
        proc.wait()
    if proc.returncode != 0:
        tail = "".join(pm_last_lines[-20:])
        raise RuntimeError(
            f"colmap patch_match_stereo failed (rc={proc.returncode}):\n"
            f"Output (tail):\n{tail}"
        )
    progress_cb(0.80, "Patch-match stereo complete")

    # ── 5. Stereo fusion ──────────────────────────────────────────────────────
    dense_ply = tmp / "dense.ply"
    progress_cb(0.82, "Running colmap stereo_fusion…")
    # Use a lower image size for fusion than patch-match to reduce peak RAM.
    fusion_image_size = min(max_image_size, 800)
    sf_cmd = [
        COLMAP_BIN, "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(dense_ply),
        "--StereoFusion.max_image_size", str(fusion_image_size),
        "--StereoFusion.min_num_pixels", "5",
    ]
    sf_log = tmp / "stereo_fusion.log"
    sf_last_lines: list[str] = []
    sf_t0 = time.time()
    sf_fusing = (0, 1)   # (done, total) from last "Fusing image" line
    sf_pts = 0           # running point count from completion lines
    sf_last_report = time.time()
    with open(sf_log, "w") as log_fh:
        proc = subprocess.Popen(sf_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log_fh.write(line)
            sf_last_lines.append(line)
            if len(sf_last_lines) > 50:
                sf_last_lines.pop(0)
            m = re.search(r'Fusing image \[(\d+)/(\d+)\]', line)
            if m:
                sf_fusing = (int(m.group(1)), int(m.group(2)))
            pts_m = re.search(r'\((\d+) points?\)', line)
            if pts_m:
                sf_pts = int(pts_m.group(1))
            if m or pts_m:
                done, total = sf_fusing
                elapsed = time.time() - sf_t0
                rate = done / elapsed if elapsed > 0 and done > 0 else 0
                eta = f" (~{int((total - done) / rate / 60)} min left)" if rate > 0 else ""
                pts_str = f", {sf_pts:,} pts" if sf_pts else ""
                progress_cb(
                    0.82 + 0.10 * done / max(total, 1),
                    f"Stereo fusion: {done}/{total}{pts_str}{eta}",
                )
                sf_last_report = time.time()
            elif time.time() - sf_last_report > 30:
                # Emit a heartbeat if COLMAP goes quiet (e.g. consistency filtering phase)
                elapsed_min = (time.time() - sf_t0) / 60
                done, total = sf_fusing
                pts_str = f", {sf_pts:,} pts" if sf_pts else ""
                progress_cb(
                    0.82 + 0.10 * done / max(total, 1),
                    f"Stereo fusion running… {elapsed_min:.0f} min elapsed{pts_str}",
                )
                sf_last_report = time.time()
        proc.wait()
    if proc.returncode != 0:
        tail = "".join(sf_last_lines[-20:])
        raise RuntimeError(
            f"colmap stereo_fusion failed (rc={proc.returncode}):\n"
            f"Output (tail):\n{tail}"
        )
    # Free depth/confidence maps — no longer needed and can be several GB.
    for subdir in ("depth_maps", "consistency_graphs"):
        d = dense_dir / "stereo" / subdir
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

    n_dense = _count_ply_points(dense_ply)
    runtime = time.time() - t0
    progress_cb(
        0.95,
        f"Stereo fusion produced ~{n_dense} points in {runtime:.1f}s",
    )

    # ── 6. Upload ─────────────────────────────────────────────────────────────
    dense_key = f"{project_id}/mvs/dense.ply"
    await storage.upload(dense_ply, dense_key)
    progress_cb(1.0, f"Dense reconstruction complete: {n_dense} points")

    return {
        "dense_cloud_key": dense_key,
        "dense_point_count": int(n_dense),
        "runtime_seconds": float(runtime),
    }
