"""
Stage 3: Structure from Motion using pycolmap, fed by the LightGlue
matches from Stage 2.

Pipeline:
  1. Download matches.h5 (which carries DISK keypoints + LightGlue
     pair-wise match indices) and all frame JPEGs.
  2. Build a fresh COLMAP database. One camera model per unique image
     resolution (stills and multi-resolution videos each get their own),
     with focal length scaled proportionally from the primary video prior.
  3. Inject DISK keypoints + LightGlue matches directly into the DB
     so COLMAP doesn't have to re-extract / re-match.
  4. Geometric verification via pycolmap.verify_matches (RANSAC F).
  5. Run pycolmap.incremental_mapping to recover camera poses + a
     sparse 3D reconstruction.
  6. Pick the largest reconstruction, extract per-point RGB, export
     a sparse.ply + a cameras.json with per-image intrinsics +
     world-from-camera extrinsics.

Returns:
    {
      "sparse_cloud_key": "...",
      "camera_poses_key": "...",
      "workspace_key": "...",        # tar.gz of database.db + sparse/0/*.bin
      "image_keys": [...],
      "registered_images": int,
      "num_points3D": int,
      "mean_reprojection_error": float,
    }
"""

from __future__ import annotations

import json
import logging
import tarfile
from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np

from backend.core.storage import get_storage

logger = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────────────

# DISK keypoint format from matcher.py is (x, y) only; pycolmap accepts
# 2-column (x,y), 4-column (x,y,scale,orient), or 6-column. We use 2-col.
INITIAL_FOCAL_FACTOR = 1.2          # focal = INITIAL_FOCAL_FACTOR * max(w, h)
CAMERA_MODEL = "SIMPLE_RADIAL"      # f, cx, cy, k (single radial param)


def _build_database(
    db_path: Path,
    image_paths: list[Path],
    h5_path: Path,
    video_metadata: Optional[dict] = None,
) -> tuple[list[int], list[tuple[int, int]], int]:
    """
    Build a COLMAP DB from DISK keypoints + LightGlue matches.

    video_metadata (optional): if present and contains focal_length_px, cx, cy,
    those values are injected as the initial camera prior instead of the heuristic.

    Returns:
        image_ids: per-frame image_id in the DB (same order as image_paths)
        pair_list: list of (image_id_i, image_id_j) injected
        camera_id: shared-camera id
    """
    import pycolmap

    if db_path.exists():
        db_path.unlink()

    # Read per-frame image sizes from the H5 file (stored by the matcher).
    # This avoids loading 1000+ images just to check their dimensions and
    # gives us the original resolution before any matcher downscaling.
    with h5py.File(h5_path, "r") as h5:
        frame_sizes: list[tuple[int, int]] = []  # (W, H) per frame
        for i in range(len(image_paths)):
            sz = h5[f"frames/{i}/image_size"][...]  # (W, H)
            frame_sizes.append((int(sz[0]), int(sz[1])))

    # Reference resolution: use video_metadata width if present (focal_length_px
    # was derived from the primary video), else fall back to the most common size.
    fl_meta  = (video_metadata or {}).get("focal_length_px")
    ref_w    = int((video_metadata or {}).get("width",  0)) or \
               max(set(frame_sizes), key=frame_sizes.count)[0]
    ref_h    = int((video_metadata or {}).get("height", 0)) or \
               max(set(frame_sizes), key=frame_sizes.count)[1]

    # Build one COLMAP camera per unique (W, H) with focal length scaled
    # proportionally — stills at 4000px and 8K video at 7680px get separate
    # cameras so COLMAP's undistortion step never sees a width mismatch.
    unique_sizes = list(dict.fromkeys(frame_sizes))  # deduplicate, preserve order
    db = pycolmap.Database.open(str(db_path))
    try:
        camera_id_map: dict[tuple[int, int], int] = {}
        for (w, h) in unique_sizes:
            scale = w / ref_w if ref_w else 1.0
            if fl_meta:
                focal = float(fl_meta) * scale
                cx_v  = float((video_metadata or {}).get("cx", ref_w / 2.0)) * scale
                cy_v  = float((video_metadata or {}).get("cy", ref_h / 2.0)) * scale
            else:
                focal = INITIAL_FOCAL_FACTOR * max(w, h)
                cx_v  = w / 2.0
                cy_v  = h / 2.0

            cam = pycolmap.Camera.create_from_model_id(
                camera_id=pycolmap.INVALID_CAMERA_ID,
                model=pycolmap.CameraModelId.SIMPLE_RADIAL,
                focal_length=float(focal),
                width=int(w),
                height=int(h),
            )
            if fl_meta and (abs(cx_v - w / 2.0) > 1.0 or abs(cy_v - h / 2.0) > 1.0):
                params = list(cam.params)
                params[1] = cx_v
                params[2] = cy_v
                cam.params = params

            cid = db.write_camera(cam, use_camera_id=False)
            camera_id_map[(w, h)] = cid
            logger.info(
                "SfM: camera %dx%d focal=%.0fpx (scale=%.3f vs ref %dx%d)",
                w, h, focal, scale, ref_w, ref_h,
            )

        # Register images, each assigned to its resolution-matched camera.
        image_ids: list[int] = []
        for p, (w, h) in zip(image_paths, frame_sizes):
            img = pycolmap.Image(
                name=p.name,
                camera_id=camera_id_map[(w, h)],
                image_id=pycolmap.INVALID_IMAGE_ID,
            )
            iid = db.write_image(img, use_image_id=False)
            image_ids.append(iid)

        # Inject keypoints + matches from H5.
        with h5py.File(h5_path, "r") as h5:
            n_frames = int(h5.attrs["n_frames"])
            assert n_frames == len(image_paths), (
                f"matches.h5 has {n_frames} frames but {len(image_paths)} images on disk"
            )
            # Keypoints
            for i in range(n_frames):
                kpts = h5[f"frames/{i}/keypoints"][...].astype(np.float32)
                # COLMAP expects (N, 2) at minimum.
                db.write_keypoints(image_ids[i], kpts)

            # Matches — write each pair.
            pair_list: list[tuple[int, int]] = []
            for name in h5["pairs"]:
                a, b = name.split("_")
                i, j = int(a), int(b)
                m = h5[f"pairs/{name}/matches"][...]
                if m.shape[0] == 0:
                    continue
                # COLMAP wants uint32 indices into kpts of img1/img2.
                m_u32 = m.astype(np.uint32)
                db.write_matches(image_ids[i], image_ids[j], m_u32)
                pair_list.append((image_ids[i], image_ids[j]))

        return image_ids, pair_list, list(camera_id_map.values())[0]
    finally:
        db.close()


def _write_pairs_file(pairs_path: Path, image_paths: list[Path], image_ids: list[int],
                     pair_list: list[tuple[int, int]]) -> None:
    """Write a colmap pairs.txt mapping image_id pairs back to filenames."""
    id_to_name = {iid: p.name for iid, p in zip(image_ids, image_paths)}
    with open(pairs_path, "w") as fh:
        for (a, b) in pair_list:
            fh.write(f"{id_to_name[a]} {id_to_name[b]}\n")


def _incremental_mapping_with_progress(
    db_path: Path,
    image_dir: Path,
    out_dir: Path,
    map_opts,
    total_images: int,
    progress_cb: Callable[[float, str], None],
    base_progress: float = 0.56,
    end_progress: float = 0.87,
    poll_interval: int = 8,
) -> dict:
    """
    Run pycolmap.incremental_mapping in a background thread and send progress
    updates every poll_interval seconds by reading the partial reconstruction
    from disk (registered image count).
    """
    import threading
    import time

    result_holder: list = [None]
    error_holder:  list = [None]

    def _run():
        try:
            import pycolmap as _pycolmap
            result_holder[0] = _pycolmap.incremental_mapping(
                database_path=str(db_path),
                image_path=str(image_dir),
                output_path=str(out_dir),
                options=map_opts,
            )
        except Exception as exc:
            error_holder[0] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    t0 = time.time()

    while thread.is_alive():
        thread.join(timeout=poll_interval)
        if not thread.is_alive():
            break

        # Count registered images across all partial reconstructions on disk
        n_reg = 0
        if out_dir.exists():
            import pycolmap as _pycolmap
            for sub in out_dir.iterdir():
                try:
                    rec = _pycolmap.Reconstruction(str(sub))
                    n_reg = max(n_reg, rec.num_reg_images())
                except Exception:
                    pass

        elapsed = time.time() - t0
        elapsed_str = (
            f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
            if elapsed >= 60 else f"{int(elapsed)}s"
        )
        if n_reg > 0:
            frac = base_progress + (end_progress - base_progress) * min(1.0, n_reg / max(total_images, 1))
            progress_cb(frac, f"SfM: {n_reg}/{total_images} cameras registered ({elapsed_str} elapsed)")
        else:
            progress_cb(base_progress + 0.01, f"SfM: initializing reconstruction… ({elapsed_str} elapsed)")

    if error_holder[0]:
        raise error_holder[0]
    return result_holder[0] or {}


def _largest_reconstruction(recs: dict) -> "object | None":
    if not recs:
        return None
    return max(recs.values(), key=lambda r: r.num_reg_images())


def _detect_trajectory_jumps(cams_out: dict, all_names: list[str]) -> list[dict]:
    """Detect mis-registered frame blocks caused by a tracking break.

    When the camera dips into a small, mostly-featureless side space (a
    bathroom, closet, stairwell nook, etc.) and back out, COLMAP can lose
    continuous feature tracking and re-anchor the following frames to the
    wrong local point cluster — often a near-duplicate of nearby geometry
    (e.g. the same stairs reconstructed twice, offset by a few metres).

    Signature of this failure: two temporally-consecutive registered frames
    (no gap in the source video) whose camera centres are implausibly far
    apart, yet whose viewing directions are nearly identical — a spatial
    "teleport" with no corresponding change in where the camera is looking.
    """
    name_to_pose = {img["name"]: img["cam_from_world"] for img in cams_out["images"]}

    centers: dict[str, np.ndarray] = {}
    forwards: dict[str, np.ndarray] = {}
    for name, pose in name_to_pose.items():
        mat = np.asarray(pose["matrix_3x4"])
        R, t = mat[:, :3], mat[:, 3]
        centers[name] = -R.T @ t
        forwards[name] = R.T @ np.array([0.0, 0.0, 1.0])

    # Consecutive-in-source-video pairs that are both registered.
    steps = []
    for prev_name, next_name in zip(all_names, all_names[1:]):
        if prev_name in centers and next_name in centers:
            dist = float(np.linalg.norm(centers[next_name] - centers[prev_name]))
            steps.append((prev_name, next_name, dist))

    if len(steps) < 5:
        return []

    dists = np.array([s[2] for s in steps])
    median_step = float(np.median(dists))
    if median_step <= 1e-6:
        return []

    # Both an absolute and a relative-to-typical-step threshold: ordinary
    # turns/pans between oversampled bursts can produce a few-metre step with
    # a *changed* viewing direction, which is normal. A teleport is large in
    # absolute terms, far larger than the typical step, AND the camera keeps
    # looking the same way — that combination is what's anomalous.
    jumps: list[dict] = []
    for prev_name, next_name, dist in steps:
        if dist < max(2.0, 15 * median_step):
            continue
        f_prev = forwards[prev_name] / (np.linalg.norm(forwards[prev_name]) + 1e-9)
        f_next = forwards[next_name] / (np.linalg.norm(forwards[next_name]) + 1e-9)
        view_similarity = float(np.dot(f_prev, f_next))
        if view_similarity < 0.97:
            continue  # a real turn, not a teleport
        jumps.append({
            "before_frame": prev_name,
            "after_frame": next_name,
            "jump_distance": round(dist, 3),
            "median_step": round(median_step, 3),
            "view_similarity": round(view_similarity, 3),
        })

    return jumps


def _reconstruction_to_cameras_json(rec, image_paths: list[Path]) -> dict:
    """Serialize per-image intrinsics + world-from-cam pose."""
    cams_out = {"cameras": [], "images": []}
    # Cameras
    for cid, cam in rec.cameras.items():
        cams_out["cameras"].append({
            "camera_id": int(cid),
            "model": cam.model.name,
            "width": int(cam.width),
            "height": int(cam.height),
            "params": [float(x) for x in cam.params],
        })
    # Images (only those that were registered have valid poses).
    for iid in rec.reg_image_ids():
        img = rec.images[iid]
        # cam_from_world is a method in pycolmap 4.x returning a Rigid3d.
        cw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        quat = np.asarray(cw.rotation.quat)        # (qx, qy, qz, qw)
        tvec = np.asarray(cw.translation)
        mat = np.asarray(cw.matrix())              # (3, 4) world->cam
        cams_out["images"].append({
            "image_id": int(iid),
            "name": img.name,
            "camera_id": int(img.camera_id),
            "cam_from_world": {
                "qvec_xyzw": [float(x) for x in quat.ravel().tolist()],
                "tvec": [float(x) for x in tvec.tolist()],
                "matrix_3x4": [[float(v) for v in row] for row in mat.tolist()],
            },
        })
    cams_out["num_registered_images"] = int(rec.num_reg_images())
    cams_out["num_points3D"] = int(rec.num_points3D())
    cams_out["mean_reprojection_error"] = float(rec.compute_mean_reprojection_error())
    return cams_out


async def run_sfm_colmap(
    project_id: str,
    match_data_key: str,
    tmp: Path,
    progress_cb: Callable[[float, str], None],
    video_metadata: Optional[dict] = None,
) -> dict:
    import pycolmap

    storage = get_storage()
    progress_cb(0.0, "Starting SfM (pycolmap)…")

    # ── 1. Download matches.h5 + frames ───────────────────────────────────────
    h5_path = tmp / "matches.h5"
    await storage.download(match_data_key, h5_path)

    image_dir = tmp / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as h5:
        frame_keys = list(h5.attrs["frame_keys"])
    # h5py returns bytes for stored strings.
    frame_keys = [k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
                  for k in frame_keys]

    image_paths: list[Path] = []
    n = len(frame_keys)
    for i, key in enumerate(frame_keys):
        local = image_dir / Path(key).name
        await storage.download(key, local)
        image_paths.append(local)
        if i % 25 == 0:
            progress_cb(0.05 * (i + 1) / n, f"Downloaded {i+1}/{n} frames")
    progress_cb(0.10, f"Downloaded {n} frames")

    # ── 2. Build COLMAP database ──────────────────────────────────────────────
    db_path = tmp / "database.db"
    progress_cb(0.12, "Building COLMAP database (keypoints + matches)…")
    image_ids, pair_list, camera_id = _build_database(
        db_path, image_paths, h5_path, video_metadata=video_metadata
    )
    progress_cb(
        0.30,
        f"DB built: {len(image_ids)} images, {len(pair_list)} match pairs, camera_id={camera_id}",
    )

    # ── 3. Geometric verification ─────────────────────────────────────────────
    pairs_path = tmp / "pairs.txt"
    _write_pairs_file(pairs_path, image_paths, image_ids, pair_list)
    progress_cb(0.32, f"Running geometric verification on {len(pair_list)} pairs…")
    tv_opts = pycolmap.TwoViewGeometryOptions()
    pycolmap.verify_matches(str(db_path), str(pairs_path), tv_opts)
    progress_cb(0.55, "Geometric verification complete")

    # ── 4. Incremental mapping ────────────────────────────────────────────────
    out_dir = tmp / "sparse"
    out_dir.mkdir(parents=True, exist_ok=True)
    map_opts = pycolmap.IncrementalPipelineOptions()
    # Single physical camera across the whole video.
    # (CameraMode.SINGLE is the matching default since we wrote one camera.)
    # Be permissive about merging sub-reconstructions — key for video where
    # sections only connect via long-range pairs.
    map_opts.mapper.init_min_num_inliers = 50   # default 100; video can have sparse overlap
    map_opts.mapper.abs_pose_min_num_inliers = 20  # default 30
    map_opts.mapper.max_reg_trials = 5           # retry registration more times per image
    progress_cb(0.56, f"SfM: starting incremental mapping ({len(frame_keys)} frames)…")
    recs = _incremental_mapping_with_progress(
        db_path=db_path,
        image_dir=image_dir,
        out_dir=out_dir,
        map_opts=map_opts,
        total_images=len(frame_keys),
        progress_cb=progress_cb,
        base_progress=0.56,
        end_progress=0.87,
        poll_interval=8,
    )
    progress_cb(0.88, f"SfM: mapping complete — {len(recs)} reconstruction(s)")

    rec = _largest_reconstruction(recs)
    if rec is None or rec.num_reg_images() == 0:
        raise RuntimeError(
            "pycolmap.incremental_mapping returned no valid reconstruction "
            "(0 registered images). Check the LightGlue matches."
        )

    # ── 5. Export sparse PLY + cameras.json ───────────────────────────────────
    # Per-point RGB color
    try:
        rec.extract_colors_for_all_images(str(image_dir))
    except Exception as e:
        # Non-fatal: PLY still exports without color in worst case.
        logger.warning("[sfm] extract_colors_for_all_images failed: %s", e)

    sparse_ply = tmp / "sparse.ply"
    rec.export_PLY(str(sparse_ply))

    cameras_json_path = tmp / "cameras.json"
    cams_out = _reconstruction_to_cameras_json(rec, image_paths)
    cameras_json_path.write_text(json.dumps(cams_out, indent=2))

    sparse_key = f"{project_id}/sfm/sparse.ply"
    cameras_key = f"{project_id}/sfm/cameras.json"
    await storage.upload(sparse_ply, sparse_key)
    await storage.upload(cameras_json_path, cameras_key)

    # ── 6. Persist COLMAP workspace (database + reconstruction binaries) ─────
    # Downstream stages (MVS, anchor scaling) need the raw COLMAP database and
    # the reconstruction binaries (cameras.bin / images.bin / points3D.bin).
    # We also need to also write the largest reconstruction's *.bin files,
    # so re-export the chosen `rec` to a clean directory.
    workspace_export = tmp / "workspace_export"
    workspace_export.mkdir(parents=True, exist_ok=True)
    rec_export = workspace_export / "sparse" / "0"
    rec_export.mkdir(parents=True, exist_ok=True)
    rec.write_binary(str(rec_export))

    workspace_tar = tmp / "workspace.tar.gz"
    with tarfile.open(workspace_tar, "w:gz") as tar:
        tar.add(db_path, arcname="database.db")
        tar.add(rec_export / "cameras.bin", arcname="sparse/0/cameras.bin")
        tar.add(rec_export / "images.bin", arcname="sparse/0/images.bin")
        tar.add(rec_export / "points3D.bin", arcname="sparse/0/points3D.bin")
    workspace_key = f"{project_id}/sfm/workspace.tar.gz"
    await storage.upload(workspace_tar, workspace_key)

    # Compute reporting numbers
    registered = int(rec.num_reg_images())
    n_points = int(rec.num_points3D())
    mean_err = float(rec.compute_mean_reprojection_error())

    # ── Detect consecutive unregistered frame runs ────────────────────────────
    # A run of 3+ consecutive unregistered frames means a section of the room
    # was filmed without enough overlap to connect to the rest — likely a fast
    # pan or low-texture area.  Warn so the user knows where coverage broke.
    reg_names = {img.name for img in rec.images.values() if img.image_id in rec.reg_image_ids()}
    all_names = [Path(k).name for k in frame_keys]
    unreg_indices = [i for i, name in enumerate(all_names) if name not in reg_names]

    gap_warnings: list[str] = []
    sfm_gaps:     list[dict] = []   # structured gap data for the Scene Overview viewer
    if unreg_indices:
        # Group into consecutive runs
        runs: list[list[int]] = []
        run = [unreg_indices[0]]
        for idx in unreg_indices[1:]:
            if idx == run[-1] + 1:
                run.append(idx)
            else:
                runs.append(run)
                run = [idx]
        runs.append(run)

        for run in runs:
            if len(run) >= 3:
                pct_start = run[0] / len(all_names) * 100
                pct_end   = run[-1] / len(all_names) * 100
                msg = (f"{len(run)} consecutive unregistered frames "
                       f"({run[0]}–{run[-1]}, ~{pct_start:.0f}–{pct_end:.0f}% through video) — "
                       f"that section of the room may be missing. "
                       f"Re-film that area more slowly with better overlap.")
                logger.warning("[%s] SfM gap: %s", project_id, msg)
                gap_warnings.append(msg)

                # Find the registered frames bracketing this gap so the viewer
                # can draw a red segment between their known 3D positions.
                before_name = next(
                    (all_names[i] for i in range(run[0] - 1, -1, -1) if all_names[i] in reg_names),
                    None,
                )
                after_name = next(
                    (all_names[i] for i in range(run[-1] + 1, len(all_names)) if all_names[i] in reg_names),
                    None,
                )
                sfm_gaps.append({
                    "before_frame": before_name,
                    "after_frame":  after_name,
                    "n_missing":    len(run),
                    "pct_start":    round(pct_start, 1),
                    "pct_end":      round(pct_end, 1),
                })

    # ── Detect mis-registered "teleport" blocks (tracking-break duplicates) ──
    # e.g. the camera peeks into a small featureless side room and SfM loses
    # tracking, re-anchoring the rest of the sequence to a duplicate cluster.
    trajectory_jumps = _detect_trajectory_jumps(cams_out, all_names)
    jump_warnings: list[str] = []
    for j in trajectory_jumps:
        msg = (
            f"Possible duplicated/mis-registered section starting after "
            f"{j['before_frame']} (jump of {j['jump_distance']:.2f}m vs. typical "
            f"{j['median_step']:.2f}m step, same viewing direction) — this can "
            f"happen when the camera briefly enters a small featureless space "
            f"(e.g. a closet or stairwell) and tracking breaks. Re-film that "
            f"transition more slowly with overlapping frames."
        )
        logger.warning("[%s] SfM trajectory jump: %s", project_id, msg)
        jump_warnings.append(msg)

    # Embed gap/jump data in cameras.json so the frontend can render it without
    # an extra API call.
    if sfm_gaps or trajectory_jumps:
        if sfm_gaps:
            cams_out["gaps"] = sfm_gaps
        if trajectory_jumps:
            cams_out["trajectory_jumps"] = trajectory_jumps
        cameras_json_path.write_text(json.dumps(cams_out, indent=2))
        await storage.upload(cameras_json_path, cameras_key)

    drop_pct = (len(all_names) - registered) / max(len(all_names), 1) * 100
    progress_cb(
        1.0,
        f"SfM: {registered}/{len(all_names)} frames registered ({drop_pct:.0f}% dropped), "
        f"{n_points:,} points, reproj={mean_err:.2f}px"
        + (f" ⚠ {len(gap_warnings)} coverage gap(s)" if gap_warnings else "")
        + (f" ⚠ {len(jump_warnings)} possible duplicate section(s)" if jump_warnings else ""),
    )

    return {
        "sparse_cloud_key": sparse_key,
        "camera_poses_key": cameras_key,
        "workspace_key": workspace_key,
        "image_keys": frame_keys,
        "registered_images": registered,
        "num_points3D": n_points,
        "mean_reprojection_error": mean_err,
        "sfm_gap_warnings": gap_warnings,
        "sfm_gaps":         sfm_gaps,
        "sfm_jump_warnings": jump_warnings,
        "sfm_trajectory_jumps": trajectory_jumps,
    }
