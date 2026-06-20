#!/usr/bin/env python3
"""
Per-stage smoke harness for the Photogram pipeline.

New pipeline:
  extract_metadata → extract_frames → detect_aruco → feature_matching →
  sfm → mvs → scale_from_aruco → apply_scale → coverage → export

Usage (inside the worker-gpu container):

    python /app/scripts/smoke.py --stage extract_metadata
    python /app/scripts/smoke.py --stage extract_frames
    python /app/scripts/smoke.py --stage detect_aruco
    python /app/scripts/smoke.py --stage feature_matching
    python /app/scripts/smoke.py --stage sfm
    python /app/scripts/smoke.py --stage mvs
    python /app/scripts/smoke.py --stage scale_from_aruco
    python /app/scripts/smoke.py --stage coverage
    python /app/scripts/smoke.py --stage export
    python /app/scripts/smoke.py --stage all

Each stage reads its input from samples/manifests/<prior_stage>.json and
writes samples/manifests/<this_stage>.json so reruns can resume mid-pipeline.

NOTE: The ArUco detect stage fails fast if no markers are found.  Print
DICT_4X4_100 markers (≥15 cm side length) and place them in the scene.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SAMPLES_DIR   = REPO_ROOT / "samples"
MANIFESTS_DIR = SAMPLES_DIR / "manifests"
MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ID = "smoke"

CANDIDATE_VIDEOS = [
    SAMPLES_DIR / "20260520_195121.mp4",
    SAMPLES_DIR / "20260509_232204.mp4",
    SAMPLES_DIR / "input.mp4",
]


def find_sample_video() -> Path:
    for p in CANDIDATE_VIDEOS:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No sample video found. Looked in: {[str(p) for p in CANDIDATE_VIDEOS]}"
    )


def manifest_path(stage: str) -> Path:
    return MANIFESTS_DIR / f"{stage}.json"


def save_manifest(stage: str, data: dict) -> None:
    path = manifest_path(stage)
    path.write_text(json.dumps(data, indent=2, default=str))
    print(f"[smoke] wrote manifest -> {path}")


def load_manifest(stage: str) -> dict:
    path = manifest_path(stage)
    if not path.exists():
        raise FileNotFoundError(
            f"Required manifest '{stage}' not found at {path}. "
            f"Run an earlier stage first."
        )
    return json.loads(path.read_text())


def progress(stage: str):
    def cb(p: float, msg: str = ""):
        print(f"[{stage}] {p*100:6.2f}%  {msg}")
    return cb


def ensure_video_in_storage() -> str:
    """Upload the sample video to local storage.  Returns the storage key."""
    from backend.core.storage import get_storage

    video   = find_sample_video()
    storage = get_storage()
    key     = f"{PROJECT_ID}/video/input.mp4"
    asyncio.run(storage.upload(video, key))
    print(f"[smoke] uploaded {video.name} -> storage://{key}")
    return key


# ── Stage 0: Extract Metadata ─────────────────────────────────────────────────

def run_extract_metadata() -> dict:
    from backend.workers.pipeline.extract_metadata import run_extract_metadata as fn

    storage_key = ensure_video_in_storage()
    tmp = Path("/tmp/photogram_smoke/extract_metadata")
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    result = asyncio.run(
        fn(
            project_id=PROJECT_ID,
            storage_key=storage_key,
            tmp=tmp,
            progress_cb=progress("extract_metadata"),
        )
    )
    meta = result.get("video_metadata", {})
    print(
        f"[extract_metadata] {meta.get('width')}×{meta.get('height')} "
        f"rot={meta.get('rotation_deg')}° "
        f"fl={meta.get('focal_length_px')} "
        f"hint={meta.get('scene_hint')} "
        f"stable_ts={len(meta.get('stable_frame_timestamps', []))}"
    )
    save_manifest("extract_metadata", result)
    return result


# ── Stage 1: Extract Frames ───────────────────────────────────────────────────

def run_extract_frames() -> dict:
    from backend.workers.pipeline.extractor import run_extract_frames as fn

    prev = load_manifest("extract_metadata")
    storage_key    = prev["storage_key"]
    video_metadata = prev.get("video_metadata")

    tmp = Path("/tmp/photogram_smoke/extract_frames")
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    result = asyncio.run(
        fn(
            project_id=PROJECT_ID,
            storage_key=storage_key,
            tmp=tmp,
            job_id="smoke-extract",
            video_metadata=video_metadata,
            progress_cb=progress("extract_frames"),
        )
    )
    result["video_metadata"] = video_metadata
    result["storage_key"]    = storage_key
    print(
        f"[extract_frames] frame_count={result.get('frame_count')} "
        f"rotation_deg={result.get('rotation_deg', 0)}"
    )
    save_manifest("extract_frames", result)
    return result


# ── Stage 2: Detect ArUco ─────────────────────────────────────────────────────

def run_detect_aruco() -> dict:
    from backend.workers.pipeline.aruco_detector import run_aruco_detection as fn

    prev           = load_manifest("extract_frames")
    frame_keys     = prev["frame_keys"]
    video_metadata = prev.get("video_metadata") or {}

    tmp = Path("/tmp/photogram_smoke/detect_aruco")
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    aruco_result = asyncio.run(
        fn(
            project_id=PROJECT_ID,
            frame_keys=frame_keys,
            video_metadata=video_metadata,
            tmp=tmp,
            progress_cb=progress("detect_aruco"),
        )
    )

    ids   = aruco_result.get("aruco_ids_found", [])
    n_bas = len(aruco_result.get("aruco_baselines", []))
    print(
        f"[detect_aruco] found {len(ids)} marker(s) — IDs {ids} — "
        f"{n_bas} inter-marker baseline(s)"
    )
    print(
        f"[detect_aruco] floor marker = {aruco_result.get('aruco_floor_marker_id')}  "
        f"frames_checked = {aruco_result.get('aruco_frames_checked')}"
    )

    result = dict(prev)
    result["aruco_result"] = aruco_result
    save_manifest("detect_aruco", result)
    return result


# ── Stage 3: Feature Matching ─────────────────────────────────────────────────

def run_feature_matching() -> dict:
    from backend.workers.pipeline.matcher import run_feature_matching as fn

    prev       = load_manifest("detect_aruco")
    frame_keys = prev["frame_keys"]

    tmp = Path("/tmp/photogram_smoke/match")
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    result = asyncio.run(
        fn(
            project_id=PROJECT_ID,
            frame_keys=frame_keys,
            tmp=tmp,
            progress_cb=progress("feature_matching"),
        )
    )
    result.update({k: v for k, v in prev.items() if k not in result})
    save_manifest("feature_matching", result)
    print(
        f"[feature_matching] pairs={result.get('pair_count')} "
        f"total_matches={result.get('total_matches')}"
    )
    return result


# ── Stage 4: SfM ─────────────────────────────────────────────────────────────

def run_sfm() -> dict:
    from backend.workers.pipeline.sfm import run_sfm_colmap as fn

    prev           = load_manifest("feature_matching")
    video_metadata = prev.get("video_metadata")

    tmp = Path("/tmp/photogram_smoke/sfm")
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    result = asyncio.run(
        fn(
            project_id=PROJECT_ID,
            match_data_key=prev["match_data_key"],
            tmp=tmp,
            video_metadata=video_metadata,
            progress_cb=progress("sfm"),
        )
    )
    result.update({k: v for k, v in prev.items() if k not in result})
    save_manifest("sfm", result)
    print(
        f"[sfm] registered={result.get('registered_images')} "
        f"points3D={result.get('num_points3D')} "
        f"mean_reproj_err={result.get('mean_reprojection_error')}"
    )
    return result


# ── Stage 5: MVS ─────────────────────────────────────────────────────────────

def run_mvs() -> dict:
    from backend.workers.pipeline.mvs import run_mvs_openmvs as fn

    prev = load_manifest("sfm")
    tmp  = Path("/tmp/photogram_smoke/mvs")
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    result = asyncio.run(
        fn(
            project_id=PROJECT_ID,
            sparse_cloud_key=prev["sparse_cloud_key"],
            camera_poses_key=prev["camera_poses_key"],
            tmp=tmp,
            progress_cb=progress("mvs"),
        )
    )
    result.update({k: v for k, v in prev.items() if k not in result})
    save_manifest("mvs", result)

    import open3d as o3d
    dense_local = REPO_ROOT / "storage" / result["dense_cloud_key"]
    if dense_local.exists():
        pcd = o3d.io.read_point_cloud(str(dense_local))
        n   = len(pcd.points)
        print(f"[mvs] dense_point_count={n:,} runtime={result.get('runtime_seconds', 0):.1f}s")
        if n == 0:
            raise RuntimeError("dense.ply contains zero points")
    return result


# ── Stage 6: Scale from ArUco ─────────────────────────────────────────────────

def run_scale_from_aruco() -> dict:
    from backend.workers.pipeline.scale_from_aruco import run_scale_from_aruco as fn

    prev = load_manifest("mvs")
    tmp  = Path("/tmp/photogram_smoke/scale_from_aruco")
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    result = asyncio.run(
        fn(
            project_id=PROJECT_ID,
            prev_result=prev,
            tmp=tmp,
            progress_cb=progress("scale_from_aruco"),
        )
    )
    save_manifest("scale_from_aruco", result)
    scale = result.get("confirmed_scale_factor")
    diag  = result.get("scale_diagnostics", {})
    if scale:
        print(f"[scale_from_aruco] scale={scale:.6f} m/unit  "
              f"n_estimates={diag.get('n_scale_estimates', 0)}  "
              f"gravity_up={result.get('gravity_up_world')}")
    else:
        print(f"[scale_from_aruco] scale NOT derivable: {diag.get('error', 'unknown')}")
    return result


# ── Stage 7: Coverage ─────────────────────────────────────────────────────────

def run_coverage() -> dict:
    from backend.workers.pipeline.coverage import run_coverage_analysis as fn

    prev             = load_manifest("scale_from_aruco")
    cloud_key        = prev.get("dense_cloud_key", "")
    camera_poses_key = prev.get("camera_poses_key", f"{PROJECT_ID}/sfm/cameras.json")

    tmp = Path("/tmp/photogram_smoke/coverage")
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    result = asyncio.run(
        fn(
            project_id=PROJECT_ID,
            scaled_cloud_key=cloud_key,
            camera_poses_key=camera_poses_key,
            tmp=tmp,
            progress_cb=progress("coverage"),
        )
    )
    result.update({k: v for k, v in prev.items() if k not in result})
    save_manifest("coverage", result)
    print(
        f"[coverage] score={result.get('coverage_score', 0):.3f} "
        f"n_points={result.get('n_points', 0):,} "
        f"suggestions={len(result.get('suggestions', []))}"
    )
    return result


# ── Stage 8: Export ───────────────────────────────────────────────────────────

def run_export() -> dict:
    from backend.workers.pipeline.exporter import run_export as fn

    prev               = load_manifest("coverage")
    coverage_cloud_key = prev.get("coverage_cloud_key", "")

    tmp = Path("/tmp/photogram_smoke/export")
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    result = asyncio.run(
        fn(
            project_id=PROJECT_ID,
            coverage_cloud_key=coverage_cloud_key,
            tmp=tmp,
            progress_cb=progress("export"),
        )
    )
    save_manifest("export", result)
    for exp in result.get("exports", []):
        print(f"[export]  {exp['label']:35s}  key={exp['key']}")
    return result


# ── Stage registry ────────────────────────────────────────────────────────────

STAGES = {
    "extract_metadata": run_extract_metadata,
    "extract_frames":   run_extract_frames,
    "detect_aruco":     run_detect_aruco,
    "feature_matching": run_feature_matching,
    "sfm":              run_sfm,
    "mvs":              run_mvs,
    "scale_from_aruco": run_scale_from_aruco,
    "coverage":         run_coverage,
    "export":           run_export,
}


def main():
    parser = argparse.ArgumentParser(description="Photogram per-stage smoke harness")
    parser.add_argument(
        "--stage",
        required=True,
        choices=list(STAGES.keys()) + ["all"],
        help="Which stage to run (or 'all' for the full pipeline).",
    )
    args = parser.parse_args()

    if args.stage == "all":
        for name in STAGES:
            print(f"\n======= stage: {name} =======")
            STAGES[name]()
    else:
        STAGES[args.stage]()


if __name__ == "__main__":
    main()
