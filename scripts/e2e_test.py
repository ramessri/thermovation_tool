#!/usr/bin/env python3
"""
End-to-end pipeline test.

Creates a project, uploads the sample video, launches the pipeline,
polls until Phase 1 completes (AWAITING_SCALE_CONFIRMATION), picks
the best auto-detected anchor (or falls back to a manual scale),
confirms scale to trigger Phase 2, polls until complete, and prints
a final summary.

Usage:
    python scripts/e2e_test.py
    python scripts/e2e_test.py --api http://localhost:8000 --timeout 3600
    python scripts/e2e_test.py --manual-scale-mm 1000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_VIDEO = REPO_ROOT / "samples" / "20260509_232204.mp4"

POLL_INTERVAL = 10  # seconds

# ── ANSI helpers ──────────────────────────────────────────────────────────────
G = "\033[32m"; Y = "\033[33m"; R = "\033[31m"; B = "\033[34m"; RESET = "\033[0m"
def ok(msg):      print(f"{G}✓{RESET} {msg}")
def warn(msg):    print(f"{Y}⚠{RESET}  {msg}")
def fail(msg):    print(f"{R}✗{RESET} {msg}")
def info(msg):    print(f"{B}→{RESET} {msg}")
def section(msg): print(f"\n{B}── {msg} {'─'*(55-len(msg))}{RESET}")


def api(base: str, method: str, path: str, **kwargs) -> dict:
    url = f"{base}{path}"
    resp = getattr(requests, method)(url, timeout=60, **kwargs)
    if not resp.ok:
        fail(f"{method.upper()} {path} → HTTP {resp.status_code}")
        print(resp.text[:500])
        resp.raise_for_status()
    return resp.json()


def poll(base: str, task_id: str, timeout: int, label: str) -> dict:
    """
    Poll GET /api/jobs/{task_id} until status == SUCCESS or FAILURE.
    Returns the final result dict.
    """
    deadline = time.time() + timeout
    last_print = ""
    spin = ["|", "/", "—", "\\"]
    tick = 0

    while time.time() < deadline:
        try:
            data = api(base, "get", f"/api/jobs/{task_id}")
        except Exception as e:
            warn(f"poll error: {e}; retrying…")
            time.sleep(POLL_INTERVAL)
            continue

        status = data.get("status", "PENDING")
        result = data.get("result") or {}

        # Extract progress info from result when available.
        current = result.get("current_stage", "")
        progress = result.get("progress", 0)
        msg = result.get("message", "")
        line = f"{status}  {current}  {msg}  ({int(progress*100)}%)" if current else status

        if line != last_print:
            sys.stdout.write("\r" + " " * 80 + "\r")
            if current:
                info(f"[{label}] {current} — {msg} ({int(progress*100)}%)")
            last_print = line

        if status == "SUCCESS":
            sys.stdout.write("\r" + " " * 80 + "\r")
            ok(f"[{label}] done")
            return result

        if status in ("FAILURE", "REVOKED"):
            sys.stdout.write("\r" + " " * 80 + "\r")
            fail(f"[{label}] {status}")
            traceback = data.get("traceback") or (result if isinstance(result, str) else "")
            print(str(traceback)[-800:])
            sys.exit(1)

        sys.stdout.write(f"\r  {spin[tick % 4]}  [{label}] {status}  {current or ''}   ")
        sys.stdout.flush()
        tick += 1
        time.sleep(POLL_INTERVAL)

    fail(f"[{label}] timed out after {timeout}s")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Photogram end-to-end pipeline test")
    parser.add_argument("--api",  default="http://localhost:8000")
    parser.add_argument("--video", default=str(SAMPLE_VIDEO))
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Max seconds to wait per phase (default: 3600)")
    parser.add_argument("--manual-scale-mm", type=float, default=None,
                        help="Skip auto-anchor and use this scale in mm (e.g. 1000 = 1 m)")
    args = parser.parse_args()

    base = args.api.rstrip("/")
    video = Path(args.video)

    if not video.exists():
        fail(f"Video not found: {video}")
        sys.exit(1)

    t_start = time.time()

    # ─────────────────────────────────────────────────────────────────────────
    section("Step 1 — create project")
    proj = api(base, "post", "/api/projects/", json={
        "name":        f"e2e-{int(time.time())}",
        "description": "Automated end-to-end test",
    })
    pid = proj["id"]
    ok(f"Project created: {pid}")

    # ─────────────────────────────────────────────────────────────────────────
    section("Step 2 — upload video")
    info(f"Uploading {video.name}  ({video.stat().st_size / 1e6:.1f} MB)…")
    with open(video, "rb") as f:
        upload = api(base, "post", f"/api/projects/{pid}/uploads",
                     files={"file": (video.name, f, "video/mp4")})
    upload_id    = upload["upload_id"]
    storage_key  = upload["storage_key"]
    ok(f"Upload id:     {upload_id}")
    ok(f"Storage key:   {storage_key}")

    # ─────────────────────────────────────────────────────────────────────────
    section("Step 3 — launch Phase 1")
    launch = api(base, "post", f"/api/projects/{pid}/launch",
                 params={"upload_id": upload_id})
    task1 = launch["task_id"]
    ok(f"Task id: {task1}")

    # ─────────────────────────────────────────────────────────────────────────
    section("Step 4 — Phase 1 poll  (extract → sfm → scene → mvs → depth_fusion → anchors)")
    info(f"Poll interval: {POLL_INTERVAL}s   timeout: {args.timeout}s")

    # Poll PROJECT STATUS, not the Celery task ID.
    # chain.apply_async().id is the FIRST task (extract_frames) which completes
    # in ~15 s. Polling task status would trigger scale confirmation immediately.
    # Phase 1 is truly done when project.status == 'awaiting_scale_confirmation'.
    deadline = time.time() + args.timeout
    spin = ["|", "/", "—", "\\"]
    tick = 0
    proj_now = {}
    while time.time() < deadline:
        try:
            proj_now = api(base, "get", f"/api/projects/{pid}")
        except Exception as e:
            warn(f"Project poll error: {e}")
            time.sleep(POLL_INTERVAL)
            continue

        status = proj_now.get("status", "")
        if status == "awaiting_scale_confirmation":
            ok("Phase 1 complete — project awaiting scale confirmation")
            break
        if status == "failed":
            fail("Project failed during Phase 1")
            sys.exit(1)
        if status == "complete":
            ok("Project already complete (skipping scale confirmation)")
            break

        sys.stdout.write(f"\r  {spin[tick % 4]}  Phase 1 running ({status})…   ")
        sys.stdout.flush()
        tick += 1
        time.sleep(POLL_INTERVAL)
    else:
        fail(f"Phase 1 timed out after {args.timeout}s")
        sys.exit(1)

    sys.stdout.write("\r" + " " * 60 + "\r")

    # Read Phase 1 outputs from the project record (set by auto_detect_anchors_task).
    sa = proj_now.get("scene_analysis") or {}
    if sa:
        ok(f"Scene type: {sa.get('scene_type','?')}")
        if sa.get("featureless_surface_types"):
            info(f"  Featureless surfaces: {sa['featureless_surface_types']}")
        if sa.get("thin_structure_types"):
            info(f"  Thin structures: {sa['thin_structure_types']}")
        info(f"  Depth fusion strength: {sa.get('recommended_depth_fusion_strength','?')}")

    # ─────────────────────────────────────────────────────────────────────────
    section("Step 5 — scale confirmation")

    # Cloud key comes from the job result stored in Redis. We read it from the
    # Celery task result of auto_detect_anchors (last Phase 1 task).
    # For now derive from the project's storage prefix as a fallback.
    last_job_result = api(base, "get", f"/api/jobs/{task1}")
    r1 = last_job_result.get("result") or {}
    cloud_key = (r1.get("fused_cloud_key") or r1.get("dense_cloud_key")
                 or f"{pid}/depth_fusion/fused_cloud.ply")

    if args.manual_scale_mm:
        scale   = args.manual_scale_mm
        source  = "manual"
        info(f"Manual scale: {scale} mm")

    else:
        # Candidates come from the project record (written by auto_detect_anchors_task).
        candidates = proj_now.get("anchor_candidates") or []

        if candidates:
            best = max(candidates, key=lambda c: c.get("confidence", 0))
            scale  = best.get("scale_factor") or 1000.0
            source = best.get("object_name", "auto")
            ok(f"Best anchor: '{source}'  conf={best.get('confidence',0):.2f}  "
               f"scale={scale:.1f} mm")
            info("All candidates:")
            for c in sorted(candidates, key=lambda x: x.get("confidence", 0), reverse=True):
                info(f"  {c.get('object_name','?'):32s}  conf={c.get('confidence',0):.2f}  "
                     f"scale_mm={c.get('scale_factor','?')}  frames={c.get('n_frames_detected','?')}")
        else:
            scale  = 1000.0
            source = "fallback_1m"
            warn("No anchor candidates found — using 1000 mm fallback")

    confirm = api(base, "post", f"/api/projects/{pid}/confirm_scale", json={
        "scale_factor":    scale,
        "source_object":   source,
        "dense_cloud_key": cloud_key,
    })
    task2 = confirm.get("task_id") or confirm.get("celery_task_id", "")
    ok(f"Phase 2 task id: {task2}")

    # ─────────────────────────────────────────────────────────────────────────
    section("Step 6 — Phase 2 poll  (apply_scale → coverage → export)")
    r2 = poll(base, task2, args.timeout, "phase2")

    # ─────────────────────────────────────────────────────────────────────────
    section("Results")

    proj_final = api(base, "get", f"/api/projects/{pid}")
    status = proj_final.get("status", "?")
    cov = proj_final.get("coverage_score") or r2.get("coverage_score")

    info(f"Project status:      {status}")
    if cov is not None:
        grade = "GOOD" if cov >= 0.7 else ("OK" if cov >= 0.4 else "POOR")
        bar = "█" * int(cov * 20) + "░" * (20 - int(cov * 20))
        info(f"Coverage score:      {bar}  {cov*100:.1f}%  [{grade}]")

    info(f"Scale applied:       {proj_final.get('confirmed_scale_factor', scale)} mm")
    info(f"Scale source:        {proj_final.get('confirmed_scale_source', source)}")

    exports = r2.get("exports", [])
    if exports:
        ok(f"Exports ({len(exports)}):")
        for e in exports:
            info(f"  {e.get('label','?'):8s}  {e.get('mime_type',''):25s}  {e.get('key','')}")
    else:
        warn("No exports found in phase 2 result")

    suggestions = r2.get("suggestions", [])
    if suggestions:
        info(f"Coverage suggestions ({len(suggestions)}):")
        for s in suggestions[:3]:
            info(f"  • {s}")

    elapsed = time.time() - t_start
    mins, secs = divmod(int(elapsed), 60)
    info(f"Total time:          {mins}m {secs}s")
    info(f"Frontend URL:        http://localhost:3000/projects/{pid}")

    print()
    ok(f"Pipeline completed successfully  — project {pid}")


if __name__ == "__main__":
    main()
