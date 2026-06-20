#!/usr/bin/env python3
"""
Inject a synthetic detect_aruco manifest so the smoke harness can continue
to feature_matching/sfm/mvs/etc. when test footage doesn't have real markers.

Usage (inside worker-gpu container):
    python /app/scripts/inject_mock_aruco.py

This reads the existing `samples/manifests/extract_frames.json` and writes
a `samples/manifests/detect_aruco.json` with realistic-looking ArUco fields.

The synthetic scale_from_aruco stage will later produce scale=None (because
no real SfM cameras will match the fake frame observations), which is fine for
smoke-testing the downstream stages.  apply_scale will pass through the cloud
unscaled, and coverage/export will still run.

WARNING: This is for smoke-testing only.  Real scans MUST have physical markers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT     = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = REPO_ROOT / "samples" / "manifests"


def main():
    extract_path = MANIFESTS_DIR / "extract_frames.json"
    if not extract_path.exists():
        print(f"ERROR: {extract_path} not found. Run extract_frames stage first.")
        sys.exit(1)

    prev = json.loads(extract_path.read_text())
    frame_keys = prev.get("frame_keys", [])

    if not frame_keys:
        print("ERROR: extract_frames manifest has no frame_keys.")
        sys.exit(1)

    n_sampled  = min(150, len(frame_keys))
    step       = max(1, len(frame_keys) // n_sampled)
    sampled_keys = frame_keys[::step]

    # Synthesise 2 markers (IDs 0 and 1) visible in the first 5 sampled frames.
    # Corners are plausible for a 1920×1080 frame.
    aruco_markers: dict[str, list] = {}
    for i in range(min(5, len(sampled_keys))):
        frame_dets = []
        # Marker 0 — bottom-left of frame
        frame_dets.append({
            "id":         0,
            "corners_px": [[200, 800], [300, 800], [300, 900], [200, 900]],
            "rvec":       [0.0, 0.0, 0.0],
            "tvec":       [0.0, 0.0, 1.5],  # 1.5 m away
        })
        # Marker 1 — bottom-right of frame
        frame_dets.append({
            "id":         1,
            "corners_px": [[1620, 800], [1720, 800], [1720, 900], [1620, 900]],
            "rvec":       [0.0, 0.0, 0.0],
            "tvec":       [0.5, 0.0, 1.5],  # 0.5 m to the right, 1.5 m away
        })
        aruco_markers[str(i)] = frame_dets

    # Synthetic baselines from the solvePnP tvecs above
    # dist_m ≈ |tvec_1 - tvec_0| = 0.5 m
    aruco_baselines = [
        {"frame": 0, "id_a": 0, "id_b": 1, "dist_m": 0.5},
        {"frame": 1, "id_a": 0, "id_b": 1, "dist_m": 0.5},
        {"frame": 2, "id_a": 0, "id_b": 1, "dist_m": 0.5},
    ]

    aruco_result = {
        "aruco_markers":        aruco_markers,
        "aruco_ids_found":      [0, 1],
        "aruco_baselines":      aruco_baselines,
        "aruco_marker_size_m":  0.15,
        "aruco_frames_checked": n_sampled,
        "aruco_floor_marker_id": 0,
        "aruco_sampled_keys":   sampled_keys,
        "_mock":                True,  # flag so scale_from_aruco knows this is synthetic
    }

    result = dict(prev)
    result["aruco_result"] = aruco_result

    out = MANIFESTS_DIR / "detect_aruco.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"[inject_mock_aruco] Wrote {out}")
    print(f"  Markers: IDs [0, 1] — synthetic (not real detections)")
    print(f"  Baseline: 0.5 m (synthetic)")
    print(f"  Sampled keys: {len(sampled_keys)} frames")
    print()
    print("  Next: python /app/scripts/smoke.py --stage feature_matching")
    print("  NOTE: scale_from_aruco will produce scale=None (no SfM match) — that's expected.")


if __name__ == "__main__":
    main()
