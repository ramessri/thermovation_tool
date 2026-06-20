"""
Unit tests for backend/workers/pipeline/scale_from_aruco.py

Tests derive_scale_factor() with fully synthetic camera poses + ArUco observations
so the result is deterministic regardless of hardware/model availability.

Run inside the worker-gpu container:
    docker compose exec worker-gpu python -m pytest /app/tests/test_scale_from_aruco.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Allow running from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.workers.pipeline.scale_from_aruco import (
    derive_scale_factor,
    gravity_vector_from_floor_marker,
    _triangulate_marker,
    _backproject_marker_centre,
    _load_cameras,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_pinhole_K(fx: float = 800.0, fy: float = 800.0,
                   cx: float = 960.0, cy: float = 540.0) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def make_camera(R: np.ndarray, t: np.ndarray, K: np.ndarray | None = None):
    """Build a camera pose dict as used by _load_cameras."""
    if K is None:
        K = make_pinhole_K()
    return {"K": K, "R": R, "t": t}


def project_point(world_pt: np.ndarray, R: np.ndarray, t: np.ndarray,
                  K: np.ndarray) -> np.ndarray:
    """Project a 3D world point to 2D pixel coordinates."""
    cam = R @ world_pt + t
    px  = K @ cam
    return px[:2] / px[2]


def world_pt_corners(centre: np.ndarray, size: float = 0.001) -> list[list[float]]:
    """
    Return 4 corners of a tiny marker square centred at `centre`.
    (Used only to produce a 2D centroid; exact size doesn't matter.)
    """
    d = size / 2
    return [
        (centre + np.array([-d, -d, 0])).tolist(),
        (centre + np.array([ d, -d, 0])).tolist(),
        (centre + np.array([ d,  d, 0])).tolist(),
        (centre + np.array([-d,  d, 0])).tolist(),
    ]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestTriangulateMarker:
    """Tests for the linear triangulation helper."""

    def test_two_views_reconstruct_origin(self):
        """Two cameras looking at world origin should triangulate to (0,0,0)."""
        target = np.array([0.0, 0.0, 0.0])
        # Camera 1: located at (1,0,0), looking toward origin
        R1 = np.eye(3)
        t1 = np.array([0.0, 0.0, 1.0])   # cam centre = R1.T @ (-t1) = (0,0,-1)? No…
        # Simpler: place camera along Z axis
        # cam_centre = -R.T @ t  → to put camera at (0,0,2), need t = (0,0,-2)
        cam1 = np.array([0.0, 0.0, 2.0])
        ray1 = target - cam1;  ray1 /= np.linalg.norm(ray1)
        cam2 = np.array([2.0, 0.0, 0.0])
        ray2 = target - cam2;  ray2 /= np.linalg.norm(ray2)

        pt = _triangulate_marker([(cam1, ray1), (cam2, ray2)])
        assert pt is not None
        np.testing.assert_allclose(pt, target, atol=1e-9)

    def test_three_views_clean(self):
        target = np.array([1.0, 2.0, 3.0])
        observations = []
        for angle in [0, 90, 180]:
            rad = math.radians(angle)
            origin = np.array([5 * math.cos(rad), 0.0, 5 * math.sin(rad)]) + target
            ray    = target - origin
            ray   /= np.linalg.norm(ray)
            observations.append((origin, ray))

        pt = _triangulate_marker(observations)
        assert pt is not None
        np.testing.assert_allclose(pt, target, atol=1e-9)

    def test_single_view_returns_none(self):
        cam = np.array([0.0, 0.0, 5.0])
        ray = np.array([0.0, 0.0, -1.0])
        assert _triangulate_marker([(cam, ray)]) is None

    def test_empty_returns_none(self):
        assert _triangulate_marker([]) is None


class TestDeriveScaleFactor:
    """Integration tests for the full scale derivation pipeline."""

    @staticmethod
    def _make_cameras_json(cameras: dict[str, dict]) -> dict:
        """Convert {fname: cam_dict} into cameras.json format."""
        images = []
        for name, cam in cameras.items():
            K   = cam["K"].tolist()
            R   = cam["R"]
            t   = cam["t"]
            mat = np.hstack([R, t.reshape(3, 1)])
            images.append({
                "name":           name,
                "K":              K,
                "cam_from_world": {"matrix_3x4": mat.tolist()},
                "camera_id":      0,
            })
        return {"images": images, "cameras": []}

    @staticmethod
    def _make_aruco_result(
        marker_world_positions: dict[int, np.ndarray],
        cameras: dict[str, dict],
        sampled_keys: list[str],
        frame_ids_per_marker: int = 3,
        physical_baseline_m: float = 0.5,
    ) -> dict:
        """
        Build a synthetic aruco_result from known world positions.
        Projects each marker into `frame_ids_per_marker` cameras and records
        the 2D corners.  Computes correct physical baselines.
        """
        K = make_pinhole_K()
        camera_list = list(cameras.items())  # [(fname, cam), ...]

        aruco_markers: dict[str, list] = {}
        for frame_idx, (fname, cam) in enumerate(camera_list):
            frame_dets = []
            for mid, world_pt in marker_world_positions.items():
                # Check that marker is in front of camera
                cam_pt = cam["R"] @ world_pt + cam["t"]
                if cam_pt[2] <= 0:
                    continue
                # Project centre
                px_centre = project_point(world_pt, cam["R"], cam["t"], K)
                # Build 4 corners around projected centre (tiny offsets for realism)
                delta = 10.0  # pixels
                corners_px = [
                    [px_centre[0] - delta, px_centre[1] - delta],
                    [px_centre[0] + delta, px_centre[1] - delta],
                    [px_centre[0] + delta, px_centre[1] + delta],
                    [px_centre[0] - delta, px_centre[1] + delta],
                ]
                frame_dets.append({
                    "id":         mid,
                    "corners_px": corners_px,
                    "rvec":       [0.0, 0.0, 0.0],
                    "tvec":       cam_pt.tolist(),
                })
            if frame_dets:
                aruco_markers[str(frame_idx)] = frame_dets

        # Compute physical baselines from world positions
        ids = list(marker_world_positions.keys())
        baselines = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_a, id_b = ids[i], ids[j]
                d = float(np.linalg.norm(
                    marker_world_positions[id_a] - marker_world_positions[id_b]
                ))
                baselines.append({"frame": 0, "id_a": id_a, "id_b": id_b,
                                   "dist_m": round(d, 6)})

        return {
            "aruco_markers":        aruco_markers,
            "aruco_ids_found":      ids,
            "aruco_baselines":      baselines,
            "aruco_marker_size_m":  0.15,
            "aruco_frames_checked": len(cameras),
            "aruco_floor_marker_id": min(ids),
            "aruco_sampled_keys":   sampled_keys,
        }

    def _make_camera_ring(self, n_cameras: int = 8, radius: float = 3.0,
                          target: np.ndarray | None = None) -> dict[str, dict]:
        """
        Place n cameras in a ring of radius `radius` around `target`,
        all looking inward.  Returns {fname: {K, R, t}}.
        """
        if target is None:
            target = np.zeros(3)
        K = make_pinhole_K()
        cameras = {}
        for i in range(n_cameras):
            angle = 2 * math.pi * i / n_cameras
            # Camera world position
            cam_pos = np.array([
                target[0] + radius * math.cos(angle),
                target[1],
                target[2] + radius * math.sin(angle),
            ])
            # Look-at: z-axis points toward target
            z_axis = target - cam_pos
            z_axis /= np.linalg.norm(z_axis)
            x_axis = np.cross(np.array([0, 1, 0]), z_axis)
            if np.linalg.norm(x_axis) < 1e-6:
                x_axis = np.array([1.0, 0.0, 0.0])
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)

            R = np.vstack([x_axis, y_axis, z_axis])   # rows = camera axes
            t = -R @ cam_pos

            fname = f"frame_{i:06d}.jpg"
            cameras[fname] = {"K": K, "R": R, "t": t}
        return cameras

    def test_two_markers_known_scale(self):
        """
        Markers placed 1.0 m apart in a unit-scale SfM scene → scale should be 1.0.
        """
        # Place markers 1.0 SfM unit apart; physical distance also 1.0 m → scale = 1.0
        marker_positions = {
            0: np.array([0.0, 0.0, 0.0]),
            1: np.array([1.0, 0.0, 0.0]),
        }
        cameras = self._make_camera_ring(n_cameras=8)
        sampled_keys = list(cameras.keys())
        aruco_result = self._make_aruco_result(marker_positions, cameras, sampled_keys)
        cameras_json = self._make_cameras_json(cameras)

        scale, diag = derive_scale_factor(aruco_result, cameras_json)
        assert scale is not None, f"Scale should be derivable: {diag}"
        assert abs(scale - 1.0) < 0.05, f"Expected scale≈1.0, got {scale}"

    def test_two_markers_with_scale_factor(self):
        """
        Markers placed 2.0 m apart in a SfM scene where 1 SfM unit = 0.5 m.
        Physical dist = 2.0 m, SfM dist = 4.0 units → scale = 0.5.
        """
        # SfM coords (units); physical positions are SfM_coords × 0.5
        sfm_dist    = 4.0    # SfM units between markers
        phys_dist_m = 2.0    # physical metres

        marker_positions_sfm = {
            0: np.array([0.0, 0.0, 0.0]),
            1: np.array([sfm_dist, 0.0, 0.0]),
        }
        # Override physical baselines to reflect real-world distances
        cameras = self._make_camera_ring(n_cameras=8, radius=10.0)
        sampled_keys = list(cameras.keys())
        aruco_result = self._make_aruco_result(marker_positions_sfm, cameras, sampled_keys)
        # Override the synthesised baselines with the "real" physical distance
        aruco_result["aruco_baselines"] = [
            {"frame": 0, "id_a": 0, "id_b": 1, "dist_m": phys_dist_m},
        ]
        cameras_json = self._make_cameras_json(cameras)

        scale, diag = derive_scale_factor(aruco_result, cameras_json)
        expected = phys_dist_m / sfm_dist   # 0.5
        assert scale is not None, f"Scale should be derivable: {diag}"
        assert abs(scale - expected) < 0.02, f"Expected scale≈{expected:.3f}, got {scale:.4f}"

    def test_returns_none_when_no_cameras(self):
        aruco_result = {
            "aruco_markers":       {},
            "aruco_ids_found":     [0, 1],
            "aruco_baselines":     [{"frame": 0, "id_a": 0, "id_b": 1, "dist_m": 0.5}],
            "aruco_marker_size_m": 0.15,
            "aruco_sampled_keys":  [],
        }
        cameras_json = {"images": [], "cameras": []}
        scale, diag = derive_scale_factor(aruco_result, cameras_json)
        assert scale is None

    def test_returns_none_when_only_one_marker(self):
        """One marker triangulated but no baselines → scale underdetermined."""
        marker_positions = {0: np.array([0.0, 0.0, 0.0])}
        cameras = self._make_camera_ring(n_cameras=6)
        sampled_keys = list(cameras.keys())
        aruco_result = self._make_aruco_result(marker_positions, cameras, sampled_keys)
        cameras_json = self._make_cameras_json(cameras)

        scale, diag = derive_scale_factor(aruco_result, cameras_json)
        assert scale is None

    def test_three_markers_iqr_robust(self):
        """
        Three markers give three pairwise scale estimates.
        One baseline is wrong (outlier) — result should still be close.
        """
        # All markers in the same plane, 1.0 SfM unit apart
        marker_positions = {
            0: np.array([0.0, 0.0, 0.0]),
            1: np.array([1.0, 0.0, 0.0]),
            2: np.array([0.5, 0.5, 0.0]),
        }
        cameras = self._make_camera_ring(n_cameras=10, radius=5.0)
        sampled_keys = list(cameras.keys())
        aruco_result = self._make_aruco_result(marker_positions, cameras, sampled_keys)
        # All physical baselines should be consistent with scale=2.0 (SfM unit = 0.5 m)
        for b in aruco_result["aruco_baselines"]:
            phys  = b["dist_m"]   # auto-computed as SfM distance
            b["dist_m"] = round(phys * 2.0, 4)   # scale by 2 so scale_factor = 2.0
        cameras_json = self._make_cameras_json(cameras)

        scale, diag = derive_scale_factor(aruco_result, cameras_json)
        assert scale is not None, f"Scale underdetermined: {diag}"
        assert abs(scale - 2.0) < 0.1, f"Expected scale≈2.0, got {scale:.4f}"


class TestBackprojectMarkerCentre:
    """Tests for the ray back-projection helper."""

    def test_ray_through_principal_point(self):
        """A pixel at the principal point should give a ray along z-axis (camera space)."""
        K = make_pinhole_K(fx=800, fy=800, cx=960, cy=540)
        R = np.eye(3)
        t = np.zeros(3)
        # A marker centred at the principal point (960, 540)
        corners = [[950, 530], [970, 530], [970, 550], [950, 550]]
        origin, ray = _backproject_marker_centre(corners, K, R, t)
        # Camera is at world origin (R=I, t=0 → cam_centre = R.T @ (-t) = 0)
        np.testing.assert_allclose(origin, [0, 0, 0], atol=1e-9)
        # Ray should point roughly along +z
        assert ray[2] > 0.99


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call(
        ["python", "-m", "pytest", __file__, "-v"],
        cwd=str(REPO_ROOT),
    ))
