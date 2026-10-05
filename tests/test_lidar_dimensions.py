"""
Unit tests for floor/gravity detection (detect_floor_gravity)
and L×B×H extraction, both in dimensions.py, on synthetic room clouds.

Run inside the worker-gpu container:
    docker compose exec worker-gpu python -m pytest /app/tests/test_lidar_dimensions.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.workers.pipeline.dimensions import (  # noqa: E402
    compare_to_ground_truth, compute_dimensions, rotation_aligning,
)
from backend.workers.pipeline.dimensions import detect_floor_gravity  # noqa: E402

L, B, H = 4.2, 3.1, 2.6


def room_points(n_floor=20000, n_wall=20000, n_ceiling=15000, big_wall=1.0, seed=0) -> np.ndarray:
    """Points on the 6 faces of an L×B×H room, +Z up, floor at z=0."""
    rng = np.random.default_rng(seed)
    u = lambda n: rng.random(n)  # noqa: E731
    faces = [
        np.c_[u(n_floor) * L, u(n_floor) * B, np.zeros(n_floor)],
        np.c_[u(n_ceiling) * L, u(n_ceiling) * B, np.full(n_ceiling, H)],
        np.c_[u(int(n_wall * big_wall)) * L, np.zeros(int(n_wall * big_wall)), u(int(n_wall * big_wall)) * H],
        np.c_[u(n_wall) * L, np.full(n_wall, B), u(n_wall) * H],
        np.c_[np.zeros(n_wall), u(n_wall) * B, u(n_wall) * H],
        np.c_[np.full(n_wall, L), u(n_wall) * B, u(n_wall) * H],
    ]
    pts = np.vstack(faces)
    return pts + rng.normal(scale=0.003, size=pts.shape)


def random_rotation(seed=1) -> np.ndarray:
    q = np.random.default_rng(seed).normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def to_pcd(pts: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def test_rotation_aligning_maps_a_to_b():
    a = np.array([0.3, -0.5, 0.8]); a /= np.linalg.norm(a)
    R = rotation_aligning(a, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(R @ a, [0, 0, 1], atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_floor_gravity_found_in_rotated_room(seed):
    R = random_rotation(seed)
    up, inliers = detect_floor_gravity(to_pcd(room_points() @ R.T))
    assert up is not None and inliers > 0
    assert np.dot(up, R @ [0, 0, 1]) > np.cos(np.radians(2))   # points up, not down


def test_big_wall_does_not_outvote_cluttered_floor():
    """A wall with 2× the floor's points must not be picked as vertical."""
    R = random_rotation(7)
    pts = room_points(n_floor=10000, n_wall=10000, big_wall=2.0, n_ceiling=5000)
    up, _ = detect_floor_gravity(to_pcd(pts @ R.T))
    assert np.dot(up, R @ [0, 0, 1]) > np.cos(np.radians(2))


def test_no_planes_returns_none():
    pts = np.random.default_rng(0).normal(size=(5000, 3))
    assert detect_floor_gravity(to_pcd(pts)) == (None, 0)


def test_dimensions_of_rotated_room():
    R = random_rotation(4)
    pcd = to_pcd(room_points() @ R.T)
    dims = compute_dimensions(pcd, (R @ [0, 0, 1]).tolist(), scale=1.0)
    assert dims["length_m"] == pytest.approx(L, rel=0.02)
    assert dims["breadth_m"] == pytest.approx(B, rel=0.02)
    assert dims["height_m"] == pytest.approx(H, rel=0.02)
    assert dims["footprint_m2"] == pytest.approx(L * B, rel=0.04)


def test_dimensions_apply_scale():
    dims = compute_dimensions(to_pcd(room_points() * 1000), [0, 0, 1], scale=0.001)
    assert dims["height_m"] == pytest.approx(H, rel=0.02)


def test_ground_truth_compares_only_supplied_fields():
    gt = compare_to_ground_truth({"length_m": 4.2, "breadth_m": 3.0, "height_m": 2.5},
                                 {"length_m": 4.0, "height_m": 2.5})
    assert set(gt) == {"length_m", "height_m", "mean_abs_error_pct"}
    assert gt["length_m"]["error_pct"] == pytest.approx(5.0)
    assert gt["mean_abs_error_pct"] == pytest.approx(2.5)
