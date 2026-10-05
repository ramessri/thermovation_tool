"""
Tests for HVAC wall-mount placement (hvac_placement.py + hvac_common.py) on a
synthetic metric room. No ML models are involved in this stage.

Cameras are stored in SfM units (scale factor 2) like the real cameras.json, so
the tests also cover load_registered_cameras' t-scaling into the metric frame.

Run inside the worker-gpu container:
    docker compose exec worker-gpu python -m pytest /app/tests/test_hvac_placement.py -v
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import backend.core.storage as storage_mod  # noqa: E402
from backend.core.storage import LocalStorageBackend  # noqa: E402
from backend.workers.pipeline import hvac_placement as hp  # noqa: E402
from backend.workers.pipeline.depth_fusion_common import load_registered_cameras  # noqa: E402
from backend.workers.pipeline.hvac_common import quad_is_clean, wall_basis  # noqa: E402

ROOM = np.array([4.0, 2.6, 3.0])      # metric: x, y (up), z
SCALE = 2.0                           # metres per SfM unit
W, H, F = 320, 240, 250.0
PID = "p1"
RADIATOR = [0.0, 0.6, 1.0]            # on the x=0 wall
# Mid-wall so rank 1 isn't on the floor line (a floor-level spot is correctly
# rejected for the photo overlay: the floor corner occludes its bottom edge).
RUCKLAUF = [0.0, 1.2, 2.2]


def look_at(cam_pos, target):
    z = target - cam_pos; z /= np.linalg.norm(z)
    x = np.cross([0.0, -1.0, 0.0], z); x /= np.linalg.norm(x)   # camera y = world down
    R = np.vstack([x, np.cross(z, x), z])
    return R, -R @ cam_pos


def room_cloud(rng) -> np.ndarray:
    pts = []
    for axis in range(3):
        area = np.prod(np.delete(ROOM, axis))
        for side in (0.0, ROOM[axis]):
            p = rng.random((int(3000 * area), 3)) * ROOM
            p[:, axis] = side
            pts.append(p)
    return np.vstack(pts)


def write_project(root: Path, extra_pts: np.ndarray | None = None) -> dict:
    import open3d as o3d
    rng = np.random.default_rng(0)
    pts = room_cloud(rng)
    if extra_pts is not None:
        pts = np.vstack([pts, extra_pts])
    (root / PID / "frames").mkdir(parents=True)
    (root / PID / "sfm").mkdir(parents=True)
    o3d.io.write_point_cloud(str(root / PID / "refined.ply"),
                             o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)))

    K = [[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]]
    images, frame_keys = [], []
    for i, (zc, yc) in enumerate([(0.8, 1.2), (1.5, 1.3), (2.2, 1.2), (1.5, 0.9)]):
        # level cameras (no pitch) so gravity-from-cameras is exact
        R, t_metric = look_at(np.array([2.8, yc, zc]), np.array([0.0, yc, 1.6]))
        name = f"f{i:02d}.jpg"
        cv2.imwrite(str(root / PID / "frames" / name), np.full((H, W, 3), 128, np.uint8))
        frame_keys.append(f"{PID}/frames/{name}")
        images.append({"name": name, "K": K, "width": W, "height": H,
                       "cam_from_world": {"matrix_3x4": np.hstack([R, (t_metric / SCALE)[:, None]]).tolist()}})
    (root / PID / "sfm" / "cameras.json").write_text(json.dumps({"images": images, "cameras": []}))

    return {
        "dense_cloud_key": f"{PID}/refined.ply",
        "camera_poses_key": f"{PID}/sfm/cameras.json",
        "frame_keys": frame_keys,
        "confirmed_scale_factor": SCALE,
        # floor plane from scale_from_aruco: offset in SfM units
        "fitted_planes": [{"label": "floor", "normal": [0.0, 1.0, 0.0], "offset": 0.0}],
        "wall_candidates": [{
            "normal": [1.0, 0.0, 0.0], "point_on_plane_m": [0.0, 1.3, 1.5], "offset_m": 0.0,
            "inliers": 30000, "meets_min_inliers": True, "ade20k_wall_confidence": 0.6,
        }],
        "hvac_fixtures": {"radiators": [{"position_m": RADIATOR}]},
        "rucklauf_position": {"position_m": RUCKLAUF, "method": "color_cue"},
    }


@pytest.fixture
def storage(tmp_path, monkeypatch):
    backend = LocalStorageBackend(tmp_path / "store")
    monkeypatch.setattr(storage_mod, "get_storage", lambda: backend)
    return backend


def run(prev, tmp_path):
    work = tmp_path / "work"; work.mkdir(exist_ok=True)
    return asyncio.run(hp.run_hvac_placement(PID, prev, work, lambda p, m: None))["hvac_placement"]


def test_cameras_scaled_into_metric_frame(storage, tmp_path):
    prev = write_project(storage.root)
    cams = load_registered_cameras(json.loads((storage.root / prev["camera_poses_key"]).read_text()),
                                   scale_factor=SCALE)
    centre = -cams["f01.jpg"]["R"].T @ cams["f01.jpg"]["t"]
    assert np.allclose(centre, [2.8, 1.3, 1.5])


def test_placement_on_wall_near_rucklauf_clear_of_radiator(storage, tmp_path):
    result = run(write_project(storage.root), tmp_path)
    assert result["status"] == "placed"
    cands = result["candidates"]
    assert [c["rank"] for c in cands] == list(range(1, len(cands) + 1))
    dists = [c["distance_to_rucklauf_cm"] for c in cands]
    assert dists == sorted(dists)

    u_ax, v_ax = wall_basis(np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
    for c in cands:
        corners = np.array(c["corners_world_m"])
        assert np.allclose(corners[:, 0], 0.0, atol=1e-6)              # on the x=0 wall
        assert np.linalg.norm(corners[1] - corners[0]) == pytest.approx(0.60)
        assert np.linalg.norm(corners[3] - corners[0]) == pytest.approx(0.40)
        centre = corners.mean(axis=0)
        gap = np.linalg.norm([(centre - RADIATOR) @ u_ax, (centre - RADIATOR) @ v_ax]) \
            - np.hypot(0.6, 0.4) / 2 - hp.OBSTACLE_KEEPOUT_RADIUS_M
        assert gap * 100 >= c["clearance_cm"] - 0.1 >= 45 - 0.1
        assert c["mount_height_cm"] == pytest.approx(centre[1] * 100, abs=0.1)   # metres, not SfM units
        assert c["distance_to_rucklauf_cm"] == pytest.approx(np.linalg.norm(centre - RUCKLAUF) * 100, abs=0.1)

    assert result["overlay_image_key"] and storage.exists(result["overlay_image_key"])


def test_mount_height_without_marker_floor_plane(storage, tmp_path):
    """Grid-marker scans and /reprocess have no fitted_planes — the floor comes
    from the cloud itself, so mount height is still reported in metres."""
    prev = write_project(storage.root)
    prev.pop("fitted_planes")
    for c in run(prev, tmp_path)["candidates"]:
        centre = np.array(c["corners_world_m"]).mean(axis=0)
        assert c["mount_height_cm"] == pytest.approx(centre[1] * 100, abs=1.0)


def test_no_eligible_wall(storage, tmp_path):
    prev = write_project(storage.root)
    prev["wall_candidates"][0]["ade20k_wall_confidence"] = 0.03    # ADE20K says: not a wall
    assert run(prev, tmp_path)["status"] == "no_wall_evidence"


def test_free_mode_without_rucklauf_maximizes_clearance(storage, tmp_path):
    prev = write_project(storage.root)
    prev["rucklauf_position"] = None
    result = run(prev, tmp_path)
    assert result["status"] == "no_rucklauf_free_mode"
    clear = [c["clearance_cm"] for c in result["candidates"]]
    assert clear == sorted(clear, reverse=True)


def test_occlusion_detects_furniture_in_front_of_wall(storage, tmp_path):
    prev = write_project(storage.root)
    cams = load_registered_cameras(json.loads((storage.root / prev["camera_poses_key"]).read_text()),
                                   scale_factor=SCALE)
    corners = [np.array(c) for c in ([0, 1.0, 1.2], [0, 1.0, 1.8], [0, 1.4, 1.8], [0, 1.4, 1.2])]
    import open3d as o3d
    cloud = np.asarray(o3d.io.read_point_cloud(str(storage.root / prev["dense_cloud_key"])).points)
    cam = cams["f01.jpg"]
    assert not hp._is_occluded(corners, cam, cloud)
    # a 1 m slab of "furniture" 0.8 m in front of that wall spot
    rng = np.random.default_rng(1)
    slab = np.c_[np.full(20000, 0.8), rng.uniform(0.6, 1.8, 20000), rng.uniform(0.8, 2.2, 20000)]
    assert hp._is_occluded(corners, cam, np.vstack([cloud, slab]))


def test_gravity_read_from_cam_from_world(storage):
    """Regression: gravity used to read a per-image "R" key sfm.py never writes,
    so it was always [0,1,0] whatever the camera poses were."""
    from backend.workers.pipeline.room_layout import _gravity_from_cameras
    prev = write_project(storage.root)
    cams = json.loads((storage.root / prev["camera_poses_key"]).read_text())
    assert np.allclose(_gravity_from_cameras(cams), [0, -1, 0], atol=1e-6)   # world here is y-up


def test_wall_dedup_keeps_opposite_walls():
    """Regression: opposite walls of a room centred near the origin, fitted with
    flipped normals, have equal raw offsets and used to be merged as one wall."""
    from backend.workers.pipeline.wall_plane_detection import _is_duplicate
    wall_a = {"normal": [1.0, 0, 0], "offset_m": -2.1}           # plane x = -2.1
    # plane x = +2.1 fitted with normal (-1,0,0): offset = -2.1
    assert not _is_duplicate(np.array([-1.0, 0, 0]), -2.1, [wall_a])
    # the same plane x = -2.1 re-found with the flipped normal IS a duplicate
    assert _is_duplicate(np.array([-1.0, 0, 0]), 2.1, [wall_a])
    assert _is_duplicate(np.array([1.0, 0, 0]), -2.0, [wall_a])   # 10 cm apart, same side


def test_color_cue_accepts_round_caps_rejects_squares():
    """Regression: blue poster squares passed the circularity test (2/π ≈ 0.64)
    and were taken as the Rücklauf cap."""
    from backend.workers.pipeline.locate_rucklauf import _find_color_blobs, _BLUE_HSV
    img = np.full((300, 600, 3), 200, np.uint8)
    blue = (200, 60, 20)                                             # BGR
    cv2.circle(img, (80, 150), 18, blue, -1)                         # cap, head-on
    cv2.ellipse(img, (220, 150), (20, 13), 30, 0, 360, blue, -1)     # cap seen at ~50°
    cv2.rectangle(img, (360, 130), (395, 165), blue, -1)             # poster square
    cv2.rectangle(img, (470, 140), (520, 160), blue, -1)             # label strip
    found = sorted(round(x) for x, _, _ in _find_color_blobs(img, [_BLUE_HSV]))
    assert found == [80, 220]


def test_quad_is_clean_rejects_grazing_and_out_of_frame():
    assert quad_is_clean([(10, 10), (110, 10), (110, 60), (10, 60)], W, H)
    assert not quad_is_clean([(10, 10), (100, 10), (300, 60), (210, 60)], W, H)  # sheared: diagonals 2.4x apart
    assert not quad_is_clean([(10, 10), (400, 10), (400, 60), (10, 60)], W, H)   # out of frame
    assert not quad_is_clean([(10, 10), (110, 10), None, (10, 60)], W, H)
