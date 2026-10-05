"""
Unit tests for the MetricAnything densifier's fusion logic (metricanything_fusion._fuse)
and its shared helpers (depth_fusion_common).

The depth network is replaced by a fake that returns ray-cast ground-truth depth
distorted by a different scale/offset per frame — the drift the per-frame affine
calibration exists to remove. No GPU or model checkpoint needed.

Run inside the worker-gpu container:
    docker compose exec worker-gpu python -m pytest /app/tests/test_metricanything_fusion.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.workers.pipeline import metricanything_fusion as maf  # noqa: E402
from backend.workers.pipeline.depth_fusion_common import (  # noqa: E402
    load_registered_cameras, robust_affine,
)

ROOM = np.array([4.0, 3.0, 2.5])     # x, y (up), z extents
W, H, F = 160, 120, 110.0


def look_at(cam_pos, target):
    z = target - cam_pos; z /= np.linalg.norm(z)
    x = np.cross([0.0, -1.0, 0.0], z); x /= np.linalg.norm(x)
    R = np.vstack([x, np.cross(z, x), z])
    return R, -R @ cam_pos


def room_mesh():
    return o3d.geometry.TriangleMesh.create_box(*ROOM)


def write_scene(tmp: Path, hole: bool):
    """Frames + cameras.json + a COLMAP-like dense cloud (optionally with a wall hole)."""
    frames = tmp / "frames"; frames.mkdir()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(room_mesh()))
    K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])
    centre = ROOM / 2
    images, depths = [], {}
    for i, ang in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False)):
        pos = centre + np.array([0.6 * np.cos(ang), 0.1, 0.6 * np.sin(ang)])
        R, t = look_at(pos, centre + np.array([1.5 * np.cos(ang), -0.2, 1.5 * np.sin(ang)]))
        ext = np.eye(4); ext[:3, :3] = R; ext[:3, 3] = t
        rays = scene.create_rays_pinhole(o3d.core.Tensor(K), o3d.core.Tensor(ext), W, H)
        hit = scene.cast_rays(rays)["t_hit"].numpy()
        r = rays.numpy()
        pts = r[..., :3] + r[..., 3:] * hit[..., None]
        depths[f"f{i:02d}.jpg"] = ((pts @ R.T) + t)[..., 2].astype(np.float32)
        cv2.imwrite(str(frames / f"f{i:02d}.jpg"), np.full((H, W, 3), 128, np.uint8))
        images.append({"name": f"f{i:02d}.jpg", "K": K.tolist(), "width": W, "height": H,
                       "cam_from_world": {"matrix_3x4": np.hstack([R, t[:, None]]).tolist()}})
    (tmp / "cameras.json").write_text(json.dumps({"images": images, "cameras": []}))

    rng = np.random.default_rng(0)
    pts = []
    for axis in range(3):            # uniform density (~6k pts/m²) on all 6 faces, like MVS
        area = np.prod(np.delete(ROOM, axis))
        for side in (0.0, ROOM[axis]):
            p = rng.random((int(6000 * area), 3)) * ROOM
            p[:, axis] = side
            pts.append(p)
    pts = np.vstack(pts)
    if hole:   # remove a 1×1 m patch of the x=0 wall
        keep = ~((pts[:, 0] < 0.01) & (np.abs(pts[:, 1] - 1.5) < 0.5) & (np.abs(pts[:, 2] - 1.25) < 0.5))
        pts = pts[keep]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(np.full((len(pts), 3), 0.5))
    o3d.io.write_point_cloud(str(tmp / "dense.ply"), pcd)
    return frames, depths


@pytest.fixture
def fake_model(monkeypatch):
    """Replace the network: true depth × per-frame drift (scale 0.6–1.4, offset ±0.2)."""
    state = {"depths": {}, "calls": 0}
    rng = np.random.default_rng(3)

    def predict(img_rgb, model, device, focal_px):
        name = sorted(state["depths"])[state["calls"]]
        state["calls"] += 1
        return state["depths"][name] / rng.uniform(0.6, 1.4) - rng.uniform(-0.2, 0.2)

    monkeypatch.setattr(maf, "_load_model", lambda: (None, "cpu"))
    monkeypatch.setattr(maf, "_predict_depth", predict)
    return state


def test_robust_affine_recovers_scale_with_outliers():
    rng = np.random.default_rng(0)
    dl = rng.uniform(0.5, 5, 2000)
    zc = 1.7 * dl + 0.3
    zc[:200] += rng.uniform(1, 5, 200)          # 10 % gross outliers
    a, b, frac = robust_affine(dl, zc)
    assert a == pytest.approx(1.7, rel=1e-3) and b == pytest.approx(0.3, abs=1e-2)
    assert frac == pytest.approx(0.9, abs=0.02)


def test_load_registered_cameras_from_colmap_params():
    cams = load_registered_cameras({
        "cameras": [{"camera_id": 1, "model": "SIMPLE_RADIAL", "width": 640, "height": 480,
                     "params": [500.0, 320.0, 240.0, 0.01]}],
        "images": [{"name": "a.jpg", "camera_id": 1,
                    "cam_from_world": {"matrix_3x4": np.hstack([np.eye(3), [[1], [2], [3]]]).tolist()}}],
    })
    assert cams["a.jpg"]["K"][0, 0] == 500 and cams["a.jpg"]["K"][1, 2] == 240
    assert np.allclose(cams["a.jpg"]["t"], [1, 2, 3])


def test_fuse_calibrates_drift_and_fills_only_the_hole(tmp_path, fake_model):
    frames, fake_model["depths"] = write_scene(tmp_path, hole=True)
    pcd, mesh, m = maf._fuse(frames, tmp_path / "cameras.json", tmp_path / "dense.ply",
                             lambda p, s: None, scene_type="indoor_room")

    assert m["frames_used"] == 8 and m["frames_skipped"] == 0
    assert m["fill_points"] > 0 and len(mesh.triangles) > 0
    # Calibrated depth lands on the true surfaces: every output point is on a room face
    pts = np.asarray(pcd.points)
    face_dist = np.minimum(np.abs(pts), np.abs(pts - ROOM)).min(axis=1)
    assert np.percentile(face_dist, 99) < 0.05
    # The hole is covered: most 5 cm cells of the missing 1×1 m wall patch get points
    wall = pts[(pts[:, 0] < 0.05) & (np.abs(pts[:, 1] - 1.5) < 0.5) & (np.abs(pts[:, 2] - 1.25) < 0.5)]
    cells = {(int((y - 1.0) / 0.05), int((z - 0.75) / 0.05)) for _, y, z in wall}
    assert len(cells) / 400 > 0.7
    # Additive, not duplicating: fills are a small fraction of COLMAP's points
    assert m["fill_points"] < 0.05 * m["colmap_points"]


def test_fuse_rejects_uninformative_depth(tmp_path, fake_model, monkeypatch):
    """Noise depth must be skipped, not fused as a phantom constant-depth surface."""
    frames, fake_model["depths"] = write_scene(tmp_path, hole=False)
    monkeypatch.setattr(maf, "_predict_depth",
                        lambda *a, **k: np.random.default_rng(0).uniform(0.1, 9, (H, W)).astype(np.float32))
    with pytest.raises(RuntimeError, match="no frames could be calibrated"):
        maf._fuse(frames, tmp_path / "cameras.json", tmp_path / "dense.ply",
                  lambda p, s: None, scene_type="indoor_room")
