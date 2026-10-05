"""
Unit tests for the 3×3 grid marker: grid_marker_detector.py + grid_marker_scale.py

Detection runs on synthetic renders of the board; scale derivation uses fully
synthetic camera poses so results are deterministic.

Run inside the worker-gpu container:
    docker compose exec worker-gpu python -m pytest /app/tests/test_grid_marker.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.workers.pipeline.grid_marker_detector import (  # noqa: E402
    GRID_CM, PITCH_U_CM, PITCH_V_CM, SQUARE_CM, detect_marker,
)
from backend.workers.pipeline.grid_marker_scale import derive_grid_scale  # noqa: E402

BOARD_W_CM, BOARD_H_CM = 28.6, 20.2


# ── Helpers ───────────────────────────────────────────────────────────────────

def board_image(px_per_cm: float = 20.0) -> np.ndarray:
    """Flat render of the printed sheet (white paper, 9 black squares)."""
    w, h = int(BOARD_W_CM * px_per_cm), int(BOARD_H_CM * px_per_cm)
    img = np.full((h, w), 255, np.uint8)
    off_x = (BOARD_W_CM - 2 * PITCH_U_CM) / 2
    off_y = (BOARD_H_CM - 2 * PITCH_V_CM) / 2
    half = SQUARE_CM / 2
    for cx, cy in GRID_CM:
        x0, y0 = (off_x + cx - half) * px_per_cm, (off_y + cy - half) * px_per_cm
        cv2.rectangle(img, (int(x0), int(y0)),
                      (int(x0 + SQUARE_CM * px_per_cm), int(y0 + SQUARE_CM * px_per_cm)), 0, -1)
    return img


def warp_into_scene(board: np.ndarray, dst_quad: np.ndarray, size=(1920, 1080),
                    bg: int = 120) -> np.ndarray:
    h, w = board.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    H = cv2.getPerspectiveTransform(src, dst_quad.astype(np.float32))
    scene = np.full((size[1], size[0]), bg, np.uint8)
    mask = cv2.warpPerspective(np.full_like(board, 255), H, size)
    warped = cv2.warpPerspective(board, H, size)
    scene[mask > 0] = warped[mask > 0]
    return cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)


def look_at(cam_pos: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World→camera (R, t) for a camera at cam_pos looking at target, y down."""
    z = target - cam_pos
    z /= np.linalg.norm(z)
    x = np.cross(np.array([0.0, -1.0, 0.0]), z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return R, -R @ cam_pos


# ── Detection ─────────────────────────────────────────────────────────────────

def test_detects_fronto_parallel():
    quad = np.array([[600, 300], [1172, 300], [1172, 704], [600, 704]])   # 20 px/cm
    det = detect_marker(warp_into_scene(board_image(), quad))
    assert det is not None
    assert det["px_per_cm"] == pytest.approx(20.0, rel=0.03)
    pts = np.array(det["grid_pts_full"])
    assert pts[0, 0] < pts[2, 0] and pts[0, 1] < pts[6, 1]   # TL first, row by row


def test_detects_under_perspective():
    quad = np.array([[560, 330], [1220, 290], [1240, 760], [540, 700]])
    det = detect_marker(warp_into_scene(board_image(), quad))
    assert det is not None
    assert len(det["grid_pts_full"]) == 9


def test_upside_down_board_still_labelled_top_left_first():
    quad = np.array([[1172, 704], [600, 704], [600, 300], [1172, 300]])   # 180° rotated
    det = detect_marker(warp_into_scene(board_image(), quad))
    assert det is not None
    pts = np.array(det["grid_pts_full"])
    assert pts[0, 0] < pts[2, 0]


def test_large_image_is_downscaled_but_coords_full_res():
    quad = np.array([[1200, 600], [2344, 600], [2344, 1408], [1200, 1408]])  # 40 px/cm
    det = detect_marker(warp_into_scene(board_image(40), quad, size=(3840, 2160)))
    assert det is not None
    assert det["px_per_cm"] == pytest.approx(40.0, rel=0.03)
    assert np.array(det["grid_pts_full"])[0][0] > 1200


def test_detects_on_dark_speckled_floor():
    """Regression: the flush edge squares used to merge into a dark, noisy floor
    at every threshold (0 detections on a rendered room video)."""
    rng = np.random.default_rng(3)
    floor = cv2.GaussianBlur(rng.normal(95, 30, (1080, 1920)).clip(0, 255).astype(np.uint8), (5, 5), 0)
    board = board_image(10)                                          # ~10 px/cm, small in frame
    quad = np.array([[880, 700], [1150, 690], [1170, 880], [860, 900]])
    h, w = board.shape
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), quad.astype(np.float32))
    mask = cv2.warpPerspective(np.full_like(board, 255), Hm, (1920, 1080))
    scene = floor.copy()
    scene[mask > 0] = cv2.warpPerspective(board, Hm, (1920, 1080))[mask > 0]
    assert detect_marker(cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)) is not None


def test_no_marker_returns_none():
    rng = np.random.default_rng(0)
    img = rng.integers(80, 180, (1080, 1920, 3), dtype=np.uint8)
    assert detect_marker(img) is None


def test_uniform_tile_grid_rejected():
    """Equal-pitch tiles fail the column/row ratio check."""
    img = np.full((1080, 1920), 230, np.uint8)
    for r in range(6):
        for c in range(10):
            x, y = 200 + c * 150, 150 + r * 150
            cv2.rectangle(img, (x, y), (x + 50, y + 50), 30, -1)
    assert detect_marker(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)) is None


# ── Scale from SfM cameras ────────────────────────────────────────────────────

def _synthetic_views(sfm_units_per_m: float, flip_some: bool = False):
    """Board on the z=0 world plane (metres), cameras orbiting; world in SfM units."""
    K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], dtype=np.float64)
    board_m = np.hstack([GRID_CM / 100.0, np.zeros((9, 1))])
    centre = board_m.mean(0)
    cameras_json = {"cameras": [], "images": []}
    dets = {}
    for i, ang in enumerate(np.radians([-30, -10, 10, 30])):
        cam_pos = centre + np.array([0.5 * np.sin(ang), 0.15, -0.6 * np.cos(ang)])
        R, t_m = look_at(cam_pos, centre)
        t = t_m * sfm_units_per_m
        px = []
        for p in board_m:
            c = R @ (p * sfm_units_per_m) + t
            q = K @ c
            px.append(q[:2] / q[2])
        px = np.array(px)
        if flip_some and i % 2:
            px = px[::-1]   # detector labelled this view from the opposite corner
        name = f"frame_{i:04d}.jpg"
        cameras_json["images"].append({
            "name": name, "K": K.tolist(),
            "cam_from_world": {"matrix_3x4": np.hstack([R, t[:, None]]).tolist()},
        })
        dets[name] = {"grid_pts_full": px.tolist(), "px_per_cm": 10.0 + i}
    return dets, cameras_json


@pytest.mark.parametrize("units_per_m", [1.0, 3.7, 0.25])
def test_grid_scale_recovered(units_per_m):
    dets, cams = _synthetic_views(units_per_m)
    scale, diag = derive_grid_scale(dets, cams)
    assert scale == pytest.approx(1.0 / units_per_m, rel=1e-3), diag
    assert diag["fit_rms_m"] < 1e-3


def test_grid_scale_resolves_180_label_flip():
    dets, cams = _synthetic_views(2.0, flip_some=True)
    scale, diag = derive_grid_scale(dets, cams)
    assert scale == pytest.approx(0.5, rel=1e-3), diag
    assert diag["n_flipped"] == 2


def test_grid_scale_needs_two_views():
    dets, cams = _synthetic_views(1.0)
    one = dict(list(dets.items())[:1])
    scale, diag = derive_grid_scale(one, cams)
    assert scale is None and "need" in diag["error"]
