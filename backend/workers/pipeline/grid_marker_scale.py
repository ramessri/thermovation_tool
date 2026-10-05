"""
Post-SfM metric scale from the 3×3 grid marker.

The board's geometry is known exactly (GRID_CM), so unlike ArUco no solvePnP
depth baseline is needed:
  1. For every SfM-registered frame with a grid detection, back-project each
     of the 9 detected square centres into a world-space ray.
  2. Resolve the 180° labelling ambiguity across views (TL↔BR) by aligning
     each view's board x-axis (from a per-view solvePnP) with a reference view.
  3. Triangulate each of the 9 centres from all views.
  4. Fit a similarity transform (Umeyama) from the board's metric layout to
     the triangulated points: scale = 1 / s  (metres per SfM unit).

The fit residual guards against bad triangulation — the result is rejected
if the 9 triangulated points don't form the known flat grid.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from backend.workers.pipeline.grid_marker_detector import GRID_CM, PITCH_V_CM
from backend.workers.pipeline.scale_from_aruco import _load_cameras, _triangulate_marker

logger = logging.getLogger(__name__)

GRID_MIN_VIEWS        = 2
_MIN_PARALLAX_DEG     = 1.0
_MAX_RESIDUAL_FRAC    = 0.10   # fit RMS / smaller pitch

_GRID_M_3D = np.hstack([GRID_CM / 100.0, np.zeros((9, 1))])


def _pixel_ray(px: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray):
    ray_cam = np.array([(px[0] - K[0, 2]) / K[0, 0], (px[1] - K[1, 2]) / K[1, 1], 1.0])
    ray_world = R.T @ (ray_cam / np.linalg.norm(ray_cam))
    return -R.T @ t, ray_world / np.linalg.norm(ray_world)


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity dst ≈ s·R·src + t.  Returns (s, R, t)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s)
    return s, R, mu_d - s * R @ mu_s


def derive_grid_scale(
    grid_detections: dict[str, dict],
    cameras_json: dict,
) -> tuple[Optional[float], dict]:
    """
    grid_detections: {image_filename: detect_marker() output (serialised)}.
    Returns (metres_per_sfm_unit | None, diagnostics).
    """
    cameras = _load_cameras(cameras_json)
    diag: dict = {"n_detections": len(grid_detections)}

    views = []   # (pts (9,2), cam, x_axis_world, px_per_cm)
    for fname, det in grid_detections.items():
        cam = cameras.get(fname)
        if cam is None:
            continue
        pts = np.array(det["grid_pts_full"], dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(_GRID_M_3D, pts, cam["K"], np.zeros(4), flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            continue
        R_board, _ = cv2.Rodrigues(rvec)
        views.append((pts, cam, cam["R"].T @ R_board[:, 0], float(det.get("px_per_cm", 0))))

    diag["n_views"] = len(views)
    if len(views) < GRID_MIN_VIEWS:
        diag["error"] = f"grid marker in {len(views)} registered view(s); need ≥{GRID_MIN_VIEWS}"
        return None, diag

    # Canonicalise labels against the closest (largest) view
    ref_axis = max(views, key=lambda v: v[3])[2]
    obs: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in range(9)]
    n_flipped = 0
    for pts, cam, x_axis, _ in views:
        if np.dot(x_axis, ref_axis) < 0:
            pts = pts[::-1]
            n_flipped += 1
        for i in range(9):
            obs[i].append(_pixel_ray(pts[i], cam["K"], cam["R"], cam["t"]))
    diag["n_flipped"] = n_flipped

    # Parallax on the centre square — near-identical rays can't triangulate
    dirs = np.array([d for _, d in obs[4]])
    max_angle = float(np.degrees(np.arccos(np.clip((dirs @ dirs.T).min(), -1.0, 1.0))))
    diag["max_parallax_deg"] = round(max_angle, 2)
    if max_angle < _MIN_PARALLAX_DEG:
        diag["error"] = f"grid marker views have {max_angle:.2f}° parallax (< {_MIN_PARALLAX_DEG}°)"
        return None, diag

    tri = [_triangulate_marker(o) for o in obs]
    if any(p is None for p in tri):
        diag["error"] = "grid triangulation failed"
        return None, diag
    tri_arr = np.array(tri)

    s, R, t = _umeyama(_GRID_M_3D, tri_arr)
    if s <= 1e-12:
        diag["error"] = "degenerate grid fit"
        return None, diag
    scale = 1.0 / s
    residual_m = float(np.sqrt(np.mean(np.sum((tri_arr - (s * (R @ _GRID_M_3D.T).T + t)) ** 2, 1)))) * scale
    diag["fit_rms_m"] = round(residual_m, 5)
    if residual_m > _MAX_RESIDUAL_FRAC * PITCH_V_CM / 100.0:
        diag["error"] = (f"triangulated grid doesn't match the board layout "
                         f"(rms {residual_m * 100:.2f} cm)")
        return None, diag

    diag["scale_factor"] = round(scale, 6)
    logger.info("grid_marker_scale: scale=%.6f m/unit from %d views (rms %.2f cm, parallax %.1f°)",
                scale, len(views), residual_m * 100, max_angle)
    return scale, diag

