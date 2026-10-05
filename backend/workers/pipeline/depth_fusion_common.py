"""
Shared helpers for the optional monocular-depth-fusion stages
(lingbot_fusion.py, metricanything_fusion.py): both calibrate each frame's
depth map against COLMAP's own dense cloud before trusting it, and both need
per-frame camera poses from cameras.json.
"""

from __future__ import annotations

import numpy as np


def robust_affine(dl: np.ndarray, zc: np.ndarray, iters: int = 4) -> tuple[float, float, float]:
    """Fit zc ~ a*dl + b robustly (trimmed least squares). Returns (a, b, inlier_frac)."""
    m = np.ones(len(dl), bool)
    a, b = 1.0, 0.0
    for _ in range(iters):
        if m.sum() < 30:
            break
        A = np.vstack([dl[m], np.ones(m.sum())]).T
        (a, b), *_ = np.linalg.lstsq(A, zc[m], rcond=None)
        resid = np.abs(a * dl + b - zc)
        thr = 2.5 * np.median(resid[m]) + 1e-9
        m = resid < thr
    return a, b, float(m.mean())


def load_registered_cameras(cameras_json: dict, scale_factor: float = 1.0) -> dict[str, dict]:
    """
    Parse cameras.json (as sfm.py writes it) into per-frame intrinsics and
    world→camera pose. cameras.json is in SfM units; pass scale_factor
    (confirmed_scale_factor) to scale t so R/t line up with a metric cloud —
    R is scale-invariant, and X_cam = R·(sX) + s·t = s·(R·X + t) keeps the
    projection unchanged. Leave it at 1.0 to project against SfM-unit points.

    Returns {filename: {"K": 3x3, "R": 3x3, "t": (3,), "width": int, "height": int}}.
    Uses the per-image K when present, otherwise the image's COLMAP camera
    (SIMPLE_RADIAL / SIMPLE_PINHOLE / RADIAL / PINHOLE params).
    """
    colmap_cams = {c.get("camera_id"): c for c in cameras_json.get("cameras", [])}
    cameras: dict[str, dict] = {}

    for img in cameras_json.get("images", []):
        cfw = img.get("cam_from_world", {})
        mat = cfw.get("matrix_3x4") if isinstance(cfw, dict) else cfw
        if mat is None:
            continue
        mat_arr = np.array(mat, dtype=np.float64).reshape(3, 4)

        K = img.get("K")
        if K is not None:
            K_arr = np.array(K, dtype=np.float64).reshape(3, 3)
            w = int(img.get("width", 1920))
            h = int(img.get("height", 1080))
        else:
            cam = colmap_cams.get(img.get("camera_id"))
            if cam is None:
                continue
            w = int(cam.get("width", 1920))
            h = int(cam.get("height", 1080))
            params = cam.get("params", [])
            model = cam.get("model", "SIMPLE_RADIAL")
            if model in ("SIMPLE_RADIAL", "SIMPLE_PINHOLE") and len(params) >= 3:
                fl, cx, cy = float(params[0]), float(params[1]), float(params[2])
            elif model in ("RADIAL", "PINHOLE") and len(params) >= 4:
                fl, cx, cy = float(params[0]), float(params[2]), float(params[3])
            elif len(params) >= 1:
                fl, cx, cy = float(params[0]), w / 2.0, h / 2.0
            else:
                continue
            K_arr = np.array([[fl, 0, cx], [0, fl, cy], [0, 0, 1]], dtype=np.float64)

        cameras[img.get("name", "")] = {
            "K": K_arr, "R": mat_arr[:, :3], "t": mat_arr[:, 3] * scale_factor,
            "width": w, "height": h,
        }
    return cameras
