"""
Custom 3×3 grid marker detection (complements ArUco).

The physical marker is a printed sheet (28.6 × 20.2 cm) carrying 9 black
squares of 3.9 cm side in a 3×3 grid.  Square centres are 12.35 cm apart
horizontally and 8.15 cm vertically.  Everything below follows from those
numbers — the detector compares *ratios*, not sizes, so it works at any
distance.  To use a differently sized print, change only the env vars below.

No ML: OpenCV + numpy only.  A frame without a valid grid returns None —
the detector never guesses.

Pipeline:
  1. _square_blobs  — square-looking dark blobs at 3 adaptive-threshold scales
  2. _fit_grid      — find 9 blobs forming a planar 3×3 grid (homography check)
  3. _darkness_ok   — squares must be clearly darker than the surrounding paper
  4. px_per_cm      — measured at the board centre (robust to tilt)

Environment variables
---------------------
GRID_MARKER_SQUARE_CM    Square side length in cm             (default 3.9)
GRID_MARKER_PITCH_U_CM   Horizontal centre-to-centre distance (default 12.35)
GRID_MARKER_PITCH_V_CM   Vertical centre-to-centre distance   (default 8.15)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

SQUARE_CM  = float(os.environ.get("GRID_MARKER_SQUARE_CM",  "3.9"))
PITCH_U_CM = float(os.environ.get("GRID_MARKER_PITCH_U_CM", "12.35"))
PITCH_V_CM = float(os.environ.get("GRID_MARKER_PITCH_V_CM", "8.15"))

RATIO_U = PITCH_U_CM / SQUARE_CM   # ≈ 3.17 — column gap / square size
RATIO_V = PITCH_V_CM / SQUARE_CM   # ≈ 2.09 — row gap / square size

# cm position of each of the 9 centres, top-left first, row by row.
GRID_CM = np.array(
    [[c * PITCH_U_CM, r * PITCH_V_CM] for r in range(3) for c in range(3)],
    dtype=np.float64,
)
_MAX_DIM          = 1920
_BLOCK_SIZES      = (31, 91, 241)
_MIN_AREA_PX      = 25
_ASPECT_RANGE     = (0.6, 1.67)
_MIN_FILL         = 0.65
_STEP_TOL         = (0.35, 0.45)      # −35 % / +45 % tolerance on neighbour distance
_UV_RATIO_RANGE   = (1.05, 2.2)       # column step / row step (true ≈ 1.515)
_MAX_COS_UV       = 0.45              # |cos| between steps — roughly perpendicular
_SNAP_TOL         = 0.35              # of local step length
_MAX_REPROJ_FRAC  = 0.12              # homography mean error / smaller step
_DARKNESS_RATIO   = 0.75


# ── Step 1: square blobs ──────────────────────────────────────────────────────

def _sheet_quad(hull: np.ndarray) -> Optional[np.ndarray]:
    """
    The sheet's true 4 corners from its notched outline. The corner squares
    notch the paper's corners, so the convex hull is an octagon that slices
    diagonally through them; extending the 4 longest hull sides and
    intersecting neighbours recovers the full quadrilateral.
    """
    poly = cv2.approxPolyDP(hull, 0.01 * cv2.arcLength(hull, True), True).reshape(-1, 2).astype(np.float64)
    n = len(poly)
    if n < 4:
        return None
    lengths = [np.linalg.norm(poly[(i + 1) % n] - poly[i]) for i in range(n)]
    sides = sorted(np.argsort(lengths)[-4:])                 # 4 longest, in outline order
    corners = []
    for j in range(4):
        p, r = poly[sides[j]], poly[(sides[j] + 1) % n] - poly[sides[j]]
        q, s = poly[sides[(j + 1) % 4]], poly[(sides[(j + 1) % 4] + 1) % n] - poly[sides[(j + 1) % 4]]
        denom = r[0] * s[1] - r[1] * s[0]
        if abs(denom) < 1e-9:
            return None
        t = ((q[0] - p[0]) * s[1] - (q[1] - p[1]) * s[0]) / denom
        corners.append(p + t * r)
    quad = np.array(corners)
    quad_area = cv2.contourArea(quad.astype(np.float32))
    if not cv2.isContourConvex(quad.astype(np.int32)) or \
            not 1.0 <= quad_area / cv2.contourArea(hull) <= 1.5:
        return None
    return np.round(quad).astype(np.int32)


def _paper_masked_binaries(gray: np.ndarray) -> list[np.ndarray]:
    """
    Paper-first pass. On a dark or busy background (carpet, wood) the flush
    outer squares merge with the floor at every threshold. Find bright regions
    shaped like the sheet — a rectangle whose outline is notched by its edge
    squares — white out everything outside each one's convex hull, and
    threshold inside it, so the edge squares are bounded by white.
    """
    h, w = gray.shape[:2]
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 9 * _MIN_AREA_PX or area > 0.5 * h * w:
            continue
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        # 8 notches remove ~20 % of the sheet → hull/area ≈ 1.27 (true layout)
        if not 1.05 <= hull_area / area <= 1.8:
            continue
        quad = _sheet_quad(hull)   # also rejects shapes that aren't a clean quadrilateral
        if quad is None:
            continue
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, quad, 255)
        # Shrink past the blurred paper border — otherwise that thin dark band
        # links every edge square into one ring.
        k = max(2, int(round(0.03 * np.sqrt(hull_area))))
        mask = cv2.erode(mask, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
        inside = gray[mask > 0]
        if inside.size < 9 * _MIN_AREA_PX:
            continue
        t, _ = cv2.threshold(inside.reshape(1, -1), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        out.append(((gray < t) & (mask > 0)).astype(np.uint8) * 255)
    return out


def _square_blobs(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find solid, square-ish dark blobs.  Returns (centres (N,2), sizes (N,))
    in the coordinates of `gray`.  Size = sqrt(filled area).
    """
    h, w = gray.shape[:2]
    max_area = (h * w) / 9.0
    found: list[tuple[float, float, float]] = []   # (x, y, size)

    binaries = [
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 10)
        for block in _BLOCK_SIZES if block < min(h, w)
    ]
    # The outer squares are flush with the sheet edges, so against a mid-tone
    # background adaptive thresholding merges them into the background.  Two
    # global Otsu passes (all pixels, then only the darker class) separate
    # printed black from whatever the sheet is lying on.
    t1, b1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    binaries.append(b1)
    dark = gray[gray < t1]
    if dark.size > 100:
        t2, _ = cv2.threshold(dark.reshape(1, -1), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        binaries.append(cv2.threshold(gray, t2, 255, cv2.THRESH_BINARY_INV)[1])

    binaries.extend(_paper_masked_binaries(gray))

    for binary in binaries:
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        hierarchy = hierarchy[0]

        # Hole area per outer contour so rings don't count as solid squares
        hole_area: dict[int, float] = {}
        for idx, (_, _, _, parent) in enumerate(hierarchy):
            if parent >= 0:
                hole_area[parent] = hole_area.get(parent, 0.0) + cv2.contourArea(contours[idx])

        for idx, cnt in enumerate(contours):
            if hierarchy[idx][3] >= 0:
                continue
            area = cv2.contourArea(cnt) - hole_area.get(idx, 0.0)
            if area < _MIN_AREA_PX or area > max_area:
                continue
            (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
            if rw <= 0 or rh <= 0:
                continue
            aspect = rw / rh
            if not (_ASPECT_RANGE[0] <= aspect <= _ASPECT_RANGE[1]):
                continue
            if area / (rw * rh) < _MIN_FILL:
                continue
            m = cv2.moments(cnt)
            if m["m00"] <= 0:
                continue
            found.append((m["m10"] / m["m00"], m["m01"] / m["m00"], float(np.sqrt(area))))

    if not found:
        return np.zeros((0, 2)), np.zeros(0)

    # Merge the same square found at several threshold sizes, keeping the largest
    found.sort(key=lambda b: -b[2])
    kept: list[tuple[float, float, float]] = []
    for x, y, s in found:
        if any((x - kx) ** 2 + (y - ky) ** 2 < (0.5 * ks) ** 2 for kx, ky, ks in kept):
            continue
        kept.append((x, y, s))

    arr = np.array(kept, dtype=np.float64)
    return arr[:, :2], arr[:, 2]


# ── Step 2: 3×3 grid fit ──────────────────────────────────────────────────────

def _snap(target: np.ndarray, centres: np.ndarray, sizes: np.ndarray, ref_size: float,
          tol: float, used: set[int]) -> Optional[int]:
    d = np.linalg.norm(centres - target, axis=1)
    for j in np.argsort(d):
        if d[j] > tol:
            return None
        if j in used:
            continue
        if 0.5 <= sizes[j] / ref_size <= 2.0:
            return int(j)
    return None


def _try_grid(a: int, u: int, v: int, centres: np.ndarray, sizes: np.ndarray) -> Optional[list[int]]:
    """Grow a 3×3 grid from anchor a, column neighbour u, row neighbour v."""
    s = sizes[a]
    grid = [[-1] * 3 for _ in range(3)]
    grid[0][0], grid[0][1], grid[1][0] = a, u, v
    used = {a, u, v}
    P = centres

    def put(r: int, c: int, pred: np.ndarray, step_len: float) -> bool:
        j = _snap(pred, centres, sizes, s, _SNAP_TOL * step_len, used)
        if j is None:
            return False
        grid[r][c] = j
        used.add(j)
        return True

    # Row starts (column 0): extrapolate using the local row step
    row_step = P[v] - P[a]
    if not put(2, 0, P[v] + row_step, np.linalg.norm(row_step)):
        return None
    # Fill each row left-to-right, using the column step from the row above
    for r in range(3):
        if r == 0:
            col_step = P[u] - P[a]
        else:
            col_step = P[grid[r - 1][1]] - P[grid[r - 1][0]]
            if not put(r, 1, P[grid[r][0]] + col_step, np.linalg.norm(col_step)):
                return None
        col_step = P[grid[r][1]] - P[grid[r][0]]
        if not put(r, 2, P[grid[r][1]] + col_step, np.linalg.norm(col_step)):
            return None
    return [grid[r][c] for r in range(3) for c in range(3)]


def _fit_grid(centres: np.ndarray, sizes: np.ndarray) -> Optional[dict]:
    """
    Try each blob as top-left anchor; return the largest-looking valid grid:
        {"pts": (9,2) px, "sizes": (9,), "H_cm2img": 3x3, "area": float}
    """
    n = len(centres)
    if n < 9:
        return None

    tree = cKDTree(centres)

    best: Optional[dict] = None
    seen: set[frozenset] = set()

    for a in range(n):
        s = sizes[a]
        r_max = RATIO_U * s * (1 + _STEP_TOL[1])
        nbrs = [j for j in tree.query_ball_point(centres[a], r_max) if j != a]
        if len(nbrs) < 2:
            continue

        col_c, row_c = [], []
        for j in nbrs:
            if not (0.6 <= sizes[j] / s <= 1.67):
                continue
            d = np.linalg.norm(centres[j] - centres[a])
            if RATIO_U * s * (1 - _STEP_TOL[0]) <= d <= RATIO_U * s * (1 + _STEP_TOL[1]):
                col_c.append(j)
            if RATIO_V * s * (1 - _STEP_TOL[0]) <= d <= RATIO_V * s * (1 + _STEP_TOL[1]):
                row_c.append(j)

        for u in col_c:
            cu = centres[u] - centres[a]
            lu = np.linalg.norm(cu)
            for v in row_c:
                if v == u:
                    continue
                cv_ = centres[v] - centres[a]
                lv = np.linalg.norm(cv_)
                if abs(np.dot(cu, cv_)) / (lu * lv) > _MAX_COS_UV:
                    continue
                if not (_UV_RATIO_RANGE[0] <= lu / lv <= _UV_RATIO_RANGE[1]):
                    continue
                # Reject mirrored labelings: with image y down, v must be
                # clockwise from u (a printed board can't appear mirrored).
                if cu[0] * cv_[1] - cu[1] * cv_[0] <= 0:
                    continue

                idx = _try_grid(a, u, v, centres, sizes)
                if idx is None:
                    continue
                key = frozenset(idx)
                if key in seen:
                    continue
                seen.add(key)
                # TL and BR anchors both yield valid labelings (180° apart) —
                # canonicalise so the column step points rightward in the image.
                if centres[idx[2], 0] - centres[idx[0], 0] < 0:
                    idx = idx[::-1]
                pts = centres[idx]

                H, _ = cv2.findHomography(GRID_CM, pts, 0)
                if H is None:
                    continue
                proj = cv2.perspectiveTransform(GRID_CM.reshape(-1, 1, 2), H).reshape(-1, 2)
                err = float(np.mean(np.linalg.norm(proj - pts, axis=1)))
                if err > _MAX_REPROJ_FRAC * min(lu, lv):
                    continue

                area = float(cv2.contourArea(pts[[0, 2, 8, 6]].astype(np.float32)))
                if best is None or area > best["area"]:
                    best = {"pts": pts, "sizes": sizes[idx], "H_cm2img": H, "area": area}
    return best


# ── Step 3: darkness check ────────────────────────────────────────────────────

def _patch_mean(gray: np.ndarray, pt: np.ndarray, r: float) -> Optional[float]:
    h, w = gray.shape[:2]
    r = max(1, int(round(r)))
    x, y = int(round(pt[0])), int(round(pt[1]))
    x0, x1, y0, y1 = max(0, x - r), min(w, x + r + 1), max(0, y - r), min(h, y + r + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return float(gray[y0:y1, x0:x1].mean())


def _darkness_ok(gray: np.ndarray, grid: dict) -> bool:
    """Squares' mean brightness must be < 75 % of the paper between them."""
    pts, sizes, H = grid["pts"], grid["sizes"], grid["H_cm2img"]
    sq = [_patch_mean(gray, p, 0.25 * s) for p, s in zip(pts, sizes)]

    # Paper samples: midway between horizontally / vertically adjacent squares
    paper_cm = [(c * PITCH_U_CM + PITCH_U_CM / 2, r * PITCH_V_CM) for r in range(3) for c in range(2)]
    paper_cm += [(c * PITCH_U_CM, r * PITCH_V_CM + PITCH_V_CM / 2) for r in range(2) for c in range(3)]
    paper_px = cv2.perspectiveTransform(
        np.array(paper_cm, dtype=np.float64).reshape(-1, 1, 2), H,
    ).reshape(-1, 2)
    paper = [_patch_mean(gray, p, 0.15 * float(np.median(sizes))) for p in paper_px]

    sq = [v for v in sq if v is not None]
    paper = [v for v in paper if v is not None]
    if len(sq) < 9 or len(paper) < 4:
        return False
    paper_mean = float(np.mean(paper))
    return paper_mean > 0 and float(np.mean(sq)) < _DARKNESS_RATIO * paper_mean


# ── Public API ────────────────────────────────────────────────────────────────

def detect_marker(img: np.ndarray) -> Optional[dict]:
    """
    Detect the 3×3 grid marker in a BGR or grayscale image.

    Returns None when no valid grid is found, else:
        {
          "px_per_cm":     float,
          "grid_pts_full": [[x,y]×9]  — detected centres, TL first, row by row
        }
    Pixel values are in the coordinates of the input image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    scale = min(1.0, _MAX_DIM / max(h, w))
    small = cv2.resize(gray, (int(round(w * scale)), int(round(h * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else gray

    centres, sizes = _square_blobs(small)
    grid = _fit_grid(centres, sizes)
    if grid is None or not _darkness_ok(small, grid):
        return None

    pts = grid["pts"] / scale
    H_cm2img, _ = cv2.findHomography(GRID_CM, pts, 0)
    if H_cm2img is None:
        return None

    # px per cm at the board centre (centre square), horizontally and vertically
    c = GRID_CM[4]
    probe = np.array([c + [-0.5, 0], c + [0.5, 0], c + [0, -0.5], c + [0, 0.5]])
    pp = cv2.perspectiveTransform(probe.reshape(-1, 1, 2), H_cm2img).reshape(-1, 2)
    px_per_cm = 0.5 * (np.linalg.norm(pp[1] - pp[0]) + np.linalg.norm(pp[3] - pp[2]))

    return {"px_per_cm": float(px_per_cm), "grid_pts_full": pts.tolist()}
