"""
Shared utilities for the HVAC placement stages (wall_plane_detection,
detect_hvac_fixtures, locate_rucklauf, hvac_placement): frame download,
2D<->3D projection, quad sanity checks, and lazy GDINO / SAM2 / SegFormer
model loading.

2D detections are lifted to 3D by projecting the metric dense cloud into the
frame and collecting the points inside the box/mask — no ray triangulation.
Camera poses are loaded with scale_factor=confirmed_scale_factor so t lives in
the same metric frame as the refined cloud (cameras.json itself stays in SfM
units).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from backend.core.config import settings

logger = logging.getLogger(__name__)


# ── Frames ───────────────────────────────────────────────────────────────────
# (Camera poses: depth_fusion_common.load_registered_cameras, shared with the
# densifiers.)

async def download_registered_frames(
    storage, prev_result: dict, cameras: dict[str, dict], frames_dir: Path,
) -> dict[str, Path]:
    """
    Download only the frames COLMAP actually registered (cameras dict keys),
    matched against prev_result's frame/image key list by basename — same
    key_by_name pattern aruco_sfm.py uses.
    """
    image_keys = prev_result.get("image_keys", prev_result.get("frame_keys", []))
    key_by_name = {Path(k).name: k for k in image_keys}
    frames_dir.mkdir(parents=True, exist_ok=True)

    local_by_name: dict[str, Path] = {}
    for name in cameras:
        storage_key = key_by_name.get(name)
        if not storage_key:
            continue
        local = frames_dir / name
        try:
            if not local.exists():
                await storage.download(storage_key, local)
            local_by_name[name] = local
        except Exception as e:
            logger.debug("hvac_common: skip frame %s: %s", name, e)
    return local_by_name


# ── 2D <-> 3D ───────────────────────────────────────────────────────────────

def project_point_to_pixel(
    X_world: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray,
) -> Optional[tuple[float, float]]:
    """
    Project a 3D world point into pixel coordinates: x = K @ (R @ X + t), then
    perspective divide. Returns None if the point is behind the camera.

    Used to render the placement overlay onto a real frame.
    """
    X_cam = R @ X_world + t
    if X_cam[2] <= 1e-6:
        return None
    x = K @ X_cam
    return float(x[0] / x[2]), float(x[1] / x[2])


def wall_basis(normal: np.ndarray, gravity_up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Orthonormal (right, up) basis on a wall plane, with `up` aligned to true
    vertical so (u, v) coordinates map to a meaningful mounting height.
    Shared by wall_plane_detection.py (extent) and hvac_placement.py (grid
    search) so both compute the same basis for a given (normal, gravity_up).
    """
    v_ax = gravity_up - np.dot(gravity_up, normal) * normal
    v_ax /= np.linalg.norm(v_ax) + 1e-9
    u_ax = np.cross(v_ax, normal)
    u_ax /= np.linalg.norm(u_ax) + 1e-9
    return u_ax, v_ax


def project_cloud_to_frame_depth(
    pts_sfm: np.ndarray, cam: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Project every point in `pts_sfm` into one camera frame at once, also
    returning camera-space depth — needed for occlusion testing (is
    something closer to the camera at the same screen position). Same
    formula lingbot_fusion.py's `_fuse()` already uses to calibrate depth
    against COLMAP; shared here since three HVAC stages need it.
    Returns (u, v, z, in_bounds).
    """
    K, R, t = cam["K"], cam["R"], cam["t"]
    pc = pts_sfm @ R.T + t
    z = pc[:, 2]
    front = z > 1e-3
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * pc[:, 0] / np.where(front, z, 1) + cx
    v = fy * pc[:, 1] / np.where(front, z, 1) + cy
    inb = front & (u >= 0) & (u < cam["width"]) & (v >= 0) & (v < cam["height"])
    return u, v, z, inb


def project_cloud_to_frame(pts_sfm: np.ndarray, cam: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like project_cloud_to_frame_depth but without depth — the common case
    for hit-testing (points_in_box/points_in_mask), kept as its own function
    so those callers' signatures don't change."""
    u, v, z, inb = project_cloud_to_frame_depth(pts_sfm, cam)
    return u, v, inb


def points_in_box(pts_sfm: np.ndarray, cam: dict, box: list[float]) -> np.ndarray:
    """Indices into pts_sfm whose projection into `cam` falls inside `box` (x0,y0,x1,y1)."""
    u, v, inb = project_cloud_to_frame(pts_sfm, cam)
    x0, y0, x1, y1 = box
    sel = inb & (u >= x0) & (u < x1) & (v >= y0) & (v < y1)
    return np.where(sel)[0]


def points_in_mask(pts_sfm: np.ndarray, cam: dict, mask: "np.ndarray") -> np.ndarray:
    """Indices into pts_sfm whose projection into `cam` falls inside boolean `mask` (H,W)."""
    u, v, inb = project_cloud_to_frame(pts_sfm, cam)
    u_i = np.clip(u, 0, cam["width"] - 1).astype(int)
    v_i = np.clip(v, 0, cam["height"] - 1).astype(int)
    sel = inb & mask[v_i, u_i]
    return np.where(sel)[0]


MAX_QUAD_DIAGONAL_RATIO = 1.4   # see quad_is_clean — empirically motivated, not a guess.
# 2.0 was tried first and still passed a visibly badly-skewed real render
# (confirmed by actually looking at the overlay, not just trusting the
# number) — tightened after that direct check.


def quad_is_clean(pixel_pts: list[Optional[tuple[float, float]]], width: int, height: int) -> bool:
    """
    Convexity (consistent cross-product winding) + full in-bounds containment
    + diagonal-ratio sanity check for a projected quadrilateral, before
    trusting it as an overlay frame.

    Simpler heuristics (angle-off-normal threshold, frame-centering) fail on
    grazing-angle frames. The diagonal-ratio check matters too: a real
    rectangle's two diagonals are equal and stay close under normal viewing
    angles, while a convex, fully-contained quad from a near-grazing angle can
    still be a heavily foreshortened sliver (diagonals differing 2x+).
    """
    if any(p is None for p in pixel_pts):
        return False
    pts = np.array(pixel_pts)
    if not np.all((pts[:, 0] >= 0) & (pts[:, 0] < width) & (pts[:, 1] >= 0) & (pts[:, 1] < height)):
        return False
    n = len(pts)
    signs = []
    for i in range(n):
        p0, p1, p2 = pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        signs.append(cross > 0)
    if not (all(signs) or not any(signs)):
        return False
    diag1 = float(np.linalg.norm(pts[0] - pts[2]))
    diag2 = float(np.linalg.norm(pts[1] - pts[3]))
    if min(diag1, diag2) < 1e-6:
        return False
    return max(diag1, diag2) / min(diag1, diag2) <= MAX_QUAD_DIAGONAL_RATIO


# ── GDINO (zero-shot detection) ──────────────────────────────────────────────

_gdino_model = None
_gdino_processor = None


def load_gdino():
    """
    Lazy-load Grounding DINO (transformers, already pinned in the main worker
    env — see backend/requirements.txt). Cached at module level so repeated
    calls within one worker process reuse the loaded model.
    """
    global _gdino_model, _gdino_processor
    if _gdino_model is not None:
        return _gdino_model, _gdino_processor

    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("hvac_common: loading GDINO %s on %s", settings.HVAC_GDINO_MODEL_ID, device)
    _gdino_processor = AutoProcessor.from_pretrained(settings.HVAC_GDINO_MODEL_ID)
    _gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        settings.HVAC_GDINO_MODEL_ID
    ).to(device)
    _gdino_model.eval()
    return _gdino_model, _gdino_processor


def detect_with_gdino(
    image_rgb: "np.ndarray", prompts: list[str], box_threshold: float = 0.25,
    text_threshold: float = 0.2,
) -> list[dict]:
    """
    Zero-shot box detection over `prompts` (e.g. ["pipe", "valve", "window"]).
    Returns [{"box": [x0,y0,x1,y1], "label": str, "score": float}, ...] in
    the image's own pixel coordinates.
    """
    import torch
    from PIL import Image

    model, processor = load_gdino()
    device = next(model.parameters()).device
    text = ". ".join(prompts) + "."   # GDINO's expected prompt format
    pil_img = Image.fromarray(image_rgb)

    inputs = processor(images=pil_img, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # transformers renamed this kwarg box_threshold -> threshold at some point
    # after 4.40 (confirmed: the pinned Docker range >=4.40,<5.0 still uses
    # box_threshold; a local venv on transformers 5.x needs threshold=). Try
    # the current name first, fall back to the older one.
    try:
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=[pil_img.size[::-1]],
        )[0]
    except TypeError:
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=[pil_img.size[::-1]],
        )[0]

    # "labels" is deprecated (will return integer ids from transformers>=4.51);
    # "text_labels" is the stable string-label key going forward. Fall back to
    # "labels" only on older transformers where "text_labels" doesn't exist yet.
    labels = results.get("text_labels", results.get("labels"))
    return [
        {"box": box.tolist(), "label": label, "score": float(score)}
        for box, label, score in zip(results["boxes"], labels, results["scores"])
    ]


# ── SAM2 (mask refinement — optional, compat-gated) ─────────────────────────

_sam2_predictor = "unloaded"   # sentinel distinct from None (= "load failed, don't retry")


def load_sam2():
    """
    Lazy-load SAM2, gated by HVAC_ENABLE_SAM2. Returns None (and disables
    itself for the rest of this worker process) if the load fails.
    Callers must treat None as "no mask refinement available" and fall back
    to GDINO box-centroid lifting, never as a hard error.
    """
    global _sam2_predictor
    if _sam2_predictor != "unloaded":
        return _sam2_predictor
    if not settings.HVAC_ENABLE_SAM2:
        _sam2_predictor = None
        return None
    try:
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("hvac_common: loading SAM2 %s on %s", settings.HVAC_SAM2_MODEL_ID, device)
        _sam2_predictor = SAM2ImagePredictor.from_pretrained(settings.HVAC_SAM2_MODEL_ID, device=device)
    except Exception as e:
        logger.warning(
            "hvac_common: SAM2 load failed (%s) — falling back to GDINO box-centroid "
            "lifting without mask refinement for the rest of this run", e,
        )
        _sam2_predictor = None
    return _sam2_predictor


def refine_box_to_mask(image_rgb: "np.ndarray", box: list[float]) -> Optional["np.ndarray"]:
    """Refine a GDINO box into a precise mask via SAM2. Returns None if SAM2
    is unavailable or refinement fails — caller falls back to the box itself."""
    predictor = load_sam2()
    if predictor is None:
        return None
    try:
        predictor.set_image(image_rgb)
        masks, scores, _ = predictor.predict(box=np.array(box), multimask_output=False)
        return masks[0].astype(bool)
    except Exception as e:
        logger.debug("hvac_common: SAM2 refinement failed for box %s: %s", box, e)
        return None


# ── SegFormer/ADE20K (wall-segmentation diagnostic) ─────────────────────────
# ADE20K's standard 150-class palette assigns id 0 = "wall". The wall mask
# selects which cloud points the wall-plane fit sees (wall_plane_detection.py),
# and its whole-frame wall fraction gates candidates in hvac_placement.py.
ADE20K_WALL_CLASS_ID = 0

_segformer_model = None
_segformer_processor = None


def load_segformer():
    """Lazy-load the SegFormer/ADE20K wall-segmentation diagnostic model."""
    global _segformer_model, _segformer_processor
    if _segformer_model is not None:
        return _segformer_model, _segformer_processor

    import torch
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("hvac_common: loading SegFormer %s on %s", settings.HVAC_SEGFORMER_MODEL_ID, device)
    _segformer_processor = AutoImageProcessor.from_pretrained(settings.HVAC_SEGFORMER_MODEL_ID)
    _segformer_model = AutoModelForSemanticSegmentation.from_pretrained(
        settings.HVAC_SEGFORMER_MODEL_ID
    ).to(device)
    _segformer_model.eval()
    return _segformer_model, _segformer_processor


def segformer_wall_mask(image_rgb: "np.ndarray") -> Optional["np.ndarray"]:
    """
    Full-resolution boolean mask of `image_rgb`'s "wall"-classified pixels
    (ADE20K). Returns None on any failure (never raises). Also used to render
    the highlighted-photo overlay for the frontend's Segmentation tab.
    """
    try:
        import torch
        import torch.nn.functional as F
        from PIL import Image

        model, processor = load_segformer()
        device = next(model.parameters()).device
        inputs = processor(images=Image.fromarray(image_rgb), return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits   # (1, num_labels, h, w) at reduced resolution
        upsampled = F.interpolate(
            logits, size=image_rgb.shape[:2], mode="bilinear", align_corners=False,
        )
        pred = upsampled.argmax(dim=1)[0].cpu().numpy()
        return pred == ADE20K_WALL_CLASS_ID
    except Exception as e:
        logger.debug("hvac_common: SegFormer wall-mask failed: %s", e)
        return None
