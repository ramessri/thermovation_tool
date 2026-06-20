"""
Stage 2: Extract frames from one or more uploaded media files.

Strategy: step-based extraction with burst oversampling.
  - Videos are sampled at OVERSAMPLE_FACTOR × target FPS.
  - Consecutive decoded frames are grouped into bursts of OVERSAMPLE_FACTOR.
  - The sharpest frame (highest Laplacian variance) from each burst is kept.
  - No frames are rejected globally — every time slot contributes its best
    available frame, even if the camera was moving during that section.
  - Groups whose best frame falls below BLURRY_WARN_THRESHOLD are recorded in
    the result as `blurry_sections` so the caller can surface a re-shoot hint.
  - Still images (jpg, png, heic, …) are included as-is — no extraction needed.

Device rotation from video_metadata is applied to every decoded frame and still.
"""
import asyncio
import logging
from pathlib import Path
from typing import Callable, Optional
import cv2
import numpy as np

from backend.core.config import settings
from backend.core.storage import get_storage

logger = logging.getLogger(__name__)

# Rotation map: degrees → OpenCV rotation code
_CV2_ROTATE = {
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

# ── Oversampling / quality-selection knobs ────────────────────────────────────
OVERSAMPLE_FACTOR     = 5      # evaluate N candidates per target slot; best is kept
BURST_INTERVAL_S      = 0.1    # seconds between burst frames
QUALITY_SHARP_WEIGHT  = 0.6    # weight for sharpness in combined quality score
QUALITY_MOTION_WEIGHT = 0.4    # weight for motion stability (lower motion = better)
BLURRY_WARN_THRESHOLD = 60.0   # Laplacian variance below this → flag as blurry

# ── Resize presets (max long-edge in px) ──────────────────────────────────────
# "native" keeps the source resolution. Frames are only ever downscaled, never
# upscaled — a preset larger than the source resolution is a no-op.
RESIZE_PRESETS: dict[str, int | None] = {
    "native": None,
    "2k":     2560,
    "4k":     3840,
    "8k":     7680,
}


def _resize_frame(frame: np.ndarray, max_long_edge: int | None) -> np.ndarray:
    """Downscale `frame` so its longer edge is at most `max_long_edge`, preserving
    aspect ratio. No-op if already smaller, or if `max_long_edge` is None."""
    if max_long_edge is None:
        return frame
    h, w = frame.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return frame
    scale = max_long_edge / long_edge
    new_w, new_h = round(w * scale), round(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

# ── Media type detection ──────────────────────────────────────────────────────
_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.tif', '.tiff', '.bmp', '.webp'}
_VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.mts', '.m2ts', '.3gp', '.mp'}


def _is_image_key(key: str) -> bool:
    return Path(key).suffix.lower() in _IMAGE_EXTS


def _is_video_key(key: str) -> bool:
    return Path(key).suffix.lower() in _VIDEO_EXTS or not _is_image_key(key)


# ── Quality helpers ───────────────────────────────────────────────────────────

def _sharpness(frame_bgr: np.ndarray) -> float:
    """Laplacian variance — higher = sharper.  Fast: single-channel, quarter-res."""
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2))
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


def _frame_diff(a: np.ndarray, b: np.ndarray) -> float:
    """
    Mean absolute pixel difference at 1/4 resolution — lower = less camera motion.
    Used as a proxy for motion blur within a burst.
    """
    sa = cv2.resize(a, (a.shape[1] // 4, a.shape[0] // 4))
    sb = cv2.resize(b, (b.shape[1] // 4, b.shape[0] // 4))
    return float(np.mean(np.abs(sa.astype(np.float32) - sb.astype(np.float32))))


def _best_in_group(group: list[tuple[np.ndarray, float]]) -> int:
    """
    Given [(frame_bgr, sharpness), ...] for a burst of OVERSAMPLE_FACTOR frames,
    return the index of the frame with the best combined quality score.

    Score = QUALITY_SHARP_WEIGHT  * norm(sharpness)
          + QUALITY_MOTION_WEIGHT * norm(motion_stability)   [inverted: lower diff = better]

    Both metrics are normalised to [0, 1] within the group before combining.
    """
    n = len(group)
    if n == 1:
        return 0

    frames      = [f for f, _ in group]
    sharpnesses = np.array([s for _, s in group], dtype=np.float64)

    # Motion: average absolute frame diff to adjacent frames in the burst
    motion = np.zeros(n, dtype=np.float64)
    for i, f in enumerate(frames):
        diffs: list[float] = []
        if i > 0:
            diffs.append(_frame_diff(f, frames[i - 1]))
        if i < n - 1:
            diffs.append(_frame_diff(f, frames[i + 1]))
        motion[i] = float(np.mean(diffs)) if diffs else 0.0

    def _norm01(arr: np.ndarray, invert: bool = False) -> np.ndarray:
        rng = arr.max() - arr.min()
        if rng < 1e-8:
            return np.ones(len(arr), dtype=np.float64)
        normed = (arr - arr.min()) / rng
        return (1.0 - normed) if invert else normed

    quality = (QUALITY_SHARP_WEIGHT  * _norm01(sharpnesses) +
               QUALITY_MOTION_WEIGHT * _norm01(motion, invert=True))
    return int(np.argmax(quality))


# ── Per-source extractors ─────────────────────────────────────────────────────

def _extract_video_candidates(
    project_id: str,
    video_path: Path,
    frames_dir: Path,
    cand_offset: int,
    rotation_deg: int,
    max_frames: int,
    progress_cb: Callable[[float, str], None],
    progress_range: tuple[float, float] = (0.0, 1.0),
    max_long_edge: int | None = None,
) -> tuple[list[tuple[Path, float]], list[float]]:
    """Extract best-of-burst frame candidates from a single video file."""
    rotate_code = _CV2_ROTATE.get(rotation_deg)
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target_fps   = settings.FRAME_EXTRACTION_FPS
    step         = max(1, int(source_fps / (target_fps * OVERSAMPLE_FACTOR)))

    logger.info(
        "[%s] video %s: source_fps=%.1f step=%d oversample=%d",
        project_id, video_path.name, source_fps, step, OVERSAMPLE_FACTOR,
    )

    p0, p1 = progress_range
    candidates:    list[tuple[Path, float]] = []
    blurry_groups: list[float]              = []
    frame_idx     = 0
    current_group: list[tuple[np.ndarray, float]] = []

    def _flush_group():
        best_i = _best_in_group(current_group)
        best_frame, best_score = current_group[best_i]
        idx   = cand_offset + len(candidates)
        fpath = frames_dir / f"cand_{idx:06d}.jpg"
        cv2.imwrite(str(fpath), best_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        candidates.append((fpath, best_score))
        if best_score < BLURRY_WARN_THRESHOLD:
            t = frame_idx / max(source_fps, 1)
            blurry_groups.append(round(t, 1))

    while True:
        ret, raw = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frame = cv2.rotate(raw, rotate_code) if rotate_code is not None else raw
            frame = _resize_frame(frame, max_long_edge)
            current_group.append((frame.copy(), _sharpness(frame)))
            if len(current_group) >= OVERSAMPLE_FACTOR:
                _flush_group()
                current_group = []
                if len(candidates) % 10 == 0:
                    frac = p0 + (p1 - p0) * (frame_idx / max(total_frames, 1))
                    progress_cb(frac, f"Video {video_path.name}: {len(candidates)} frames…")
                if len(candidates) >= max_frames:
                    break
        frame_idx += 1

    if current_group and len(candidates) < max_frames:
        _flush_group()

    cap.release()
    return candidates, blurry_groups


def _extract_still_candidates(
    still_paths: list[Path],
    frames_dir: Path,
    cand_offset: int,
    rotation_deg: int,
    max_long_edge: int | None = None,
) -> tuple[list[tuple[Path, float]], int]:
    """
    Load still images as frame candidates.

    All readable, sharp-enough stills are included — no budget cap applied.
    Stills below BLURRY_WARN_THRESHOLD are skipped with a warning (not silently
    discarded). Returns (candidates, n_blurry_skipped).
    """
    rotate_code = _CV2_ROTATE.get(rotation_deg)
    candidates:    list[tuple[Path, float]] = []
    n_unreadable   = 0
    n_blurry       = 0

    for src in still_paths:
        img = cv2.imread(str(src))
        if img is None:
            logger.warning("Still image unreadable — skipped: %s", src.name)
            n_unreadable += 1
            continue
        if rotate_code is not None:
            img = cv2.rotate(img, rotate_code)
        img = _resize_frame(img, max_long_edge)
        score = _sharpness(img)
        if score < BLURRY_WARN_THRESHOLD:
            logger.warning(
                "Still image too blurry (sharpness=%.1f < %.1f) — skipped: %s",
                score, BLURRY_WARN_THRESHOLD, src.name,
            )
            n_blurry += 1
            continue
        idx   = cand_offset + len(candidates)
        fpath = frames_dir / f"cand_{idx:06d}.jpg"
        cv2.imwrite(str(fpath), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        candidates.append((fpath, score))

    if n_unreadable or n_blurry:
        logger.warning(
            "Stills rejected: %d unreadable, %d blurry (of %d total)",
            n_unreadable, n_blurry, len(still_paths),
        )
    return candidates, n_blurry


# ── Main entrypoint ───────────────────────────────────────────────────────────

async def run_extract_frames(
    project_id: str,
    storage_key: str,
    tmp: Path,
    job_id: str,
    progress_cb: Callable[[float, str], None],
    video_metadata: Optional[dict] = None,
    max_frames_override: Optional[int] = None,
    all_storage_keys: Optional[list[str]] = None,
    resize_preset: Optional[str] = None,
) -> dict:
    """
    Extract frames from one or more source files.

    all_storage_keys: if provided, process all keys (videos + stills).
                      Falls back to [storage_key] for single-source projects.
    resize_preset: one of RESIZE_PRESETS ("native", "2k", "4k", "8k"). Frames are
                    downscaled (never upscaled) to this max long-edge before saving.
    """
    storage = get_storage()

    max_long_edge = RESIZE_PRESETS.get(resize_preset or "native", None)

    sources = all_storage_keys if all_storage_keys else [storage_key]
    video_keys = [k for k in sources if not _is_image_key(k)]
    still_keys = [k for k in sources if _is_image_key(k)]

    rotation_deg = (video_metadata or {}).get("rotation_deg", 0)
    frames_dir   = tmp / "frames"
    frames_dir.mkdir(exist_ok=True)
    stills_dir   = tmp / "stills"
    stills_dir.mkdir(exist_ok=True)

    logger.info(
        "[%s] extract_frames: %d video(s), %d still(s)",
        project_id, len(video_keys), len(still_keys),
    )

    # Each video gets the full per-video cap. Short nook clips naturally
    # contribute fewer frames (bounded by their duration), so no division needed.
    # Dividing the budget would shortchange clips that have dense but limited footage.
    max_frames_per_video = max_frames_override or settings.MAX_FRAMES_PER_VIDEO

    all_candidates:         list[tuple[Path, float]] = []
    all_blurry_groups:      list[float]              = []
    frame_source_boundaries: list[dict]              = []
    n_blurry_stills         = 0

    # ── Process still images first ────────────────────────────────────────────
    if still_keys:
        progress_cb(0.02, f"Downloading {len(still_keys)} still image(s)…")
        still_paths: list[Path] = []
        for i, key in enumerate(still_keys):
            ext   = Path(key).suffix or ".jpg"
            dest  = stills_dir / f"still_{i:04d}{ext}"
            await storage.download(key, dest)
            still_paths.append(dest)

        progress_cb(0.08, f"Processing {len(still_keys)} stills…")
        still_start = len(all_candidates)
        still_cands, n_blurry_stills = _extract_still_candidates(
            still_paths, frames_dir, still_start, rotation_deg, max_long_edge=max_long_edge
        )
        all_candidates.extend(still_cands)
        if still_cands:
            frame_source_boundaries.append({
                "source": "stills",
                "start":  still_start,
                "end":    len(all_candidates),
            })
        logger.info(
            "[%s] stills: %d accepted, %d blurry-skipped (of %d)",
            project_id, len(still_cands), n_blurry_stills, len(still_keys),
        )

    # ── Process videos ────────────────────────────────────────────────────────
    n_videos = len(video_keys)
    for vi, key in enumerate(video_keys):
        p_base  = 0.10 + 0.55 * (vi / max(n_videos, 1))
        p_end   = 0.10 + 0.55 * ((vi + 1) / max(n_videos, 1))

        progress_cb(p_base, f"Downloading video {vi+1}/{n_videos}…")
        video_path = tmp / f"video_{vi:02d}{Path(key).suffix or '.mp4'}"
        await storage.download(key, video_path)

        src_start = len(all_candidates)
        cands, blurry = _extract_video_candidates(
            project_id, video_path, frames_dir,
            cand_offset=src_start,
            rotation_deg=rotation_deg,
            max_frames=max_frames_per_video,
            progress_cb=progress_cb,
            progress_range=(p_base + 0.02, p_end),
            max_long_edge=max_long_edge,
        )
        all_candidates.extend(cands)
        all_blurry_groups.extend(blurry)
        if cands:
            frame_source_boundaries.append({
                "source": key,
                "start":  src_start,
                "end":    len(all_candidates),
            })
        logger.info("[%s] video %d/%d: %d frames extracted", project_id, vi + 1, n_videos, len(cands))

        # Free disk space between videos
        video_path.unlink(missing_ok=True)

    if not all_candidates:
        progress_cb(1.0, "No frames extracted — check source files")
        return {"frame_keys": [], "frame_count": 0, "rotation_deg": rotation_deg,
                "blurry_dropped": 0, "blurry_stills_skipped": n_blurry_stills}

    if all_blurry_groups:
        logger.warning(
            "[%s] extract_frames: %d blurry video groups across all sources",
            project_id, len(all_blurry_groups),
        )

    n_still_cands = sum(
        b["end"] - b["start"] for b in frame_source_boundaries if b["source"] == "stills"
    )
    n_video_cands = len(all_candidates) - n_still_cands
    logger.info(
        "[%s] total candidates: %d (%d from stills, %d from %d video(s))",
        project_id, len(all_candidates), n_still_cands, n_video_cands, n_videos,
    )
    progress_cb(0.68, f"Extracted {len(all_candidates)} total frames — uploading…")

    # ── Upload all candidates with sequential frame numbering ─────────────────
    frame_keys: list[str] = []
    n_keep = len(all_candidates)
    for i, (fpath, _) in enumerate(all_candidates):
        dest_key = f"{project_id}/frames/frame_{i:06d}.jpg"
        await storage.upload(fpath, dest_key)
        frame_keys.append(dest_key)
        if i % 20 == 0 or i == n_keep - 1:
            progress_cb(0.68 + 0.30 * (i + 1) / n_keep,
                        f"Uploading {i+1}/{n_keep} frames…")

    rot_note = f" (rotation corrected {rotation_deg}°)" if _CV2_ROTATE.get(rotation_deg) else ""
    progress_cb(1.0, f"Done — {len(frame_keys)} frames{rot_note} "
                     f"({len(still_keys)} stills + {n_videos} video(s))")

    return {
        "frame_keys":              frame_keys,
        "frame_count":             len(frame_keys),
        "rotation_deg":            rotation_deg,
        "blurry_dropped":          0,
        "blurry_sections":         all_blurry_groups,
        "blurry_stills_skipped":   n_blurry_stills,
        "oversample_factor":       OVERSAMPLE_FACTOR,
        "resize_preset":           resize_preset or "native",
        "source_counts":           {"videos": n_videos, "stills": len(still_keys)},
        "frame_source_boundaries": frame_source_boundaries,
    }
