"""
Video metadata extraction for the Photogram pipeline.

Extracts camera intrinsics, orientation, GPS, and sensor specs from a
smartphone video file using ffprobe and exiftool.  The resulting profile
is stored on the project and used by every downstream stage.

Key outputs
-----------
focal_length_px   : float  — exact focal length in pixels (for COLMAP priors + ArUco solvePnP)
cx, cy            : float  — principal point (pixels)
rotation_deg      : int    — device rotation 0/90/180/270
gravity_vec       : [x,y,z] | None — from accelerometer track if present
gps_present       : bool
gps_lat/lon/alt   : float | None
gps_accuracy_m    : float | None
scene_hint        : "indoor" | "outdoor" | "unknown"
fps               : float
duration_s        : float
width, height     : int    — native frame resolution (pre-rotation)
stable_frame_ts   : list[float] — timestamps (s) of low-motion frames
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Known sensor databases ────────────────────────────────────────────────────
# Maps (make, model_prefix) → sensor_width_mm for common phones.
# Used when EXIF doesn't embed sensor size directly.
_SENSOR_DB: dict[tuple[str, str], float] = {
    # Samsung Galaxy S21 family
    ("samsung", "sm-g991"):  5.60,   # S21
    ("samsung", "sm-g996"):  6.43,   # S21+
    ("samsung", "sm-g998"):  6.43,   # S21 Ultra
    ("samsung", "sm-g990"):  5.60,   # S21 FE
    # S22
    ("samsung", "sm-s901"):  5.60,
    ("samsung", "sm-s906"):  6.43,
    # S23
    ("samsung", "sm-s911"):  5.60,
    ("samsung", "sm-s916"):  6.43,
    # Pixel
    ("google",  "pixel 6"):  5.76,
    ("google",  "pixel 7"):  5.76,
    ("google",  "pixel 8"):  5.76,
    # iPhone (main camera, approximate)
    ("apple",   "iphone 13"): 5.10,
    ("apple",   "iphone 14"): 5.10,
    ("apple",   "iphone 15"): 5.10,
}

_DEFAULT_SENSOR_WIDTH_MM = 5.60   # reasonable default if lookup fails


def _sensor_width(make: str, model: str) -> float:
    make_l  = (make  or "").lower()
    model_l = (model or "").lower()
    for (m, mp), w in _SENSOR_DB.items():
        if m in make_l and model_l.startswith(mp):
            return w
    return _DEFAULT_SENSOR_WIDTH_MM


# ── ffprobe helpers ───────────────────────────────────────────────────────────

def _ffprobe(video_path: Path) -> dict:
    """Run ffprobe and return parsed JSON."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(video_path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=30)
        return json.loads(out)
    except Exception as e:
        logger.warning("ffprobe failed: %s", e)
        return {}


def _exiftool(video_path: Path) -> dict:
    """Run exiftool and return parsed JSON (first record)."""
    cmd = ["exiftool", "-json", "-n", str(video_path)]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=30)
        records = json.loads(out)
        return records[0] if records else {}
    except Exception as e:
        logger.warning("exiftool failed: %s", e)
        return {}


# ── GPS helpers ───────────────────────────────────────────────────────────────

def _parse_gps(exif: dict) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Returns (lat, lon, alt, accuracy_m). All may be None."""
    lat = exif.get("GPSLatitude")
    lon = exif.get("GPSLongitude")
    alt = exif.get("GPSAltitude")
    hdop = exif.get("GPSDOP") or exif.get("GPSHPositioningError")
    # Some tools return strings with ref suffixes — handle numeric only (-n flag used)
    try:
        lat = float(lat) if lat is not None else None
        lon = float(lon) if lon is not None else None
        alt = float(alt) if alt is not None else None
        acc = float(hdop) * 5.0 if hdop is not None else None   # HDOP × 5 ≈ accuracy_m
    except (TypeError, ValueError):
        lat = lon = alt = acc = None
    return lat, lon, alt, acc


# ── Rotation helpers ──────────────────────────────────────────────────────────

def _parse_rotation(probe: dict, exif: dict) -> int:
    """Return device rotation in degrees (0 / 90 / 180 / 270)."""
    # ffprobe video stream tags
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            tags = stream.get("tags", {})
            rot = tags.get("rotate") or tags.get("Rotate")
            if rot is not None:
                try:
                    return int(float(rot))
                except ValueError:
                    pass
            # side_data_list for display matrix
            for sd in stream.get("side_data_list", []):
                if sd.get("side_data_type") == "Display Matrix":
                    rot = sd.get("rotation")
                    if rot is not None:
                        return int(abs(float(rot))) % 360
    # exiftool Rotation tag
    rot = exif.get("Rotation")
    if rot is not None:
        try:
            return int(float(rot)) % 360
        except ValueError:
            pass
    return 0


# ── Focal length helpers ──────────────────────────────────────────────────────

def _focal_length_px(exif: dict, width_px: int, sensor_width_mm: float) -> Optional[float]:
    """
    Convert focal length (mm) → pixels using sensor width.
    Returns None if focal length is unavailable.
    """
    fl_mm = exif.get("FocalLength")
    if fl_mm is None:
        # Try 35mm equivalent + sensor crop factor
        fl_35 = exif.get("FocalLengthIn35mmFormat")
        crop  = 36.0 / sensor_width_mm   # 36mm = full-frame width
        if fl_35:
            fl_mm = float(fl_35) / crop
    if fl_mm is None:
        return None
    try:
        fl_mm = float(fl_mm)
        return fl_mm * (width_px / sensor_width_mm)
    except (TypeError, ValueError):
        return None


# ── Stable frame selection ────────────────────────────────────────────────────

def _select_stable_frames(video_path: Path,
                           fps: float = 30.0, duration_s: float = 0.0) -> list[float]:
    """
    Return timestamps of frames with low camera motion, via optical flow scan.

    Results are stored in video_metadata["stable_frame_timestamps"] and surfaced
    to callers as metadata.  NOTE: extract_frames does NOT currently use these
    timestamps — it always uses step-based extraction with burst oversampling.
    The data is retained here for potential future use (e.g. a guided-capture mode
    that recommends stable positions to the user).

    No fixed ceiling on the count — every frame passing the per-window quality
    threshold is returned.

    Strategy:
      - Scan the video sequentially at ~1 fps (one decoded frame per 1s),
        using cap.grab() to skip without decoding between samples.
      - At each sampled position compute dense optical flow at 240px width.
      - Collect all (timestamp, flow_magnitude) pairs.
      - Quality threshold: adaptive — reject frames whose flow is in the top
        REJECT_PERCENTILE of the distribution (default 30%). This means we keep
        the 70% least-blurry moments regardless of how good or bad the video is.
      - Return all passing timestamps sorted by time.
    """
    import cv2
    import numpy as np

    # These timestamps are stored in video_metadata["stable_frame_timestamps"]
    # as potentially useful metadata but are NOT currently consumed by
    # extract_frames (which uses step-based oversampling). Scan at 1fps to
    # keep the compute cost reasonable (~3 min for a 4-min video at 8K).
    SCAN_FPS        = 1.0    # positions per second to evaluate
    REJECT_PCTILE   = 30.0   # keep the bottom 70% by motion magnitude per window
    DECODE_W        = 240    # decode width for optical flow (smaller = faster)

    if duration_s <= 0:
        return []

    start = min(2.0, duration_s * 0.05)   # skip shaky camera pickup
    end   = max(start + 1.0, duration_s - 2.0)

    step_frames = max(1, int(fps / SCAN_FPS))   # source frames between samples

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("VideoCapture failed to open")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        start_frame  = max(0, int(start * fps))
        end_frame    = min(total_frames - 1, int(end * fps))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        timestamps: list[float] = []
        flow_mags:  list[float] = []
        prev_gray: "np.ndarray | None" = None
        frame_idx = start_frame

        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            small = cv2.resize(frame, (DECODE_W, max(1, DECODE_W * h // w)),
                               interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None and gray.shape == prev_gray.shape:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=13,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
                mag = float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean())
            else:
                mag = 999.0

            timestamps.append(round(frame_idx / max(fps, 1), 3))
            flow_mags.append(mag)
            prev_gray = gray

            for _ in range(step_frames - 1):
                if not cap.grab():
                    break
            frame_idx += step_frames

        cap.release()

        if not timestamps:
            raise RuntimeError("no frames decoded")

        # ── Windowed quality selection ─────────────────────────────────────
        # A global threshold silently drops entire sections of the video when
        # the camera was moving (e.g. panning to film the floor) — the whole
        # scene section gets rejected, creating reconstruction blind spots.
        #
        # Fix: divide the video into temporal windows and apply the quality
        # threshold WITHIN each window.  Every window contributes its best
        # frames regardless of how the camera moved during that period.
        # Windows with lots of motion still contribute — just fewer frames —
        # but no section is completely abandoned.
        # One window per ~12s of usable video so coverage scales with length.
        # 12s ≈ the time to walk across a typical room section. A 60s clip
        # gets 5 windows; a 5-min walkthrough gets 25; a 1-min clip gets 5.
        WIN_DURATION_S = 4.0
        MIN_PER_WIN    = 2     # always keep at least this many from each window

        span           = end - start
        N_WINDOWS      = max(4, int(span / WIN_DURATION_S))
        window_s       = span / N_WINDOWS
        selected: list[float] = []

        for w in range(N_WINDOWS):
            w_start = start + w * window_s
            w_end   = w_start + window_s
            win_pairs = [
                (mag, ts)
                for ts, mag in zip(timestamps, flow_mags)
                if w_start <= ts < w_end
            ]
            if not win_pairs:
                continue
            win_pairs.sort()                           # best (lowest flow) first
            # Keep frames below the window's own 70th-percentile threshold,
            # but always keep at least MIN_PER_WIN so no section is empty.
            w_mags     = [m for m, _ in win_pairs]
            w_thresh   = float(np.percentile(w_mags, 100.0 - REJECT_PCTILE))
            qualifying = [ts for mag, ts in win_pairs if mag <= w_thresh]
            if len(qualifying) < MIN_PER_WIN:
                qualifying = [ts for _, ts in win_pairs[:MIN_PER_WIN]]
            selected.extend(qualifying)

        selected.sort()
        logger.info(
            "stable_frames: %d/%d candidates selected across %d windows "
            "(reject top %.0f%% per window, min %d/window)",
            len(selected), len(timestamps), N_WINDOWS, REJECT_PCTILE, MIN_PER_WIN,
        )
        return selected

    except Exception as e:
        logger.warning("stable_frames optical flow failed (%s) — uniform fallback", e)
        span = end - start
        step = max(span / 200, 1.0 / fps)
        ts, frames = start, []
        while ts <= end:
            frames.append(round(ts, 3))
            ts += step
        return frames


# ── Main entry point ──────────────────────────────────────────────────────────

def extract_video_metadata(video_path: Path) -> dict:
    """
    Extract all useful metadata from a smartphone video file.

    Returns a dict suitable for JSON serialisation and storage in
    projects.video_metadata.  All keys have safe defaults so callers
    never need to guard against missing keys.
    """
    probe = _ffprobe(video_path)
    exif  = _exiftool(video_path)

    # ── Resolution + FPS ──────────────────────────────────────────────────────
    width = height = 0
    fps   = 30.0
    duration_s = 0.0
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            width      = int(stream.get("width",  0))
            height     = int(stream.get("height", 0))
            r_fps      = stream.get("r_frame_rate", "30/1")
            try:
                num, den = r_fps.split("/")
                fps = float(num) / float(den)
            except Exception:
                fps = 30.0
            try:
                duration_s = float(stream.get("duration") or
                                   probe.get("format", {}).get("duration", 0))
            except Exception:
                duration_s = 0.0
            break

    if width == 0:
        width  = int(exif.get("ImageWidth",  0)) or int(exif.get("SourceImageWidth",  0))
        height = int(exif.get("ImageHeight", 0)) or int(exif.get("SourceImageHeight", 0))

    # Last resort: read dimensions directly via OpenCV (works when ffprobe binary
    # is absent but OpenCV was compiled with built-in FFmpeg support).
    if width == 0:
        try:
            import cv2 as _cv2
            _cap = _cv2.VideoCapture(str(video_path))
            if _cap.isOpened():
                width      = int(_cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
                height     = int(_cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
                fps        = float(_cap.get(_cv2.CAP_PROP_FPS)) or fps
                n_frames   = _cap.get(_cv2.CAP_PROP_FRAME_COUNT)
                if n_frames > 0 and fps > 0:
                    duration_s = n_frames / fps
                _cap.release()
                logger.info("dimensions from OpenCV fallback: %dx%d @ %.1ffps %.1fs",
                            width, height, fps, duration_s)
        except Exception as _e:
            logger.warning("OpenCV dimension fallback failed: %s", _e)

    # ── Device orientation ────────────────────────────────────────────────────
    rotation_deg = _parse_rotation(probe, exif)

    # Dimensions as seen by the pipeline (post-rotation)
    if rotation_deg in (90, 270):
        frame_w, frame_h = height, width
    else:
        frame_w, frame_h = width, height

    # ── Camera make / model ───────────────────────────────────────────────────
    make  = str(exif.get("Make",  "") or "").strip()
    model = str(exif.get("Model", "") or "").strip()

    # ── Sensor + focal length ─────────────────────────────────────────────────
    sensor_w_mm    = _sensor_width(make, model)
    focal_length_px = _focal_length_px(exif, frame_w, sensor_w_mm)
    # Principal point: assume sensor centre
    cx = frame_w / 2.0
    cy = frame_h / 2.0

    # ── GPS ───────────────────────────────────────────────────────────────────
    lat, lon, alt, gps_acc = _parse_gps(exif)
    gps_present = lat is not None and lon is not None
    if gps_present and gps_acc is not None and gps_acc < 20.0:
        scene_hint = "outdoor"
    elif not gps_present:
        scene_hint = "indoor"   # no GPS signal → very likely indoors
    else:
        scene_hint = "unknown"

    # ── Gravity / accelerometer ───────────────────────────────────────────────
    # Standard MP4 from phones doesn't carry per-frame IMU.
    # Some Samsung devices embed a "CameraElevationAngle" tag via exiftool.
    gravity_vec: Optional[list[float]] = None
    elev = exif.get("CameraElevationAngle")
    if elev is not None:
        try:
            # CameraElevationAngle is degrees above horizontal; convert to gravity vec
            import math
            a = math.radians(float(elev))
            # gravity points down: if camera horizontal, gravity is [0,1,0] in image space
            gravity_vec = [0.0, round(math.cos(a), 4), round(math.sin(a), 4)]
        except Exception:
            pass

    # ── Stable frame timestamps (metadata only — not used by extract_frames) ───
    # Stored in video_metadata for potential future use; extract_frames uses
    # step-based oversampling regardless of this value.
    stable_ts = _select_stable_frames(video_path, fps=fps, duration_s=duration_s)

    # ── EIS detection ─────────────────────────────────────────────────────────
    eis_applied = False
    # Look for Samsung/Android EIS markers in format tags
    fmt_tags = probe.get("format", {}).get("tags", {})
    for v in fmt_tags.values():
        if isinstance(v, str) and "eis" in v.lower():
            eis_applied = True
            break
    # Alternatively: if video has smaller resolution than sensor default, EIS crop likely
    # (heuristic — not definitive)

    profile = {
        # Resolution
        "width":        frame_w,
        "height":       frame_h,
        "raw_width":    width,
        "raw_height":   height,
        "rotation_deg": rotation_deg,
        # Camera intrinsics
        "focal_length_px":  focal_length_px,   # None if unavailable
        "cx":               cx,
        "cy":               cy,
        "sensor_width_mm":  sensor_w_mm,
        # Recording parameters
        "fps":         round(fps, 3),
        "duration_s":  round(duration_s, 2),
        # Device
        "make":        make,
        "model":       model,
        "eis_applied": eis_applied,
        # GPS / scene hint
        "gps_present":  gps_present,
        "gps_lat":      lat,
        "gps_lon":      lon,
        "gps_alt":      alt,
        "gps_accuracy_m": gps_acc,
        "scene_hint":   scene_hint,      # "indoor" | "outdoor" | "unknown"
        # Gravity
        "gravity_vec":  gravity_vec,     # [x, y, z] in image coords, or None
        # Frame selection
        "stable_frame_timestamps": stable_ts,
    }

    logger.info(
        "Video metadata: %dx%d rot=%d° fl=%.0fpx fps=%.1f dur=%.1fs gps=%s hint=%s make=%s %s",
        frame_w, frame_h, rotation_deg,
        focal_length_px or 0, fps, duration_s,
        gps_present, scene_hint, make, model,
    )
    return profile


def colmap_camera_params(profile: dict) -> Optional[str]:
    """
    Return COLMAP --ImageReader.camera_params string for SIMPLE_RADIAL model,
    or None if focal length is not known from metadata.

    Format: "fx,cx,cy,k1"  (k1=0 — undistorted starting point)
    """
    fl = profile.get("focal_length_px")
    if fl is None:
        return None
    cx = profile.get("cx", profile["width"] / 2.0)
    cy = profile.get("cy", profile["height"] / 2.0)
    return f"{fl:.2f},{cx:.2f},{cy:.2f},0"
