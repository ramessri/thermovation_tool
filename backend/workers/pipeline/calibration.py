"""
Camera calibration from a still image.

Extracts focal length from EXIF tags so the SfM stage gets an exact
prior instead of the 1.2×max(w,h) heuristic.

The key formula:
    fl_px = fl_35mm_equiv × image_width_px / 36.0

This works because the 35mm equivalent already encodes the ratio of the
actual focal length to the sensor's crop factor, referenced to the 36mm
full-frame standard.  We don't need sensor width in mm.

When actual fl_mm is available (better), we use:
    sensor_width_mm = fl_mm × 36.0 / fl_35mm_equiv
    fl_px = fl_mm × image_width_px / sensor_width_mm

Both formulae yield the same result; the second form stores the sensor
width explicitly so it can be reused if the 35mm equiv is later missing.

Usage:
    from backend.workers.pipeline.calibration import extract_calibration_from_image
    data = extract_calibration_from_image(path_to_jpg)
    # → {make, model, fl_mm, fl_35mm, sensor_width_mm, fl_px_at_width,
    #    calibration_width, calibration_height, source}
"""

from __future__ import annotations

import logging
import subprocess
import json
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _run_exiftool(path: Path) -> dict:
    """Run exiftool -json on the image and return the first tag dict."""
    try:
        out = subprocess.check_output(
            ["exiftool", "-json", "-FocalLength", "-FocalLengthIn35mmFormat",
             "-Make", "-Model", "-ImageWidth", "-ImageHeight",
             "-FocalPlaneXResolution", "-FocalPlaneYResolution",
             "-FocalPlaneResolutionUnit",
             str(path)],
            text=True, timeout=15,
        )
        records = json.loads(out)
        return records[0] if records else {}
    except Exception as e:
        logger.warning("exiftool failed on %s: %s", path, e)
        return {}


def _parse_mm(value) -> Optional[float]:
    """Parse exiftool focal length value which can be '5.4 mm' or 5.4."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).split()[0]
    try:
        return float(s)
    except ValueError:
        return None


def extract_calibration_from_image(image_path: Path) -> Optional[dict]:
    """
    Extract camera calibration data from a still image's EXIF tags.

    Returns a dict with:
        make              str   — e.g. "samsung"
        model             str   — e.g. "Galaxy S23"
        fl_mm             float — physical focal length in mm (may be None)
        fl_35mm           float — 35mm-equivalent focal length in mm
        sensor_width_mm   float — derived sensor width in mm
        fl_px_at_width    float — focal length in pixels at calibration_width
        calibration_width int   — pixel width of the calibration image
        calibration_height int  — pixel height
        source            str   — "photo_exif"

    Returns None if insufficient EXIF data to derive focal length.
    """
    tags = _run_exiftool(image_path)

    make  = str(tags.get("Make",  "") or "").strip().lower()
    model = str(tags.get("Model", "") or "").strip()

    fl_mm   = _parse_mm(tags.get("FocalLength"))
    fl_35mm = _parse_mm(tags.get("FocalLengthIn35mmFormat"))

    # Image dimensions from EXIF (may differ from actual file if cropped)
    try:
        import PIL.Image
        with PIL.Image.open(image_path) as img:
            w, h = img.size
    except Exception:
        w = int(tags.get("ImageWidth",  0) or 0)
        h = int(tags.get("ImageHeight", 0) or 0)

    if w == 0 or h == 0:
        logger.warning("calibration: could not determine image dimensions for %s", image_path)
        return None

    # ── Derive fl_px ──────────────────────────────────────────────────────────
    sensor_width_mm: Optional[float] = None

    # Method 1: FocalPlaneResolution (exact sensor dimensions from EXIF)
    fp_x    = tags.get("FocalPlaneXResolution")
    fp_unit = tags.get("FocalPlaneResolutionUnit", "inch")
    if fp_x and fl_mm:
        try:
            fp_x_val = float(str(fp_x).split("/")[0]) / (
                float(str(fp_x).split("/")[1]) if "/" in str(fp_x) else 1.0
            )
            # FocalPlaneXResolution = pixels per unit
            unit_mm = {"inch": 25.4, "cm": 10.0, "mm": 1.0}.get(
                str(fp_unit).lower().split()[0], 25.4
            )
            sensor_width_mm = w / fp_x_val * unit_mm
            logger.debug("calibration: sensor_width from FocalPlaneResolution = %.3fmm", sensor_width_mm)
        except Exception:
            pass

    # Method 2: fl_mm + fl_35mm → sensor_width (most reliable for phones)
    if sensor_width_mm is None and fl_mm and fl_35mm and fl_35mm > 0:
        sensor_width_mm = fl_mm * 36.0 / fl_35mm
        logger.debug("calibration: sensor_width from 35mm equiv = %.3fmm", sensor_width_mm)

    # Compute fl_px using the best available data
    if fl_mm and sensor_width_mm:
        fl_px = fl_mm * w / sensor_width_mm
        logger.info(
            "calibration: %s %s — fl=%.1fmm, sensor=%.2fmm, fl_px@%dpx = %.1f",
            make, model, fl_mm, sensor_width_mm, w, fl_px,
        )
    elif fl_35mm:
        # Fallback: 35mm equiv only (no physical fl_mm or sensor width)
        fl_px = fl_35mm * w / 36.0
        sensor_width_mm = None
        logger.info(
            "calibration: %s %s — fl_35mm=%.1fmm (no physical fl), fl_px@%dpx = %.1f",
            make, model, fl_35mm, w, fl_px,
        )
    else:
        logger.warning(
            "calibration: %s %s — no focal length in EXIF (tried FocalLength, "
            "FocalLengthIn35mmFormat, FocalPlaneResolution)",
            make, model,
        )
        return None

    return {
        "make":               make,
        "model":              model,
        "fl_mm":              round(fl_mm, 4) if fl_mm else None,
        "fl_35mm":            round(fl_35mm, 4) if fl_35mm else None,
        "sensor_width_mm":    round(sensor_width_mm, 4) if sensor_width_mm else None,
        "fl_px_at_width":     round(fl_px, 2),
        "calibration_width":  w,
        "calibration_height": h,
        "source":             "photo_exif",
    }


def focal_length_for_video(calibration_data: dict, video_width: int) -> float:
    """
    Scale the calibration focal length to a different pixel width (e.g. video).

    Uses sensor_width_mm if available (more accurate), otherwise scales
    fl_px_at_width proportionally (assumes same sensor area used).
    """
    fl_mm          = calibration_data.get("fl_mm")
    sensor_width_mm = calibration_data.get("sensor_width_mm")
    fl_px_at_width = calibration_data["fl_px_at_width"]
    calib_width    = calibration_data["calibration_width"]

    if fl_mm and sensor_width_mm:
        return fl_mm * video_width / sensor_width_mm

    # Proportional fallback (assumes full-width crop for both still and video)
    return fl_px_at_width * video_width / calib_width
