#!/usr/bin/env python3
"""
Generate a printable PDF sheet of ArUco markers for Photogram scans.

Produces a 2×3 grid of DICT_4X4_100 markers (IDs 0–5) on a single A4 page
(or letter) at 15 cm per marker side.  Also writes individual PNG files.

Usage:
    python /app/scripts/generate_aruco_sheet.py [--output samples/aruco_sheet.pdf]
    python /app/scripts/generate_aruco_sheet.py --ids 0 1 2 --size 0.20

Requirements: opencv-contrib-python, reportlab (pip install reportlab)

Printing instructions
---------------------
• Print at 100% scale (no "fit to page" / "scale to margins").
• Measure the printed square to verify the side length matches ARUCO_MARKER_SIZE_M.
• Laminate or tape flat to a rigid surface.
• Place markers flat on the floor / walls and in clear sight of the camera.
• Minimum of 2 markers is recommended for accurate metric scale.
• Lowest-ID marker is assumed to be on the floor (gravity alignment).
  Override with ARUCO_FLOOR_MARKER_ID env var.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "samples" / "aruco_sheet.pdf"
DEFAULT_MARKER_SIZE_M = float(os.environ.get("ARUCO_MARKER_SIZE_M", "0.15"))

# A4 dimensions in mm
A4_W_MM = 210.0
A4_H_MM = 297.0


def generate_marker_png(marker_id: int, size_px: int = 300) -> np.ndarray:
    """Return a square BGR image of the ArUco marker with white border."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px)
    # Add white border (~10% of size)
    border = size_px // 10
    bordered = cv2.copyMakeBorder(
        img, border, border, border, border,
        cv2.BORDER_CONSTANT, value=255,
    )
    # Convert to BGR (single-channel → 3-channel)
    return cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)


def write_pngs(ids: list[int], out_dir: Path, size_px: int = 600) -> list[Path]:
    """Write individual PNG files for each marker ID."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for mid in ids:
        img = generate_marker_png(mid, size_px)
        p   = out_dir / f"aruco_4x4_100_id{mid:03d}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)
        print(f"  Wrote {p}")
    return paths


def write_pdf(
    ids: list[int],
    output: Path,
    marker_size_m: float = DEFAULT_MARKER_SIZE_M,
    cols: int = 2,
    px: int = 600,
) -> None:
    """Write a multi-marker PDF sheet."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas as _canvas
    except ImportError:
        print("[warn] reportlab not installed — skipping PDF generation.")
        print("       Install with: pip install reportlab")
        return

    page_w, page_h = A4  # points (1 pt = 1/72 inch)

    marker_size_mm  = marker_size_m * 1000.0
    marker_size_pt  = marker_size_mm / 25.4 * 72   # mm → points

    # Safety border around the edge of each marker cell (including white border)
    cell_size_pt    = marker_size_pt * 1.2           # 10% on each side
    margin_pt       = 20 * mm

    rows = (len(ids) + cols - 1) // cols
    total_w = cols * cell_size_pt
    total_h = rows * cell_size_pt

    # Scale down if markers don't fit
    if total_w > page_w - 2 * margin_pt:
        cell_size_pt = (page_w - 2 * margin_pt) / cols
    if total_h > page_h - 60 * mm - 2 * margin_pt:
        cell_size_pt = min(cell_size_pt, (page_h - 60 * mm - 2 * margin_pt) / rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    c = _canvas.Canvas(str(output), pagesize=A4)

    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(page_w / 2, page_h - 25 * mm, "Photogram ArUco Markers — DICT_4X4_100")
    c.setFont("Helvetica", 9)
    c.drawCentredString(
        page_w / 2, page_h - 33 * mm,
        f"Print at 100% scale · Marker side = {marker_size_mm:.0f} mm · "
        f"IDs: {', '.join(str(i) for i in ids)}"
    )
    c.drawCentredString(
        page_w / 2, page_h - 39 * mm,
        "Verify: measure the grey square (not including white border) — must match above."
    )

    import tempfile, io

    top_start = page_h - 50 * mm

    for idx, mid in enumerate(ids):
        row = idx // cols
        col = idx % cols

        x = margin_pt + col * cell_size_pt
        y = top_start - (row + 1) * cell_size_pt

        # Render marker to in-memory PNG
        img = generate_marker_png(mid, px)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        cv2.imwrite(tmp_path, img)
        c.drawImage(tmp_path, x, y, width=cell_size_pt, height=cell_size_pt)
        os.unlink(tmp_path)

        # Label below marker
        c.setFont("Helvetica", 8)
        c.drawCentredString(
            x + cell_size_pt / 2,
            y - 6,
            f"ID {mid}" + (" ← floor (gravity ref)" if mid == min(ids) else ""),
        )

    # Footer
    c.setFont("Helvetica", 7)
    c.drawCentredString(
        page_w / 2, 15 * mm,
        "ARUCO_MARKER_SIZE_M env var must match the printed marker side length."
    )

    c.save()
    print(f"  Wrote {output}")


def main():
    parser = argparse.ArgumentParser(description="Generate printable ArUco marker sheet")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output PDF path")
    parser.add_argument("--ids", nargs="+", type=int, default=list(range(6)),
                        help="Marker IDs to include (default: 0-5)")
    parser.add_argument("--size", type=float, default=DEFAULT_MARKER_SIZE_M,
                        help="Physical marker side length in metres (default: 0.15)")
    parser.add_argument("--cols", type=int, default=2, help="Columns per row (default: 2)")
    parser.add_argument("--pngs-only", action="store_true",
                        help="Skip PDF, only write individual PNG files")
    args = parser.parse_args()

    output = Path(args.output)
    ids    = args.ids

    print(f"\nGenerating ArUco markers: IDs {ids}, size={args.size*100:.0f} cm")

    # Always write individual PNGs
    png_dir = output.parent / "aruco_pngs"
    write_pngs(ids, png_dir)

    if not args.pngs_only:
        write_pdf(ids, output, args.size, args.cols)

    print(f"\nDone.  Lowest ID ({min(ids)}) = floor/gravity reference marker.")
    print(f"Set ARUCO_MARKER_SIZE_M={args.size} before scanning.\n")


if __name__ == "__main__":
    main()
