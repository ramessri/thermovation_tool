"""
Photogram — FastAPI application entry point.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from backend.core.config import settings
from backend.core.storage import get_storage, LocalStorageBackend
from backend.core.db import engine
from backend.api.routes import projects, jobs, anchors, ws


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure temp dir exists
    settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    settings.LOCAL_STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="Photogram API",
    version="0.1.0",
    lifespan=lifespan,
)

cors_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",")] if isinstance(settings.CORS_ORIGINS, str) else settings.CORS_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve local storage files with no-cache headers so the browser always
# re-validates after a pipeline re-run regenerates a cloud file.
storage = get_storage()
if isinstance(storage, LocalStorageBackend):
    settings.LOCAL_STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

    @app.api_route("/files/{file_path:path}", methods=["GET", "HEAD"])
    async def serve_storage_file(file_path: str, request: Request):
        # Strip query-string cache-busters (e.g. ?v=12345) — path only matters.
        full_path = settings.LOCAL_STORAGE_ROOT / file_path
        if not full_path.exists() or not full_path.is_file():
            return Response(status_code=404)
        # Detect content type by suffix
        suffix = full_path.suffix.lower()
        media_map = {
            ".ply":  "application/octet-stream",
            ".las":  "application/octet-stream",
            ".laz":  "application/octet-stream",
            ".obj":  "text/plain",
            ".json": "application/json",
            ".jpg":  "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png":  "image/png",
            ".mp4":  "video/mp4",
            ".mov":  "video/quicktime",
            ".avi":  "video/x-msvideo",
            ".mkv":  "video/x-matroska",
            ".webm": "video/webm",
            ".heic": "image/heic",
            ".heif": "image/heif",
        }
        media_type = media_map.get(suffix, "application/octet-stream")
        is_video = suffix in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        # Videos need range request support for scrubbing — allow conditional
        # caching but not no-store (which breaks partial content buffering).
        cache_headers = (
            {"Cache-Control": "no-cache", "Accept-Ranges": "bytes"}
            if is_video else
            {"Cache-Control": "no-cache, no-store, must-revalidate",
             "Pragma": "no-cache", "Expires": "0"}
        )
        return FileResponse(
            path=str(full_path),
            media_type=media_type,
            headers=cache_headers,
        )

    @app.get("/preview/video/{file_path:path}")
    async def preview_video(file_path: str, request: Request):
        """Stream a browser-compatible H.264 transcode of any video file.

        Transcodes on the fly via ffmpeg piping — no temp file written.
        Scales to max 1280px wide, 2 Mbps, fast-start for web playback.
        """
        import asyncio, subprocess as _sp
        from starlette.responses import StreamingResponse

        full_path = settings.LOCAL_STORAGE_ROOT / file_path
        if not full_path.exists() or not full_path.is_file():
            return Response(status_code=404)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(full_path),
            "-vf", "scale='min(1280,iw)':-2",   # max 1280px wide, keep aspect
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "frag_keyframe+empty_moov+faststart",  # streamable MP4
            "-f", "mp4",
            "pipe:1",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

        async def stream():
            try:
                while True:
                    chunk = await proc.stdout.read(65536)
                    if not chunk:
                        break
                    yield chunk
            finally:
                try:
                    proc.kill()
                except Exception:
                    pass
                await proc.wait()

        return StreamingResponse(
            stream(),
            media_type="video/mp4",
            headers={"Cache-Control": "no-cache", "Accept-Ranges": "none"},
        )

# Routers
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(jobs.router,     prefix="/api/jobs",     tags=["jobs"])
app.include_router(anchors.router,  prefix="/api/anchors",  tags=["anchors"])
app.include_router(ws.router,       prefix="/ws",           tags=["websocket"])


@app.get("/api/health")
async def health():
    return {"status": "ok", "storage_backend": settings.STORAGE_BACKEND}


@app.get("/api/aruco-sheet.pdf", response_class=Response)
async def aruco_sheet_pdf(size: float = 0.15):
    """
    Generate a printable multi-page A4 PDF of all 50 DICT_4X4_100 ArUco markers
    (IDs 0–49).  4 markers per page — large enough to read reliably at 15 cm side.
    Print the pages you need and place them around the room.

    Query params:
        size — physical marker side in metres (default 0.15)
    """
    import os, tempfile
    import cv2

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas as _canvas
    except ImportError:
        return Response(status_code=500, content="reportlab not installed")

    marker_ids  = list(range(50))
    aruco_dict  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    marker_size_mm = size * 1000.0
    page_w, page_h = A4
    margin_pt   = 18 * mm
    cols        = 2
    per_page    = 4   # 2×2 grid — keeps markers large and easy to cut out
    cell_pt     = (page_w - 2 * margin_pt) / cols   # fill page width

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = f.name

    tmp_pngs: list[str] = []
    try:
        c = _canvas.Canvas(pdf_path, pagesize=A4)

        for page_idx, page_start in enumerate(range(0, len(marker_ids), per_page)):
            page_ids = marker_ids[page_start:page_start + per_page]
            page_num = page_idx + 1
            total_pages = (len(marker_ids) + per_page - 1) // per_page

            if page_idx > 0:
                c.showPage()

            # Page header
            c.setFont("Helvetica-Bold", 11)
            c.drawCentredString(page_w / 2, page_h - 12 * mm,
                                "Photogram ArUco Markers — DICT_4X4_100")
            c.setFont("Helvetica", 8)
            c.drawCentredString(page_w / 2, page_h - 18 * mm,
                                f"Print at 100% · Side = {marker_size_mm:.0f} mm · "
                                f"Page {page_num}/{total_pages} · "
                                f"IDs {page_ids[0]}–{page_ids[-1]}")

            top_start = page_h - 24 * mm
            for slot, mid in enumerate(page_ids):
                row, col = divmod(slot, cols)
                x = margin_pt + col * cell_pt
                y = top_start - (row + 1) * cell_pt

                img_gray = cv2.aruco.generateImageMarker(aruco_dict, mid, 700)
                border   = 70   # white quiet zone
                img_bordered = cv2.copyMakeBorder(
                    img_gray, border, border, border, border,
                    cv2.BORDER_CONSTANT, value=255,
                )
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tp:
                    cv2.imwrite(tp.name, img_bordered)
                    tmp_pngs.append(tp.name)
                c.drawImage(tp.name, x, y, width=cell_pt, height=cell_pt)

                # Label
                c.setFont("Helvetica-Bold", 9)
                note = "  ← floor marker (gravity ref)" if mid == 0 else ""
                c.drawCentredString(x + cell_pt / 2, y - 6 * mm, f"ID {mid}{note}")

            # Footer
            c.setFont("Helvetica", 7)
            c.drawCentredString(page_w / 2, 10 * mm,
                                f"Set ARUCO_MARKER_SIZE_M={size} · "
                                "Lowest ID in scene = floor/gravity reference")

        c.save()
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
    finally:
        os.unlink(pdf_path)
        for p in tmp_pngs:
            try:
                os.unlink(p)
            except OSError:
                pass

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "inline; filename=aruco_markers.pdf"},
    )
