# Photogram — Claude Code orientation

Photogrammetry pipeline across three scene modes (indoor_room · outdoor · object):
video or photos → metric-scaled dense point cloud, mesh, and Gaussian splat.
**Scale is automatic — derived from ArUco fiducial markers printed in the scene.**

Three scan modes are supported, selected at project creation via `scene_type`:
- `indoor_room` (default) — walk-through of an enclosed space
- `outdoor` — general exterior survey; non-flat terrain, no walls
- `object` — orbit around a discrete object (stills and/or video)

## Pipeline architecture

Single continuous chain — no user gate.  Stages marked ✦ are conditional on `scene_type`.

| Stage | Task name | Queue | Key lib | Modes | Notes |
|-------|-----------|-------|---------|-------|-------|
| Extract metadata | `pipeline.extract_metadata` | cpu | OpenCV + exiftool | all | focal length, GPS, rotation; injects `scene_type` into chain |
| Extract frames | `pipeline.extract_frames` | cpu | OpenCV | all | step-based oversampling, best-of-burst; blurry groups flagged |
| Detect ArUco | `pipeline.detect_aruco` | cpu | cv2.aruco | all | **FAIL FAST** if no markers found |
| Feature matching | `pipeline.feature_matching` | gpu | LightGlue + DISK | all | object mode uses all-pairs window (≥999) instead of default sliding window of 15 |
| SfM | `pipeline.sfm` | gpu | pycolmap | all | focal length prior from metadata; gap warnings in result |
| MVS dense | `pipeline.mvs` | gpu | COLMAP CLI (CUDA) | all | |
| Trajectory correction ✦ | `pipeline.correct_trajectory_jumps` | cpu | open3d ICP | indoor, outdoor | skipped for object — orbit paths have no teleport jumps |
| Gaussian splatting ✦ | `pipeline.gaussian_splatting` | gpu | nerfstudio splatfacto | object only | inserted after MVS for object scans |
| Detect ArUco (SfM) | `pipeline.detect_aruco_sfm` | cpu | cv2.aruco | all | re-scan registered frames with COLMAP intrinsics |
| Scale from ArUco | `pipeline.scale_from_aruco` | cpu | numpy | all | triangulates markers, derives m/unit; fits floor plane |
| Apply scale | `pipeline.apply_known_scale` | cpu | open3d | all | scales dense cloud to metric |
| Fill planes ✦ | `pipeline.fill_planes` | cpu | open3d | indoor only | skipped for outdoor (terrain not flat) and object (no floor concept) |
| Refine cloud | `pipeline.refine_cloud` | cpu | open3d | all | SOR outlier removal + voxel downsampling; object mode uses finer voxel (1.5 mm vs 3 mm) |
| Coverage | `pipeline.coverage` | cpu | open3d HPR + DBSCAN | all | boundary clip and suggestions vary by scene_type (see below) |
| Export | `pipeline.export` | cpu | laspy, open3d | all | |

Pipeline starts at `processing` and ends at `complete` or `needs_more`.
Coverage threshold for `needs_more`: **80 %** for object, **50 %** for indoor and outdoor.

### Coverage boundary per scene type

| Mode | Boundary | How |
|------|----------|-----|
| `indoor_room` | None (full cloud) | — |
| `object` | 3D convex hull of camera positions | Object sits inside the orbit; hull excludes floor/background |
| `outdoor` | 2D horizontal footprint of camera walk + height band | Footprint = 2D convex hull of camera XZ projections; height band = terrain floor (5th-pct camera height − 0.5 m) to highest camera + 5 m |

Shot suggestions use scene-aware labels:
- indoor: "floor", "ceiling", "+X wall" etc.
- object: object-relative labels ("top", "lower front-right side", "back") derived from cluster centroid vs object center + gravity
- outdoor: "ground/terrain surface", "upper/overhead surface"

### ArUco requirement

Every scan must include printed ArUco markers (DICT_4X4_100).  The pipeline
fails immediately at `detect_aruco` if no markers are found — before any
expensive COLMAP stages run.

**Minimum setup:** 1 marker (scale from solvePnP tvec — less accurate).
**Recommended:** 2+ markers visible in the same frames (scale from triangulated
inter-marker baselines — accurate to ~2%).

Generate a printable marker sheet:
```bash
docker compose exec worker-gpu python /app/scripts/generate_aruco_sheet.py
# produces samples/aruco_sheet.pdf
```

Marker physical size is set via `ARUCO_MARKER_SIZE_M` (default `0.15` = 15 cm side).
Floor marker ID is auto-detected as the lowest numeric ID, or override with
`ARUCO_FLOOR_MARKER_ID`.

## Mesh export pipeline (`backend/workers/pipeline/exporter.py`)

Reconstruction algorithm selected by `scene_type` (read from DB if not in `prev_result`):

| scene_type | Algorithm | Source cloud | Key params |
|---|---|---|---|
| `object` | Ball Pivoting (BPA) | MVS `dense.ply` | Orbit sphere clip 0.9×, metric scale, KNN radii |
| `outdoor` | Ball Pivoting (BPA) | Coverage cloud (downsampled) | KNN-based voxel sizing, normals toward camera centroid |
| `indoor_room` | Poisson depth=9 | Coverage cloud | 5% density trim |

**Why BPA for object/outdoor:** Poisson fills open space with hallucinated geometry — catastrophic for swing sets, lawn mowers, any scene with thin structures.

Post-processing chain (all scene types):
1. Component filter — keep components ≥1% of largest (preserves wheels/handles)
2. Centroid-distance filter (object only) — drop background slabs >0.6× orbit radius from centroid
3. Long-edge filter — remove triangles >10× median edge (background tent triangles)
4. Hole filling via `trimesh.repair.fill_holes()`
5. Taubin smoothing — 30 iterations

For object mode, `exports/output.ply` is the scaled MVS cloud with real RGB (orbit-clipped), not the coverage-colored GS cloud.

## `pipeline_results` JSON column

Added to `projects` table. Populated by `emit_stage_complete()` after every stage with metrics + `_duration_s`, `_started_at`, `_finished_at`. Used by the frontend for:
- Stage elapsed time display on page reload
- Quality score card (sfm/mvs/coverage metrics)
- Project listing score chip (color-coded `/100`)

## Key files

- `backend/workers/tasks.py` — Celery app, all task definitions, `launch_pipeline`
- `backend/workers/pipeline/extract_metadata.py` — ffprobe + exiftool video profiling
- `backend/workers/pipeline/extractor.py` — step-based frame extraction with burst oversampling
- `backend/workers/pipeline/aruco_sfm.py` — post-SfM ArUco re-scan on registered frames
- `backend/workers/pipeline/scout_calibrate.py` — scout calibration + full pipeline launch
- `backend/workers/pipeline/room_layout.py` — gravity-aware floor plane projection + void fill (called by fill_planes task)
- `backend/workers/pipeline/refine_cloud.py` — SOR outlier removal + voxel downsampling
- `backend/workers/pipeline/aruco_detector.py` — ArUco detection + baseline measurement
- `backend/workers/pipeline/matcher.py` — LightGlue feature matching
- `backend/workers/pipeline/sfm.py` — pycolmap SfM with focal prior injection
- `backend/workers/pipeline/mvs.py` — COLMAP dense reconstruction
- `backend/workers/pipeline/scale_from_aruco.py` — post-SfM triangulation + scale derivation
- `backend/workers/pipeline/coverage.py` — coverage scoring + shot suggestions
- `backend/workers/pipeline/exporter.py` — PLY/OBJ/LAS export
- `ml/metadata/extractor.py` — `extract_video_metadata()`: ffprobe + exiftool → focal length, rotation, GPS
- `alembic/versions/` — DB migrations (010 is current head)
- `frontend/components/ResultsViewer.tsx` — tabbed Reconstruction view (Point Cloud, Mesh, Scene Overview, Walkthrough, RGB Cloud, Re-shoot)
- `frontend/components/CameraWalkthroughViewer.tsx` — "Walkthrough" tab: step through real SfM camera poses, rotate-in-place free-look (FOV matched to lens), cloud/mesh toggle, photo-overlay compare, fullscreen
- `frontend/components/CameraPathViewer.tsx` — "Scene Overview" tab: top-down camera trajectory + re-shoot/gap markers
- `frontend/components/lib/poseMath.ts` — shared cameras.json pose math (`worldPosition`/`worldDirection`/`worldBasis`), used by both viewers above

## Infrastructure quirks

**`worker-gpu` must run privileged.**
CUDA CDI on this host requires `privileged: true` in the compose service. Without it, the GPU worker starts but torch cannot access the device.

**Frontend has NO hot-reload — rebuild required after every `frontend/` change.**
The `frontend` service builds a production image (`next start` from
`.next/standalone`, no source volume mount). Editing files under `frontend/`
has zero effect until you:
```bash
docker compose build frontend
docker compose up -d frontend
```
Verify the change actually shipped (don't trust "container is up"):
```bash
docker compose exec frontend sh -c "grep -rl '<ComponentName>' /app/.next/static/chunks/"
```
A container that's been `Up` for hours despite a fresh edit is the tell that
it's serving a stale build.

**GPU task routing via `task_routes`, not `.set(queue=)`.**
Queue overrides on chained tasks using `.set(queue="gpu")` do not reliably override `task_routes`. The authoritative routing is in `celery_app.conf.task_routes` inside `tasks.py`. Adding `.set(queue=)` to chains is redundant and can confuse debugging.

**DB enum casing — use `.value` (lowercase).**
The `jobstatus` and `projectstatus` Postgres enums have both UPPERCASE variants (from the initial `create_all` run) and lowercase variants (added by alembic later). All ORM code uses `.value` which produces lowercase strings. Raw psycopg2 inserts must cast explicitly: `%s::jobstatus`.

**First-time DB setup outside migrations.**
If Postgres tables were created by SQLAlchemy `create_all` before Alembic was introduced, stamp the current head before running `alembic upgrade head`:

```bash
docker compose exec api alembic stamp 001
docker compose exec api alembic upgrade head
```

**Hot-reload mounts (filesystem only — Celery does NOT hot-reload code).**
The GPU worker mounts `./backend` and `./ml`, so file changes are immediately visible on the container filesystem. However, Celery workers load Python modules once at startup and do not re-import on file change. After editing any `.py` file under `backend/` or `ml/`, you must restart the worker:
```bash
docker compose restart worker-gpu
```
The API container (`uvicorn --reload`) does hot-reload Python, so API changes take effect without a restart.

**NEVER restart workers during an active pipeline run.**
Celery uses `acks_early` — tasks are acknowledged before execution. If the worker dies mid-task, the task is lost and NOT retried. Projects get stuck in `processing` with empty queues. Manual re-dispatch is needed: use `/api/projects/{id}/reprocess` for post-SfM stages, or manually call `task.apply_async()` for earlier stages.

**`worker-cpu` cannot run open3d tasks — libgomp1 missing.**
`worker-cpu` doesn't have `libgomp.so.1`. Any task using open3d (refine_cloud, fill_planes, coverage, correct_trajectory_jumps) will crash on worker-cpu. Fix added to `docker/Dockerfile.worker-cpu` (`apt-get install libgomp1`) but not yet rebuilt. Keep worker-cpu stopped (`docker compose stop worker-cpu`) — worker-gpu subscribes to both `gpu` and `celery` queues and handles everything.

**`scene_type` lost in reprocess chains.**
The `/reprocess` endpoint reads from the checkpoint (detect_aruco_sfm output). The checkpoint may not carry `scene_type`. Both `analyze_coverage` and `export_outputs` tasks now read `scene_type` from the DB as fallback. Same pattern for `confirmed_scale_factor` in the exporter.

## Running the smoke harness

All stages run inside `worker-gpu`:

```bash
make smoke STAGE=extract_metadata   # requires samples/20260520_195121.mp4
make smoke STAGE=extract_frames
make smoke STAGE=detect_aruco       # FAILS if no ArUco markers in footage
make smoke STAGE=feature_matching
make smoke STAGE=sfm
make smoke STAGE=mvs
make smoke STAGE=scale_from_aruco
make smoke STAGE=coverage
make smoke STAGE=export
make smoke STAGE=all
```

Manifests are persisted to `samples/manifests/<stage>.json`. Each stage reads the prior
stage's manifest, so you can resume mid-pipeline.

### Bypassing ArUco for smoke tests (no markers available)

If test footage doesn't have ArUco markers, you can inject a synthetic
`detect_aruco` manifest and skip to `feature_matching`:

```bash
docker compose exec worker-gpu python /app/scripts/smoke.py --stage extract_frames
# Then inject synthetic ArUco manifest:
docker compose exec worker-gpu python /app/scripts/inject_mock_aruco.py
# Continue from feature_matching:
docker compose exec worker-gpu python /app/scripts/smoke.py --stage feature_matching
```

## Debugging a failed pipeline stage

Pipeline errors surface in two places:

1. **Redis db1** — Celery result backend stores the full traceback. Connect with:
   ```bash
   docker compose exec redis redis-cli -n 1
   GET celery-task-meta-<celery_task_id>
   ```

2. **`jobs` table** — `error` column holds the exception string. Query:
   ```sql
   SELECT stage, status, error FROM jobs WHERE project_id = '<id>' ORDER BY created_at DESC;
   ```

WebSocket progress events are published to `project:{project_id}:progress` on Redis db0.

## Tunable environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ARUCO_MARKER_SIZE_M` | `0.15` | Physical side length of printed ArUco markers (metres) |
| `ARUCO_DICT` | `DICT_4X4_100` | ArUco dictionary — must match printed markers |
| `ARUCO_MIN_FRAMES` | `2` | Minimum frames a marker must appear in to be trusted |
| `ARUCO_FLOOR_MARKER_ID` | lowest ID | Which marker is on the floor (for gravity alignment) |
| `GSPLAT_ITERATIONS` | `15000` | 3DGS training iterations (used in object chain after MVS) |
