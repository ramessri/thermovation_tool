# Photogram — Claude Code orientation

Photogrammetry pipeline across three scene modes (indoor_room · outdoor · object):
video or photos → metric-scaled dense point cloud, mesh, and Gaussian splat.
**Scale is automatic — derived from printed fiducials in the scene: ArUco markers or the custom 3×3 grid marker (chosen per project via `marker_type`).**

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
| LingBot fusion ✦ | `pipeline.lingbot_fusion` | gpu | LingBot-Map + open3d TSDF | all (flagged) | **optional, off by default** (`ENABLE_LINGBOT_FUSION`). Densifies via depth fusion; **additive** — emits `lingbot_cloud_key`/`lingbot_mesh_key`, does not alter `dense_cloud_key`. Depth inference runs in an isolated torch-2.8 venv via subprocess |
| MetricAnything fusion ✦ | `pipeline.metricanything_fusion` | gpu | MetricAnything + open3d TSDF | all video (flagged) | **optional, off by default** (`ENABLE_METRICANYTHING_FUSION` + per-project `metricanything_enabled`). Same additive role as LingBot — emits `metricanything_cloud_key`/`metricanything_mesh_key`. Runs **in-process** (torch<2.5 requirement met by 2.4.1) |
| HVAC placement ✦ (4 stages) | `pipeline.wall_plane_detection` → `detect_hvac_fixtures` → `locate_rucklauf` → `hvac_placement` | gpu | SegFormer/ADE20K, GDINO, SAM2, open3d | indoor only (flagged) | **optional, off by default** (`ENABLE_HVAC_PLACEMENT` + per-project `hvac_mode`). Non-fatal. See "HVAC placement" below |
| Coverage | `pipeline.coverage` | cpu | open3d HPR + DBSCAN | all | boundary clip and suggestions vary by scene_type (see below) |
| Export | `pipeline.export` | cpu | laspy, open3d | all | |

Pipeline starts at `processing` and ends at `complete` or `needs_more`.

### HVAC placement (indoor_room, opt-in)

Recommends where to mount a wall unit (`HVAC_UNIT_SIZE_CM`, default 60×40), anchored on the Rücklauf. All four stages work in the metric frame of the refined cloud; camera poses come from `depth_fusion_common.load_registered_cameras(..., scale_factor=confirmed_scale_factor)` so `t` matches. 2D detections are lifted to 3D by projecting the dense cloud into the frame (`hvac_common.points_in_box/points_in_mask`), not by ray triangulation.
1. **Walls** — SegFormer/ADE20K wall masks on ≤`HVAC_MAX_WALL_SEG_FRAMES` frames select the cloud points; RANSAC (`room_layout._ransac_plane`) fits ≤6 planes on only those; hard gate ≤25° from vertical; dedup; green photo overlays per top-3 wall.
2. **Fixtures** — one GDINO pass (pipe, valve, radiator, outlet, window, return/supply pipe), SAM2 mask refinement (box fallback if SAM2 won't load), ≥20 points per detection, DBSCAN (0.30 m) into instances with `best_frame` + `best_box_frac`.
3. **Rücklauf / Vorlauf** — blue / red cap colour cues (HSV), else the best GDINO "return/supply pipe" instance; snaps to a pipe cluster within 40 cm. None → free-wall mode.
4. **Placement** — walls must clear `HVAC_MIN_WALL_INLIERS` and `HVAC_MIN_ADE20K_CONFIDENCE`; grid search with fixture keep-outs (0.15 m) and `HVAC_MIN_CLEARANCE_CM`; score = distance to Rücklauf (or −clearance). Top 5 keep `corners_world_m`. Rank 1 is drawn on the largest clean (`quad_is_clean`), unoccluded (`_is_occluded`) photo. **No minimum mounting height** — a low Rücklauf pulls rank 1 to the floor line (and such a spot then gets no photo overlay, since the floor corner occludes it).

Results: `projects.hvac_placement` + `projects.hvac_segmentation` (export; kept via COALESCE on reprocess). UI: **Segmentation** tab (`HvacSegmentationViewer.tsx` — photos with overlays) and a **Show placement** toggle on the Point Cloud / Mesh tabs (`lib/hvacOverlay.ts`): rank 1 filled green, ranks 2–5 amber outlines, Rücklauf blue, Vorlauf red, fixtures as dots. The overlay is added as a child of the loaded cloud/mesh so it shares the viewer's re-centring and COLMAP→Three.js flip; it's only shown on metric-frame views (export cloud/mesh, densified clouds), not the RGB tab's raw MVS cloud. Floor height for `mount_height_cm` comes from `fitted_planes` (ArUco floor marker) — grid-marker scans get `None`.

### LiDAR ingestion (`scan_source="lidar_ply"`)

Alternate chain for an already-built `.ply` point cloud (chosen per launch in the UI's "Scan source" selector; `POST /launch?scan_source=lidar_ply&lidar_scale_factor=…`):
`ingest_lidar_ply` → `refine_cloud` → `export` (`launch_lidar_pipeline` in `tasks.py`). No camera poses, so no markers, coverage, or densifier.
- **Scale:** trusted from the scanner (metres); `lidar_scale_factor` multiplies through otherwise (e.g. `0.001` for mm). Rescaled cloud → `{pid}/lidar/ingested.ply`, `confirmed_scale_factor=1.0`, source `lidar_native` / `lidar_manual`.
- **Gravity:** `dimensions.detect_floor_gravity` — iterative RANSAC (≤8 planes, 2 cm, ≥2000 inliers), group normals within 12°, vertical = group with the **smallest** cloud extent (a room is wider than tall; "largest plane = floor" fails when a clean wall out-votes a cluttered floor). Skipped for `object`. Wrong for tall narrow spaces (stairwells).

### Dimensions (L × B × H)

Every input path ends with dimensions — ArUco video, grid-marker video, and LiDAR. `export` computes them whenever the cloud is metric (`confirmed_scale_factor` set) and the scene isn't `object`. "Up" comes from the ArUco floor marker or LiDAR ingest; otherwise `dimensions.detect_floor_gravity` finds it from the cloud's own floor (same algorithm as LiDAR ingest). If marker scale couldn't be derived the cloud is in SfM units, so no dimensions are reported. SOR → rotate up to +Z → H = 0.5–99.5 pct vertical extent, L/B = `cv2.minAreaRect` of the footprint. Optional tape-measure `ground_truth_{length,breadth,height}_m` at launch (or `/reprocess?from_stage=export`) adds `ground_truth_check` with per-field error %. Stored in `projects.dimensions` (+ `pipeline_results.export.dimensions`); shown by `frontend/components/DimensionsCard.tsx` in the completion card. Furniture / scan bleed through doors inflates L/B (only H is percentile-clipped).
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

### Scale marker choice (`marker_type`)

Chosen in the UI at project creation (`projects.marker_type`, alembic 013):
- `aruco` (default) — everything in "ArUco requirement" below.
- `grid` — custom 3×3 grid sheet: 28.6 × 20.2 cm, 9 black 3.9 cm squares, centres 12.35 cm apart horizontally / 8.15 cm vertically. `detect_aruco` / `detect_aruco_sfm` run `grid_marker_detector.detect_marker()` instead of ArUco (same stage names); fail fast if the board is seen in < `ARUCO_MIN_FRAMES` frames. `scale_from_aruco` triangulates the 9 square centres from SfM cameras and fits a similarity transform to the known layout (`grid_marker_scale.py`) → `confirmed_scale_source = "grid_marker"`. No floor-marker gravity or marker-fitted floor plane in grid mode.

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
- `backend/workers/pipeline/lingbot_fusion.py` — optional densification stage: per-frame affine depth calibration + TSDF fusion (runs in main env; calls the inference subprocess)
- `backend/workers/pipeline/hvac_common.py` — HVAC helpers: frame download, cloud→frame projection, `wall_basis`, `quad_is_clean`, lazy GDINO / SAM2 / SegFormer loaders
- `backend/workers/pipeline/wall_plane_detection.py`, `detect_hvac_fixtures.py`, `locate_rucklauf.py`, `hvac_placement.py` — the four HVAC stages
- `frontend/components/HvacSegmentationViewer.tsx` — "Segmentation" tab; `frontend/components/lib/hvacOverlay.ts` — 3D placement overlay for the Point Cloud / Mesh tabs
- `backend/workers/pipeline/metricanything_fusion.py` — optional densifier: MetricAnything metric depth per frame (calibrated f_px = COLMAP fx), per-frame affine calibration against `mvs/dense.ply`, TSDF fusion, void-gated merge
- `backend/workers/pipeline/depth_fusion_common.py` — shared by both densifiers: `robust_affine`, `load_registered_cameras` (cameras.json → K, world→cam R/t in SfM units)
- `backend/workers/pipeline/lingbot/infer_depth.py` — LingBot depth inference, runs in the isolated `/opt/lingbot-venv` (torch 2.8); **never imported by the worker**, file-path subprocess only
- `backend/workers/pipeline/aruco_detector.py` — ArUco detection + baseline measurement (dispatches to grid detector when `marker_type="grid"`)
- `backend/workers/pipeline/grid_marker_detector.py` — 3×3 grid marker detection (OpenCV + numpy): blobs → grid fit → darkness check → px/cm
- `backend/workers/pipeline/grid_marker_scale.py` — post-SfM grid triangulation + Umeyama scale fit
- `backend/workers/pipeline/matcher.py` — LightGlue feature matching
- `backend/workers/pipeline/sfm.py` — pycolmap SfM with focal prior injection
- `backend/workers/pipeline/mvs.py` — COLMAP dense reconstruction
- `backend/workers/pipeline/scale_from_aruco.py` — post-SfM triangulation + scale derivation
- `backend/workers/pipeline/coverage.py` — coverage scoring + shot suggestions
- `backend/workers/pipeline/exporter.py` — PLY/OBJ/LAS export + L×B×H dimensions
- `backend/workers/pipeline/lidar_ingest.py` — LiDAR `.ply` ingest: scale + RANSAC gravity
- `backend/workers/pipeline/dimensions.py` — floor/gravity detection, L×B×H from a gravity-aligned metric cloud, ground-truth comparison
- `backend/core/storage.py` — storage abstraction (`STORAGE_BACKEND=local` only; files under `LOCAL_STORAGE_ROOT`)
- `ml/metadata/extractor.py` — `extract_video_metadata()`: ffprobe + exiftool → focal length, rotation, GPS
- `alembic/versions/` — DB migrations (016 is current head)
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

**LingBot fusion runs in an ISOLATED venv — never import `lingbot` in the worker.**
The main worker env is torch 2.4.1/cu124 (pycolmap, LightGlue, gaussian-splatting CUDA ext, MASt3R all built against it). LingBot-Map needs torch 2.8/cu128, installed separately in `/opt/lingbot-venv` with the repo at `/opt/lingbot-map` (pinned commit). The stage (`lingbot_fusion.py`) does only numpy/open3d/scipy and shells out to `/opt/lingbot-venv/bin/python .../infer_depth.py` for depth inference — importing the `lingbot` package in the worker would clash torch ABIs. The fusion stage is **non-fatal**: on error it logs and returns `prev_result` so the pipeline still completes. The ~4.6 GB checkpoint lazy-downloads to `/app/models/lingbot/` on first run. Default off — set `ENABLE_LINGBOT_FUSION=1` to enable. Memory knobs match the 12 GB-GPU defaults proven earlier (windowed mode >120 frames, `num_scale_frames=2`, `camera_num_iterations=1`).

**MetricAnything is vendored into the worker image at `/opt/metric-anything`.**
`docker/Dockerfile.worker-gpu` clones github.com/metric-anything/metric-anything at a pinned commit outside `/app/backend` (the `./backend` bind mount would hide it there); `METRICANYTHING_VENDOR_DIR` points at it. The checkpoint (`yjh001/metricanything_student_depthmap`) lazy-downloads to `/app/models/metricanything` on first run. Unlike LingBot it runs in the worker process — keep the worker on torch <2.5 or move it to a venv+subprocess like LingBot. Fusion runs in SfM units against the **raw** `mvs/dense.ply` (poses and reference cloud must share units) and scales by `confirmed_scale_factor` at the end. Frames with <50 projected COLMAP points, <40 % inliers, or depth uncorrelated with COLMAP (r < 0.5) are skipped; zero calibrated frames raises → caught as non-fatal. Not available for LiDAR scans (no frames/poses). `scale_std` in the stage metrics shows per-frame scale drift.

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
| `GRID_MARKER_SQUARE_CM` / `GRID_MARKER_PITCH_U_CM` / `GRID_MARKER_PITCH_V_CM` | `3.9` / `12.35` / `8.15` | Grid marker geometry (cm) — change only these for a differently sized print |
| `GSPLAT_ITERATIONS` | `15000` | 3DGS training iterations (used in object chain after MVS) |
| `ENABLE_LINGBOT_FUSION` | `false` | Enable the optional LingBot depth-fusion densification stage |
| `LINGBOT_CHECKPOINT` | `/app/models/lingbot/lingbot-map.pt` | Checkpoint path (lazy-downloaded on first run) |
| `LINGBOT_WINDOWED_THRESHOLD` | `120` | Frame count above which depth inference uses windowed mode |
| `LINGBOT_MAX_FRAMES` | `300` | Cap on frames fed to depth inference; long walkthroughs (1000+) OOM even windowed, so evenly subsample above this |
| `ENABLE_METRICANYTHING_FUSION` | `false` | Master switch for the optional MetricAnything densifier (also needs per-project opt-in) |
| `METRICANYTHING_MAX_FRAMES` | `200` | Evenly subsample above this many frames (one forward pass per frame) |
| `METRICANYTHING_VENDOR_DIR` | `/opt/metric-anything` | Vendored repo location (set by the worker image) |
| `ENABLE_HVAC_PLACEMENT` | `false` | Master switch for the HVAC stages (also needs per-project `hvac_mode`; indoor only) |
| `HVAC_UNIT_SIZE_CM` | `60x40` | Mounting rectangle searched for, W×H |
| `HVAC_MIN_CLEARANCE_CM` | `45` | Minimum gap to any fixture keep-out |
| `HVAC_MIN_WALL_INLIERS` / `HVAC_MIN_ADE20K_CONFIDENCE` | `2000` / `0.15` | Wall evidence floor / ADE20K wall-pixel exclusion |
| `HVAC_ENABLE_SAM2` | `true` | SAM2 mask refinement (falls back to GDINO boxes if off or it fails to load) |
| `HVAC_MAX_DETECTION_FRAMES` / `HVAC_MAX_WALL_SEG_FRAMES` | `150` / `40` | Frame caps for GDINO and ADE20K passes |
| `LINGBOT_CONF_PERCENTILE` | `40` | Drop this %% of lowest-confidence depth pixels before fusion |
