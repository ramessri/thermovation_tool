# Code & Contribution Guide

**Start with [`CLAUDE.md`](../CLAUDE.md)** at the project root — it's the primary developer orientation document. It covers the full pipeline architecture, which files own what, infrastructure quirks (hot-reload behaviour, DB enum casing, worker-cpu limitations, scene_type propagation in reprocess chains), and debugging workflows. Read it before touching anything.

---

## Dev setup

**Prerequisites:** Docker with CUDA CDI support, a GPU with ≥ 12 GB VRAM, `docker compose` v2.

```bash
git clone https://github.com/N0t4R0b0t/photogram
cd photogram
cp .env.example .env          # adjust DATABASE_URL / REDIS_URL / storage if needed
docker compose up --build -d
docker compose exec api alembic upgrade head
```

**Verify everything is running:**
```
http://localhost:3000    — frontend
http://localhost:8000/docs — Swagger UI
http://localhost:5555    — Flower (Celery monitor)
```

### Hot-reload behaviour

| Container | Hot-reload? | Action needed after code change |
|-----------|-------------|--------------------------------|
| `api` | ✅ Yes (uvicorn `--reload`) | None — changes apply immediately |
| `frontend` | ❌ No (production build) | `docker compose build frontend && docker compose up -d frontend` |
| `worker-gpu` | ❌ No (Celery loads modules at startup) | `docker compose restart worker-gpu` |

**This is the most common pitfall.** If a backend fix doesn't seem to take effect, you almost certainly forgot to restart the worker.

---

## Repository structure

```
photogram/
├── backend/
│   ├── api/
│   │   └── routes/          # FastAPI routers (projects, anchors, etc.)
│   ├── core/
│   │   ├── config.py        # Settings (from env vars)
│   │   └── storage.py       # Storage backend abstraction
│   ├── models/
│   │   └── models.py        # SQLAlchemy ORM models
│   └── workers/
│       ├── tasks.py         # All Celery task definitions + chain launchers
│       └── pipeline/        # Stage implementations (one file per stage)
│           ├── extract_metadata.py
│           ├── extractor.py        # frame extraction
│           ├── aruco_detector.py   # pre-SfM ArUco detection
│           ├── aruco_sfm.py        # post-SfM ArUco re-scan
│           ├── matcher.py          # LightGlue feature matching
│           ├── sfm.py              # pycolmap SfM
│           ├── mvs.py              # COLMAP MVS
│           ├── scale_from_aruco.py      # scale derivation + plane fitting
│           ├── scout_calibrate.py       # scout calibration
│           ├── trajectory_correction.py # ICP correction for mis-registered blocks
│           ├── room_layout.py           # gravity, plane projection, void fill
│           ├── refine_cloud.py          # SOR + voxel downsample
│           ├── coverage.py              # HPR + DBSCAN coverage scoring
│           └── exporter.py              # PLY/OBJ/LAS export
├── frontend/
│   ├── app/
│   │   ├── page.tsx         # Project list
│   │   └── projects/[id]/page.tsx  # Project detail (pipeline progress, viewer)
│   ├── components/          # Shared UI components
│   └── lib/api.ts           # API client (fetch wrappers + WS)
├── alembic/
│   └── versions/            # DB migrations
├── scripts/
│   ├── smoke.py             # Per-stage smoke harness
│   ├── generate_aruco_sheet.py
│   └── inject_mock_aruco.py # Bypass ArUco for non-marker footage
├── tests/                   # Unit tests
├── docker/                  # Dockerfiles
├── docker-compose.yml
├── Makefile
└── docs/
```

---

## Adding a new pipeline stage

A stage is a pair: a **pipeline module** (`backend/workers/pipeline/<stage>.py`) and a **Celery task wrapper** in `backend/workers/tasks.py`.

### 1. Write the pipeline module

```python
# backend/workers/pipeline/my_stage.py
async def run_my_stage(
    project_id: str,
    prev_result: dict,
    tmp: Path,
    progress_cb: Callable[[float, str], None],
) -> dict:
    # ... do work ...
    progress_cb(1.0, "Done")

    result = dict(prev_result)   # ← always start from prev_result
    result["my_output_key"] = some_value
    return result
```

**Rules:**
- Always `result = dict(prev_result)` then update — never drop keys that came in
- Call `progress_cb(fraction, message)` regularly (0.0 → 1.0)
- Download inputs via `await storage.download(key, local_path)`, upload outputs via `await storage.upload(local_path, key)`
- Use `tmp / "filename"` for all local files — the task wrapper `shutil.rmtree`s tmp on completion

### 2. Write the Celery task wrapper

```python
@celery_app.task(bind=True, name="pipeline.my_stage")
def my_stage_task(self, prev_result: dict, project_id: str) -> dict:
    from backend.workers.pipeline.my_stage import run_my_stage

    job_id = self.request.id
    tmp    = job_tmp(job_id)
    t0     = _time.time()
    hb     = start_heartbeat(project_id, "my_stage")
    try:
        update_job_status(job_id, JobStatus.RUNNING, 0.0, "Starting…")
        result = asyncio.run(
            run_my_stage(
                project_id, prev_result, tmp,
                progress_cb=make_progress_cb(project_id, "my_stage", job_id),
            )
        )
        update_job_status(job_id, JobStatus.SUCCESS, 1.0, "Done")
        emit_stage_complete(project_id, "my_stage", t0, "summary string")
        return result
    except Exception as e:
        update_job_status(job_id, JobStatus.FAILED, error=str(e))
        raise
    finally:
        hb.set()
        shutil.rmtree(tmp, ignore_errors=True)
```

### 3. Add it to the chain in `launch_pipeline` / `launch_full_pipeline`

```python
tasks = [
    extract_metadata_task.s(project_id, storage_key),
    ...
    my_stage_task.s(project_id),   # ← add here
    ...
]
```

### 4. Route to the right queue

If the stage needs GPU:
```python
# In celery_app.conf.task_routes:
"pipeline.my_stage": {"queue": "gpu"},
```

CPU-only stages need no entry — they go to the default `celery` queue automatically.

### 5. Add the stage to the frontend STAGES list

```typescript
// frontend/app/projects/[id]/page.tsx
const STAGES = [
  ...
  { id: 'my_stage', label: 'My Stage Label', order: <N> },
];
```

### 6. Smoke-test it

Add a `run_my_stage` function to `scripts/smoke.py` following the existing pattern, then:
```bash
make smoke STAGE=my_stage
```

---

## DB migrations

The DB schema is managed by Alembic. **Never edit the schema directly** — it breaks reproducibility.

```bash
# Generate a migration after editing backend/models/models.py
docker compose exec api alembic revision --autogenerate -m "add my_column"

# Apply migrations
docker compose exec api alembic upgrade head

# Check current revision
docker compose exec api alembic current
```

Migration files go in `alembic/versions/`. Naming convention: `NNN_short_description.py` (e.g., `011_my_feature.py`).

**Gotcha:** Postgres enums are not auto-detected by Alembic's autogenerate. New enum types must be added manually to the migration. See existing migrations for the pattern.

---

## Working on the frontend

The frontend is a Next.js 15 app. To rebuild after changes:
```bash
docker compose build frontend && docker compose up -d frontend
```

For a faster dev loop, run Next.js locally outside Docker:
```bash
cd frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
# → http://localhost:3001 (or 3000 if not occupied)
```

**Key patterns:**

- API calls go through `frontend/lib/api.ts` — add new endpoints there
- `ProjectDetail` interface in `page.tsx` must match `ProjectResponse` in `backend/api/routes/projects.py`
- Pipeline progress is driven by WebSocket + project-status polling (every 10s). Do not use `progress >= 1.0` WS events to set `pipelineStatus` — individual stage completions trigger this, not whole-pipeline completion
- The `stageLog` Map is keyed by stage ID. When the scout → full transition is detected (via polling), the stage log is reset so the full pipeline renders cleanly

### Reconstruction viewer (`ResultsViewer.tsx`)

Tabbed 3D viewer for completed projects: Point Cloud, Mesh, Scene Overview,
Walkthrough, RGB Cloud, Re-shoot.

- `CameraPathViewer.tsx` — "Scene Overview": top-down camera trajectory, re-shoot
  and coverage-gap markers.
- `CameraWalkthroughViewer.tsx` — "Walkthrough": steps through ~9 evenly-spaced
  real SfM camera poses. Rotate-in-place free-look only (camera position never
  moves), FOV matched to each pose's real lens intrinsics, point cloud/mesh
  toggle, and a "Compare to photo" overlay that opacity-blends the actual
  extracted frame (`{project}/frames/{name}`) over the render for ground-truthing
  ambiguous geometry. Has a fullscreen toggle.
- `lib/poseMath.ts` — shared `cameras.json` pose helpers (`worldPosition`,
  `worldDirection`, `worldBasis`) used by both viewers above. Camera poses are
  in **unscaled SfM units** — multiply by `scaleFactor` and apply the loaded
  geometry's own `matrixWorld` (centre + COLMAP→Three.js X-flip) before placing
  a camera in the same scene space as the point cloud/mesh.

---

## Testing

**Unit tests:**
```bash
docker compose exec worker-gpu pytest tests/ -v
```

**Smoke harness (per stage):**
```bash
make smoke STAGE=all   # requires samples/20260520_195121.mp4
```

**Scale derivation tests:**
```bash
make test-scale
```

There are no automated integration tests covering the full pipeline end-to-end. Manual testing via the UI is the current approach. A smoke harness that runs all stages sequentially (`STAGE=all`) covers the happy path for the backend.

---

## PR workflow

1. Branch from `master` — use descriptive names (`fix/aruco-scale-fallback`, `feat/refine-cloud-stage`)
2. Keep PRs focused — one logical change per PR
3. Include a test: smoke harness, unit test, or manual reproduction steps in the PR description
4. Update `docs/` if the change affects pipeline behaviour, architecture, or contribution patterns
5. After merging backend changes that affect the worker, note in the PR description that `docker compose restart worker-gpu` is needed

---

## Common gotchas

| Symptom | Cause | Fix |
|---------|-------|-----|
| Code change has no effect on pipeline | Celery doesn't hot-reload | `docker compose restart worker-gpu` |
| DB column exists in model but not in DB | Migration not applied | `docker compose exec api alembic upgrade head` |
| Task fails with `KeyError` on prev_result | Upstream task dropped a key | Ensure all tasks do `result = dict(prev_result); result.update(...)` |
| Export uses wrong cloud (not refined) | `scaled_cloud_key` not updated | `refine_cloud` must update both `dense_cloud_key` and `scaled_cloud_key` |
| Stage log resets after scout phase | Mode transition not detected | Polling detects `pipeline_mode: scout→full` and resets `stageLog` |
| ArUco scale not derived | Only 1 marker per frame | Place 2+ markers close enough to share camera frames |
