# Contributing to Photogram

Thanks for your interest in contributing. This file is the short version — for the full code guide, architecture context, and stage-by-stage pipeline internals see **[docs/contributing.md](docs/contributing.md)**.

---

## Quick path

1. **Fork & clone** — standard GitHub flow
2. **Spin up the dev stack** — `docker compose up -d` (see [docs/contributing.md § Dev setup](docs/contributing.md#dev-setup))
3. **Make your change** — backend edits require `docker compose restart worker-gpu`; API edits hot-reload automatically
4. **Smoke-test the affected stage** — `make smoke STAGE=<stage_name>`
5. **Open a PR** — describe what changed and why; link any related issue

---

## Important notes

- **Celery does not hot-reload.** After editing any `backend/` or `ml/` file, run `docker compose restart worker-gpu` before re-testing.
- **DB migrations go in `alembic/versions/`** — never edit the DB schema directly. See [docs/contributing.md § Migrations](docs/contributing.md#migrations).
- **Pipeline stage results are dicts passed forward through a Celery chain.** Every task must include all keys it received (use `result = dict(prev_result); result.update({...})`) so downstream stages don't lose data.
- **GPU task routing is via `task_routes` in `tasks.py`**, not `.set(queue=)` on signatures. The authoritative queue assignment lives in `celery_app.conf.task_routes`.

---

## Where things live

| What | Where |
|------|-------|
| Pipeline stage logic | `backend/workers/pipeline/<stage>.py` |
| Celery task wrappers | `backend/workers/tasks.py` |
| API routes | `backend/api/routes/` |
| DB models | `backend/models/models.py` |
| DB migrations | `alembic/versions/` |
| Frontend pages | `frontend/app/` |
| Frontend API client | `frontend/lib/api.ts` |
| Smoke test harness | `scripts/smoke.py` |

---

## Full contribution guide

[docs/contributing.md](docs/contributing.md) covers:
- Dev environment setup
- Adding a new pipeline stage end-to-end
- DB migration workflow
- Frontend component patterns
- Testing strategy
- Release checklist
