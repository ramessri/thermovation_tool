.PHONY: up up-d down logs shell-worker migrate build-gpu restart-gpu smoke test-scale \
        dev-infra dev-api dev-frontend dev-setup

COMPOSE      = docker compose
COMPOSE_FULL = docker compose                          # all services (prod-like)
COMPOSE_DEV  = docker compose -f docker-compose.infra.yml  # infra only (native dev)

# ── Full Docker stack (prod-like) ─────────────────────────────────────────────

## Start all services (API + frontend + worker in Docker)
up:
	$(COMPOSE_FULL) up --build

up-d:
	$(COMPOSE_FULL) up --build -d

down:
	$(COMPOSE_FULL) down

logs:
	$(COMPOSE_FULL) logs -f

shell-worker:
	$(COMPOSE_FULL) exec worker-gpu bash

## Run DB migrations via Docker API container
migrate:
	$(COMPOSE_FULL) exec api alembic upgrade head

## Build GPU worker image (slow — torch + pycolmap + lightglue)
build-gpu:
	$(COMPOSE_FULL) build worker-gpu

## Restart GPU worker after code changes (Celery does not hot-reload)
restart-gpu:
	$(COMPOSE_DEV) restart worker-gpu

# ── Native dev mode (recommended) ────────────────────────────────────────────
# postgres + redis + worker-gpu run in Docker.
# API and frontend run natively for instant code reload.

## One-time setup: create venv (Python 3.12) and install API dependencies
## Requires uv: curl -LsSf https://astral.sh/uv/install.sh | sh
dev-setup:
	uv venv --python 3.12 .venv
	uv pip install -r backend/requirements-api.txt
	@echo ""
	@echo "Also install system deps if not already present:"
	@echo "  Arch/Manjaro: sudo pacman -S perl-image-exiftool ffmpeg"
	@echo "  Ubuntu/Debian: sudo apt install libimage-exiftool-perl ffmpeg"
	@echo ""
	@echo "Then run:  make dev-infra     (in one terminal)"
	@echo "           make dev-api       (in another)"
	@echo "           make dev-frontend  (in another)"

## Start postgres + redis + GPU worker in Docker
dev-infra:
	$(COMPOSE_DEV) up -d
	@echo "Infra running. Postgres on :5433, Redis on :6379."
	@echo "Run migrations: make migrate-dev"

## Run DB migrations against the host-accessible postgres
migrate-dev:
	DATABASE_URL=postgresql+asyncpg://photogram:photogram@localhost:5433/photogram \
	  .venv/bin/alembic upgrade head

## Run FastAPI natively (instant reload on any .py change)
dev-api:
	PYTHONPATH=$(PWD) \
	  .venv/bin/uvicorn backend.main:app \
	  --reload \
	  --port 8000 \
	  --env-file .env.dev

## Run Next.js natively (true HMR — no rebuild ever needed)
dev-frontend:
	cd frontend && npm run dev

## Stop infra containers
dev-down:
	$(COMPOSE_DEV) down

# ── Testing & smoke ───────────────────────────────────────────────────────────

STAGE ?= all
smoke:
	$(COMPOSE_DEV) exec worker-gpu python /app/scripts/smoke.py --stage $(STAGE)

test-scale:
	$(COMPOSE_DEV) exec worker-gpu python -m pytest /app/tests/test_scale_from_aruco.py -v
