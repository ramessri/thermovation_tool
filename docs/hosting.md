# Hosting Guide

Self-hosting Photogram on a dedicated Linux machine with an NVIDIA GPU.

## Requirements

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| OS | Ubuntu 22.04 / Debian 12 | Any Linux with Docker support |
| GPU | NVIDIA RTX (12 GB VRAM+) | 12 GB minimum for Gaussian Splatting (object mode); 8 GB sufficient for indoor/outdoor only |
| RAM | 16 GB | 32 GB recommended for large scans |
| Disk | 100 GB free | LightGlue weights (~50 MB) + nerfstudio weights (~2 GB) + storage per scan (~2–8 GB each, Gaussian Splat adds ~300 MB per object scan) |
| Docker | 24+ with CDI support | `docker info | grep CDI` should list `nvidia.com/gpu` |
| NVIDIA driver | 530+ | `nvidia-smi` must show your GPU |

---

## 1 — Install prerequisites

```bash
# Docker (if not already installed)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER   # log out and back in

# NVIDIA Container Toolkit (enables GPU in Docker)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify CDI is registered
docker info | grep -i cdi
# Should show: nvidia.com/gpu=0  (or similar)
```

---

## 2 — Clone and configure

```bash
git clone https://github.com/N0t4R0b0t/photogram
cd photogram

cp .env.example .env
```

Edit `.env` — at minimum set these:

```bash
SECRET_KEY=<random 32+ char string>   # openssl rand -hex 32
CORS_ORIGINS=http://your-server-ip:3000
ANTHROPIC_API_KEY=                    # optional — for AI-assisted features
```

---

## 3 — Build and start

```bash
# Build all images (GPU worker takes 20–40 min on first run — compiles COLMAP from source)
docker compose -f docker-compose.prod.yml build

# Start everything
docker compose -f docker-compose.prod.yml up -d
```

Migrations run automatically on API startup. Watch logs to confirm:

```bash
docker compose -f docker-compose.prod.yml logs -f api
# Look for: INFO  [alembic.runtime.migration] Running upgrade ...
```

| Service | Default URL |
|---------|-------------|
| App | http://your-server:3000 |
| API + Swagger | http://your-server:8000/docs |

---

## 4 — Print ArUco markers

Every scan requires printed ArUco markers placed in the scene:

```bash
docker compose -f docker-compose.prod.yml exec worker-gpu \
  python /app/scripts/generate_aruco_sheet.py
# → samples/aruco_sheet.pdf
```

Print at 100% scale (no "fit to page"). Measure the printed marker side length and set it in `.env`:

```bash
ARUCO_MARKER_SIZE_M=0.15   # default 15 cm — adjust to your print
```

Place ≥ 2 markers where the camera will see them from multiple angles. See [operations.md](operations.md#aruco-scale) for placement guidance.

---

## 5 — Reverse proxy (optional but recommended)

If exposing over HTTPS or a custom domain, put Nginx or Caddy in front.

**Caddy** (simplest — handles TLS automatically):

```
your-domain.com {
    reverse_proxy localhost:3000
}

api.your-domain.com {
    reverse_proxy localhost:8000
}
```

**Nginx** (manual TLS via Certbot):

```nginx
server {
    listen 80;
    server_name your-domain.com;
    location / { proxy_pass http://localhost:3000; proxy_http_version 1.1; proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection "upgrade"; }
}

server {
    listen 80;
    server_name api.your-domain.com;
    location / { proxy_pass http://localhost:8000; proxy_http_version 1.1; proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection "upgrade"; }
}
```

Update `CORS_ORIGINS` in `.env` to match your domain, then restart the API:

```bash
docker compose -f docker-compose.prod.yml restart api
```

---

## Storage backends

Photogram supports several storage backends. Set `STORAGE_BACKEND` in `.env`:

| Backend | Value | When to use |
|---------|-------|-------------|
| Local disk | `local` | Single machine, simple setup |
| Synology WebDAV | `synology_webdav` | NAS on local network |
| S3-compatible | `synology_s3` | MinIO, Synology S3, Backblaze B2 |
| Filestack CDN | `filestack` | Distributed / CDN delivery |

See `.env.example` for the corresponding variables for each backend.

---

## Container images

All images are built locally from the repository. There are no pre-built images on Docker Hub.

| Image | Build time | Notes |
|-------|------------|-------|
| `photogram-api` | ~2 min | FastAPI + SQLAlchemy |
| `photogram-frontend` | ~3 min | Next.js standalone build |
| `photogram-worker-gpu` | **20–40 min** | Compiles COLMAP from source for your GPU's CUDA arch (`sm_75`, `sm_86`, `sm_89`) |

The GPU worker build is long because COLMAP must be compiled for your specific CUDA architecture. This happens once; subsequent `docker compose build` calls are incremental.

---

## Updating

```bash
git pull
docker compose -f docker-compose.prod.yml build api worker-gpu
docker compose -f docker-compose.prod.yml up -d
# Migrations run automatically on API restart
```

---

## Monitoring (optional)

Flower gives a real-time view of Celery task queues:

```bash
docker compose -f docker-compose.prod.yml --profile tools up -d flower
# → http://your-server:5555
```

Keep this off or firewalled in production — it exposes task metadata with no authentication by default.

---

## Troubleshooting

**GPU worker starts but CUDA is unavailable**
Run `docker info | grep -i cdi` on the host. If `nvidia.com/gpu` is missing, re-run `nvidia-ctk runtime configure` and restart Docker.

**API exits immediately on startup**
Check logs: `docker compose -f docker-compose.prod.yml logs api`. Usually a missing env var or Postgres not ready yet — the API will retry, but if Postgres is still initialising Docker may not wait long enough. Add `restart: on-failure` or a `healthcheck` to postgres if this is recurring.

**LightGlue / nerfstudio model weights re-download on every restart**
The `./models` directory is mounted into the worker. Confirm the volume mount is present in your compose file and `HF_HOME=/app/models/huggingface` is set in the worker environment. nerfstudio splatfacto weights (~2 GB) are downloaded on the first object scan.

**Scans never progress past `extract_frames`**
The GPU worker may not be running. Check: `docker compose -f docker-compose.prod.yml ps worker-gpu`. If it exited, check logs for CUDA or import errors.

**`worker-cpu` cannot run open3d tasks**
The optional `worker-cpu` container (profile `cpu-worker`) is missing `libgomp.so.1`, which open3d requires. Tasks like `refine_cloud`, `fill_planes`, `correct_trajectory_jumps`, and `coverage` will crash if routed to `worker-cpu`. Keep `worker-cpu` stopped unless you have rebuilt its image with `libgomp1` installed (`docker/Dockerfile.worker-cpu`). The `worker-gpu` container subscribes to both the `gpu` and `celery` queues and handles all stages on its own.

**Gaussian Splatting OOM on 8 GB GPU**
nerfstudio splatfacto requires ~8–10 GB VRAM during training. On an 8 GB card, COLMAP MVS and Gaussian Splatting cannot run simultaneously. The pipeline serialises them (`empty_cache` between stages) but peak usage may still exceed 8 GB. Set `GSPLAT_ITERATIONS=5000` in `.env` to reduce training memory at the cost of splat quality, or limit object scans to cards with ≥ 12 GB.
