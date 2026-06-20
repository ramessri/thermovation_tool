#!/usr/bin/env bash
# Overnight run: waits for worker-gpu image build, deploys, runs full e2e test.
# Usage: bash scripts/overnight_run.sh [build_pid]
set -euo pipefail

BUILD_PID="${1:-}"
LOG=/tmp/overnight_run.log
exec > >(tee -a "$LOG") 2>&1

echo ""
echo "════════════════════════════════════════════════════"
echo " Photogram overnight run — $(date)"
echo "════════════════════════════════════════════════════"

# ── 1. Wait for build ─────────────────────────────────────────────────────────
if [[ -n "$BUILD_PID" ]] && kill -0 "$BUILD_PID" 2>/dev/null; then
    echo "[1/4] Waiting for worker-gpu build (PID $BUILD_PID)…"
    # poll — 'wait' only works for children of this shell
    while kill -0 "$BUILD_PID" 2>/dev/null; do
        echo "      still building… $(date +%H:%M)"
        sleep 60
    done
    # Check whether the image was actually produced
    if docker image inspect photogram-worker-gpu:latest &>/dev/null; then
        echo "      Build finished — image present."
    else
        echo "      Build FAILED (image not found) — aborting."; exit 1
    fi
else
    echo "[1/4] No build PID given or build already done — proceeding."
fi

# Check build log for errors
if grep -qi "error\|failed" /tmp/worker_gpu_build.log 2>/dev/null | grep -v "warn"; then
    echo "      WARNING: build log contains errors — check /tmp/worker_gpu_build.log"
fi

# ── 2. Deploy rebuilt worker ──────────────────────────────────────────────────
echo "[2/4] Deploying rebuilt worker-gpu image…"
docker compose up -d worker-gpu
sleep 10

# Verify CUDA
if docker compose exec worker-gpu python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('CUDA OK:', torch.cuda.get_device_name(0))" 2>/dev/null; then
    echo "      GPU verified."
else
    echo "      WARNING: CUDA check failed — continuing anyway."
fi

# ── 3. Run full e2e test ──────────────────────────────────────────────────────
echo "[3/4] Launching e2e pipeline test…"
echo "      Start: $(date)"
python scripts/e2e_test.py --timeout 7200
STATUS=$?

echo ""
echo "[4/4] Test finished at $(date) — exit code: $STATUS"
if [[ $STATUS -eq 0 ]]; then
    echo "      ✓ PIPELINE COMPLETE — check the frontend for results."
    # Print the project URL from the test log
    grep "Frontend URL" /tmp/e2e_run.log 2>/dev/null || true
else
    echo "      ✗ PIPELINE FAILED — check /tmp/e2e_run.log for details."
fi

echo "════════════════════════════════════════════════════"
