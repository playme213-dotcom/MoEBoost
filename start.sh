#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DEFAULT_MODELS_DIR="/media/local-ai/Model"
if [ ! -d "$DEFAULT_MODELS_DIR" ]; then
  DEFAULT_MODELS_DIR="${ROOT}/models"
fi
export MOE_MODELS_DIR="${MOE_MODELS_DIR:-$DEFAULT_MODELS_DIR}"
export MOE_THREADS="${MOE_THREADS:-$(nproc 2>/dev/null || echo 8)}"
export MOE_THREADS_BATCH="${MOE_THREADS_BATCH:-$(nproc 2>/dev/null || echo 8)}"
export MOE_RUNTIME_VERBOSE="${MOE_RUNTIME_VERBOSE:-0}"
export MOE_MAX_RAM="${MOE_MAX_RAM:-}"
export MOE_RAM_RESERVE_PERCENT="${MOE_RAM_RESERVE_PERCENT:-35}"
export MOE_WARM_FRACTION="${MOE_WARM_FRACTION:-0.0}"
export MOE_MAX_WARM_EXPERTS="${MOE_MAX_WARM_EXPERTS:-0}"
export MOE_GPU_LAYERS="${MOE_GPU_LAYERS:--1}"
export MOE_CONTEXT_SIZE="${MOE_CONTEXT_SIZE:-4096}"
export MOE_MIN_VRAM_CHAIRS="${MOE_MIN_VRAM_CHAIRS:-500}"
export MOE_UBATCH_SIZE="${MOE_UBATCH_SIZE:-512}"
export MOE_FLASH_ATTN="${MOE_FLASH_ATTN:-1}"

PORT="${MOE_PORT:-4133}"
HOST="${MOE_HOST:-127.0.0.1}"

if command -v fuser >/dev/null 2>&1 && fuser -s "${PORT}/tcp"; then
  echo "Stopping the process using TCP port ${PORT}..."
  fuser -k -TERM "${PORT}/tcp" >/dev/null 2>&1 || true
  for _ in {1..25}; do
    fuser -s "${PORT}/tcp" || break
    sleep 0.2
  done
  if fuser -s "${PORT}/tcp"; then
    echo "Port ${PORT} is still occupied; stopping without launching a second server." >&2
    exit 1
  fi
fi

exec .venv/bin/python -m uvicorn server:app --host "$HOST" --port "$PORT"
