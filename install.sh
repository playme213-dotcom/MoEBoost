#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

mkdir -p cpp_core engine models static

if [[ "${MOE_SKIP_APT:-0}" != "1" ]] && command -v apt-get >/dev/null 2>&1; then
  if [[ "${EUID}" -eq 0 ]]; then
    apt-get update
    apt-get install -y build-essential cmake python3-dev libpybind11-dev libvulkan-dev glslc spirv-headers
  elif command -v sudo >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y build-essential cmake python3-dev libpybind11-dev libvulkan-dev glslc spirv-headers
  else
    echo "No root/sudo access for Ubuntu build packages; using the local Python CMake/pybind11 wheels."
  fi
fi

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install pybind11 cmake
CUDA_AVAILABLE=0
if command -v nvcc >/dev/null 2>&1 && command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_AVAILABLE=1
fi
BUILD_CUDA="${MOE_BUILD_CUDA:-auto}"
BUILD_VULKAN="${MOE_BUILD_VULKAN:-auto}"
VULKAN_AVAILABLE=0
if [[ "$BUILD_VULKAN" != "0" ]] && command -v vulkaninfo >/dev/null 2>&1 \
   && command -v glslc >/dev/null 2>&1 && pkg-config --exists vulkan \
   && dpkg-query -W -f='${Status}' spirv-headers 2>/dev/null | grep -q 'install ok installed'; then
  VULKAN_AVAILABLE=1
fi
RUNTIME_FALLBACKS="cpu"
if [[ "$BUILD_CUDA" == "1" || ( "$BUILD_CUDA" == "auto" && "$CUDA_AVAILABLE" == "1" ) ]]; then
  if [[ "$CUDA_AVAILABLE" != "1" ]]; then
    echo "MOE_BUILD_CUDA=1 requested, but nvcc and an NVIDIA GPU were not both detected." >&2
    exit 1
  fi
  CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')"
  if [[ -n "$CUDA_ARCH" ]]; then
    RUNTIME_BACKEND="cuda"
    if [[ "$VULKAN_AVAILABLE" == "1" ]]; then
      RUNTIME_FALLBACKS="vulkan,cpu"
      export CMAKE_ARGS="-DGGML_CUDA=on -DGGML_VULKAN=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
      echo "Building CUDA first with Vulkan and CPU fallback backends."
    else
      export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
      echo "Building CUDA first with CPU fallback."
    fi
    export FORCE_CMAKE=1
    echo "Building in-process llama.cpp CUDA runtime for compute capability ${CUDA_ARCH}."
  else
    echo "Could not read NVIDIA compute capability; using CPU runtime."
    RUNTIME_BACKEND="cpu"
    export PIP_EXTRA_INDEX_URL="https://abetlen.github.io/llama-cpp-python/whl/cpu"
  fi
elif [[ "$VULKAN_AVAILABLE" == "1" ]]; then
  RUNTIME_BACKEND="vulkan"
  RUNTIME_FALLBACKS="cpu"
  export CMAKE_ARGS="-DGGML_VULKAN=on"
  export FORCE_CMAKE=1
  echo "Building in-process llama.cpp Vulkan runtime."
else
  RUNTIME_BACKEND="cpu"
  export PIP_EXTRA_INDEX_URL="https://abetlen.github.io/llama-cpp-python/whl/cpu"
  echo "Installing the prebuilt CPU llama.cpp runtime; no usable CUDA or Vulkan build toolchain was found."
fi
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
.venv/bin/python -m pip install -r requirements.txt
unset CMAKE_ARGS FORCE_CMAKE PIP_EXTRA_INDEX_URL
MOE_RUNTIME_BACKEND="$RUNTIME_BACKEND" MOE_RUNTIME_FALLBACKS="$RUNTIME_FALLBACKS" .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

Path("runtime_backend.json").write_text(json.dumps({
    "backend": os.environ["MOE_RUNTIME_BACKEND"],
    "fallback_backends": os.environ.get("MOE_RUNTIME_FALLBACKS", "cpu").split(","),
}) + "\n", encoding="utf-8")
PY

.venv/bin/cmake -S . -B build -Dpybind11_DIR="$ROOT/.venv/lib/python$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/pybind11/share/cmake/pybind11"
.venv/bin/cmake --build build --parallel 2
cp build/moe_pager_cpp*.so "$ROOT/"
cp build/moe_pager_cpp*.so .venv/lib/python$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/

cat <<'EOF'
Install complete. Start the OpenAI-compatible API with:
  MOE_MODELS_DIR=/path/to/gguf-folder .venv/bin/uvicorn server:app --host 127.0.0.1 --port 4133
EOF
