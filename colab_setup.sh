#!/usr/bin/env bash
set -eo pipefail

# ==============================================================================
# MoEBoost - Google Colab Automated Pipeline
# 1. Check Hardware & Dependencies (apt, pip, cmake, CUDA)
# 2. Install & Compile (llama-cpp-python CUDA + MoE C++ Pager extension)
# 3. Model Selection & Fast Multi-threaded Download (including Flash-Next 4-bit)
# 4. Start MoEBoost Server Daemon (OpenAI compatible, port 4133)
# 5. Launch Interactive Python Chat with Real-Time Generation & Prefill tok/s
# ==============================================================================

BOLD="\033[1m"
GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
RESET="\033[0m"

echo -e "${CYAN}${BOLD}"
cat << 'BANNER'
   ____ ___  _        _    ____    ____       _               
  / ___/ _ \| |      / \  | __ )  / ___|  ___| |_ _   _ _ __  
 | |  | | | | |     / _ \ |  _ \  \___ \ / _ \ __| | | | '_ \ 
 | |__| |_| | |___ / ___ \| |_) |  ___) |  __/ |_| |_| | |_) |
  \____\___/|_____/_/   \_\____/  |____/ \___|\__|\__,_| .__/ 
                                                        |_|    
BANNER
echo -e "${RESET}"
echo -e "${BOLD}MoEBoost Automated Google Colab Installer & Launcher${RESET}"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# STEP 1: Hardware Inspection & CUDA Architecture Detection
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[Step 1/5] Checking Hardware & Dependencies...${RESET}"
CUDA_ARCH=""
if command -v nvidia-smi >/dev/null 2>&1; then
    echo -e "${CYAN}NVIDIA GPU Status:${RESET}"
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
    CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
    echo -e "${GREEN}✓ GPU Detected with Compute Capability: ${CUDA_ARCH}${RESET}"
else
    echo -e "${YELLOW}Warning: No NVIDIA GPU detected. Falling back to CPU backend.${RESET}"
fi

export DEBIAN_FRONTEND=noninteractive
echo "Updating apt repositories and checking C++ build tools..."
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq build-essential cmake python3-dev >/dev/null 2>&1 || true

echo "Checking Python package dependencies..."
pip install -q --upgrade pip
pip install -q huggingface_hub fastapi uvicorn psutil requests pydantic httpx pybind11 cmake

# ------------------------------------------------------------------------------
# STEP 2: Compilation (llama-cpp-python CUDA & MoE C++ Pager Accelerator)
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[Step 2/5] Compiling Runtime & MoE Accelerators...${RESET}"
RUNTIME_BACKEND="cpu"
if [ -n "$CUDA_ARCH" ]; then
    echo "Compiling CUDA-accelerated llama-cpp-python for SM_${CUDA_ARCH}..."
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}" FORCE_CMAKE=1 pip install -q llama-cpp-python --no-cache-dir || {
        echo -e "${YELLOW}Source compile busy, using prebuilt CUDA wheel...${RESET}"
        pip install -q llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 || pip install -q llama-cpp-python
    }
    RUNTIME_BACKEND="cuda"
else
    echo "Installing CPU llama-cpp-python..."
    pip install -q llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
fi

# Write runtime_backend.json so MoEBoost engine activates CUDA runtime
python3 - <<EOF
import json
with open("runtime_backend.json", "w") as f:
    json.dump({"backend": "$RUNTIME_BACKEND", "fallback_backends": ["cpu"]}, f)
print("Configured MoEBoost runtime backend:", "$RUNTIME_BACKEND")
EOF

echo "Compiling MoEBoost C++ Pager extension..."
mkdir -p build models
cmake -S . -B build -Dpybind11_DIR="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())' 2>/dev/null || echo '')" >/dev/null 2>&1 || true
cmake --build build --parallel 2 >/dev/null 2>&1 || true
if [ -f "build/moe_pager_cpp"*.so ]; then
    cp build/moe_pager_cpp*.so . 2>/dev/null || true
    echo -e "${GREEN}✓ MoEBoost C++ Pager accelerator successfully built.${RESET}"
fi

# ------------------------------------------------------------------------------
# STEP 3: Select and Download MoE Model (with Flash-Next UD-IQ4_XS)
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[Step 3/5] Select Mixture-of-Experts (MoE) Model:${RESET}"
cat << 'CATALOG'
  [1] Qwen3.8-Flash-Next (UD-IQ4_XS 4-bit)  - 125B+ Next-Gen MoE, 256K Context (~87.2 GB) [Recommended]
  [2] Qwen1.5-MoE-A2.7B-Chat (Q4_K_M)       - 14.3B Total / 2.7B Active (~9.2 GB, Ultra Fast)
  [3] DeepSeek-V2-Lite-Chat  (Q4_K_M)       - 16B Total / 2.4B Active (~10.1 GB, Math & Code)
  [4] Cerebras Qwen3-Coder-25B-A3B (IQ4_XS) - 25B Total / 3B Active (~15.0 GB, Large Coder MoE)
  [5] Mixtral-8x7B-Instruct-v0.1   (Q3_K_M) - 46.7B Total / 12.9B Active (~20.3 GB, Large MoE)
  [6] Custom Hugging Face GGUF repository
CATALOG

read -p "Select a model [1-6, Default: 1]: " MODEL_CHOICE
MODEL_CHOICE="${MODEL_CHOICE:-1}"

SHARDS=""
case "$MODEL_CHOICE" in
    1)
        REPO="unsloth/Qwen3.8-Flash-Next-GGUF"
        FILE="UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf"
        SHARDS="UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf,UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00004.gguf,UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00004.gguf,UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00004-of-00004.gguf"
        ;;
    2)
        REPO="Qwen/Qwen1.5-MoE-A2.7B-Chat-GGUF"
        FILE="qwen1_5-moe-a2_7b-chat-q4_k_m.gguf"
        ;;
    3)
        REPO="bartowski/DeepSeek-V2-Lite-Chat-GGUF"
        FILE="DeepSeek-V2-Lite-Chat-Q4_K_M.gguf"
        ;;
    4)
        REPO="Cerebras/Qwen3-Coder-REAP-25B-A3B-GGUF"
        FILE="cerebras_Qwen3-Coder-REAP-25B-A3B-IQ4_XS.gguf"
        ;;
    5)
        REPO="TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF"
        FILE="mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf"
        ;;
    6)
        read -p "Enter HF Repo ID: " REPO
        read -p "Enter GGUF Filename: " FILE
        ;;
    *)
        REPO="unsloth/Qwen3.8-Flash-Next-GGUF"
        FILE="UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf"
        SHARDS="UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf,UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00004.gguf,UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00004.gguf,UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00004-of-00004.gguf"
        ;;
esac

TARGET_MODEL="models/$FILE"
if [ -f "$TARGET_MODEL" ]; then
    echo -e "${GREEN}✓ Model already cached locally at $TARGET_MODEL!${RESET}"
else
    echo -e "\n${BOLD}Downloading weights from ${CYAN}$REPO${RESET} into Colab...${RESET}"
    if [ -n "$SHARDS" ]; then
        python3 - <<EOF
from huggingface_hub import hf_hub_download
import os

repo_id = "$REPO"
shards = "$SHARDS".split(",")
for idx, s in enumerate(shards, 1):
    print(f"[{idx}/{len(shards)}] Downloading {s}...")
    hf_hub_download(repo_id=repo_id, filename=s, local_dir="models")
print("All model shards downloaded successfully!")
EOF
    else
        python3 - <<EOF
from huggingface_hub import hf_hub_download
print("Downloading model weights...")
path = hf_hub_download(repo_id="$REPO", filename="$FILE", local_dir="models")
print("Done! Downloaded to:", path)
EOF
    fi
fi

# ------------------------------------------------------------------------------
# STEP 4: Start MoEBoost Server Daemon (Port 4133)
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[Step 4/5] Starting MoEBoost Server Daemon...${RESET}"
export MOE_MODELS_DIR="$(pwd)/models"
RELATIVE_MODEL_ID="$(python3 -c "import os; print(os.path.relpath('$TARGET_MODEL', 'models'))")"
export MOE_MODEL_FILE="$RELATIVE_MODEL_ID"
export MOE_PORT="4133"
export MOE_RUNTIME_BACKEND="$RUNTIME_BACKEND"
export MOE_GPU_LAYERS="${MOE_GPU_LAYERS:--1}"
export MOE_FLASH_ATTN="1"

# Terminate any previously running server on port 4133
if command -v fuser >/dev/null 2>&1; then
    fuser -k -9 4133/tcp >/dev/null 2>&1 || true
fi
pkill -9 -f uvicorn >/dev/null 2>&1 || true

python3 -m uvicorn server:app --host 127.0.0.1 --port 4133 > server.log 2>&1 &
SERVER_PID=$!
echo "Starting MoEBoost server daemon in background (PID: $SERVER_PID)..."

SERVER_ONLINE=0
for i in {1..35}; do
    if curl -s http://127.0.0.1:4133/api/telemetry >/dev/null 2>&1; then
        SERVER_ONLINE=1
        break
    fi
    sleep 1
done

if [ "$SERVER_ONLINE" -eq 1 ]; then
    echo -e "${GREEN}✓ MoEBoost server is ONLINE and responding at http://127.0.0.1:4133${RESET}"
    echo "Pre-loading model into GPU VRAM..."
    python3 - <<EOF
import urllib.request, json
try:
    payload = json.dumps({"model_id": "$RELATIVE_MODEL_ID"}).encode('utf-8')
    req = urllib.request.Request("http://127.0.0.1:4133/api/runtime/load", data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        print("✓ Model successfully loaded into VRAM! Active layers:", res.get("gpu_layers"), "| Backend:", res.get("backend"))
except Exception as e:
    print("Preload notice (will load on first message):", e)
EOF
else
    echo -e "${YELLOW}Server is still loading model weights into memory; continuing to chat interface...${RESET}"
fi

# ------------------------------------------------------------------------------
# STEP 5: Launch Python Chat with Generation & Prefill tok/s Telemetry
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[Step 5/5] Launching Interactive Python Chat with tok/s Telemetry...${RESET}"
echo "------------------------------------------------------------------------------"
exec python3 moeboost_chat.py --model "$RELATIVE_MODEL_ID"
