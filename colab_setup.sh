#!/usr/bin/env bash
set -eo pipefail

# MoEBoost - One-Click Google Colab Setup & Large MoE Model Selection
# Designed to run in Google Colab (T4 / A100 / L4 / V100 GPU runtimes)

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
echo "--------------------------------------------------------"

# 1. Environment & GPU Detection
echo -e "\n${BOLD}[1/4] Inspecting Colab Hardware & CUDA GPU...${RESET}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
    CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
    echo -e "${GREEN}✓ GPU Detected with Compute Capability: ${CUDA_ARCH}${RESET}"
else
    echo -e "${YELLOW}Warning: No NVIDIA GPU detected. Inference will fall back to CPU.${RESET}"
    CUDA_ARCH=""
fi

# 2. System and Python package installations
echo -e "\n${BOLD}[2/4] Installing Required Libraries & Dependencies...${RESET}"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq build-essential cmake python3-dev >/dev/null 2>&1 || true

pip install -q --upgrade pip
pip install -q huggingface_hub fastapi uvicorn psutil requests pydantic httpx pybind11

# Install CUDA-accelerated llama-cpp-python
if [ -n "$CUDA_ARCH" ]; then
    echo "Installing CUDA-accelerated llama-cpp-python for compute architecture ${CUDA_ARCH}..."
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}" FORCE_CMAKE=1 pip install -q llama-cpp-python --no-cache-dir || {
        echo -e "${YELLOW}Prebuilt CUDA compilation failed, falling back to prebuilt wheel...${RESET}"
        pip install -q llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 || pip install -q llama-cpp-python
    }
else
    pip install -q llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
fi

# Build MoE C++ Pager Accelerator
echo "Compiling MoEBoost C++ Pager extension..."
mkdir -p build models
cmake -S . -B build -Dpybind11_DIR="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())' 2>/dev/null || echo '')" >/dev/null 2>&1 || true
cmake --build build --parallel 2 >/dev/null 2>&1 || true
if [ -f "build/moe_pager_cpp"*.so ]; then
    cp build/moe_pager_cpp*.so . 2>/dev/null || true
    echo -e "${GREEN}✓ MoEBoost C++ Pager accelerator compiled successfully.${RESET}"
fi

# 3. MoE Model Catalog with Large Models
echo -e "\n${BOLD}[3/4] Select a Mixture-of-Experts (MoE) Model to Download:${RESET}"

cat << 'CATALOG'
  [1] Qwen1.5-MoE-A2.7B-Chat (Q4_K_M) [~9.2 GB]  - 14.3B Total / 2.7B Active (Fast & Light)
  [2] DeepSeek-V2-Lite-Chat  (Q4_K_M) [~10.1 GB] - 16B Total / 2.4B Active (Math & Logic)
  [3] Cerebras Qwen3-Coder-25B-A3B    [~15.0 GB] - 25B Total / 3B Active (LARGE MoE - Coder)
  [4] Mixtral-8x7B-Instruct-v0.1      [~20.3 GB] - 46.7B Total / 12.9B Active (LARGE MoE)
  [5] Custom Hugging Face GGUF repository
CATALOG

read -p "Enter choice [1-5, Default: 1]: " MODEL_CHOICE
MODEL_CHOICE="${MODEL_CHOICE:-1}"

case "$MODEL_CHOICE" in
    1)
        REPO="Qwen/Qwen1.5-MoE-A2.7B-Chat-GGUF"
        FILE="qwen1_5-moe-a2_7b-chat-q4_k_m.gguf"
        ;;
    2)
        REPO="bartowski/DeepSeek-V2-Lite-Chat-GGUF"
        FILE="DeepSeek-V2-Lite-Chat-Q4_K_M.gguf"
        ;;
    3)
        REPO="Cerebras/Qwen3-Coder-REAP-25B-A3B-GGUF"
        FILE="cerebras_Qwen3-Coder-REAP-25B-A3B-IQ4_XS.gguf"
        ;;
    4)
        REPO="TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF"
        FILE="mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf"
        ;;
    5)
        read -p "Enter HF Repo ID (e.g. Qwen/Qwen1.5-MoE-A2.7B-Chat-GGUF): " REPO
        read -p "Enter GGUF Filename: " FILE
        ;;
    *)
        REPO="Qwen/Qwen1.5-MoE-A2.7B-Chat-GGUF"
        FILE="qwen1_5-moe-a2_7b-chat-q4_k_m.gguf"
        ;;
esac

TARGET_MODEL="models/$FILE"
if [ -f "$TARGET_MODEL" ]; then
    echo -e "${GREEN}✓ Model already downloaded in $TARGET_MODEL!${RESET}"
else
    echo -e "\n${BOLD}Downloading $FILE from $REPO into Colab...${RESET}"
    python3 - <<EOF
from huggingface_hub import hf_hub_download
import sys

print("Downloading model weights via multi-threaded downloader...")
path = hf_hub_download(repo_id="$REPO", filename="$FILE", local_dir="models")
print("Done! Downloaded to:", path)
EOF
fi

# 4. Launch MoEBoost Server / Chat
echo -e "\n${BOLD}[4/4] MoEBoost Launch Configuration${RESET}"
echo -e "  [1] Launch MoEBoost Daemon & Interactive Telemetry Chat (Recommended)"
echo -e "  [2] Launch OpenAI-compatible API Server (Port 4133)"
echo -e "  [3] In-Process Direct Python Model Loader"

read -p "Select Launch Mode [1-3, Default: 1]: " LAUNCH_MODE
LAUNCH_MODE="${LAUNCH_MODE:-1}"

export MOE_MODELS_DIR="$(pwd)/models"
export MOE_PORT="4133"

case "$LAUNCH_MODE" in
    1)
        echo -e "${GREEN}Starting background MoEBoost server...${RESET}"
        python3 -m uvicorn server:app --host 127.0.0.1 --port 4133 > server.log 2>&1 &
        SERVER_PID=$!
        echo "Waiting for server to initialize..."
        for _ in {1..30}; do
            if curl -s http://127.0.0.1:4133/v1/models >/dev/null 2>&1; then
                break
            fi
            sleep 1
        done
        echo -e "${GREEN}Server online (PID: $SERVER_PID)! Launching interactive CLI chat with tok/s telemetry...${RESET}"
        python3 moeboost_chat.py --model "$TARGET_MODEL"
        ;;
    2)
        echo -e "${GREEN}Starting API Server on http://127.0.0.1:4133...${RESET}"
        exec python3 -m uvicorn server:app --host 0.0.0.0 --port 4133
        ;;
    3)
        echo -e "${GREEN}Starting direct Python model loader...${RESET}"
        exec python3 moeboost_model_loader.py --model "$TARGET_MODEL"
        ;;
esac
