#!/usr/bin/env bash
set -eo pipefail

# MoEBoost - All-in-One Setup, Model Selection & Launcher
# Works seamlessly on local Linux machines and cloud environments like Google Colab.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BOLD="\033[1m"
GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
RESET="\033[0m"

echo -e "${CYAN}${BOLD}"
cat << 'BANNER'
  __  __       ______ ____                   _   
 |  \/  |     |  ____|  _ \                 | |  
 | \  / | ___ | |__  | |_) | ___   ___  ___ | |_ 
 | |\/| |/ _ \|  __| |  _ < / _ \ / _ \/ __|| __|
 | |  | | (_) | |____| |_) | (_) | (_) \__ \| |_ 
 |_|  |_|\___/|______|____/ \___/ \___/|___/ \__|
BANNER
echo -e "${RESET}"
echo -e "${BOLD}MoEBoost All-in-One Engine & Model Manager${RESET}"
echo "--------------------------------------------------------"

MODELS_DIR="${MOE_MODELS_DIR:-$SCRIPT_DIR/models}"
mkdir -p "$MODELS_DIR"

# ---------------------------------------------------------
# Step 1: Check / Setup Python Environment & Dependencies
# ---------------------------------------------------------
echo -e "\n${BOLD}[1/4] Checking MoEBoost Environment & Dependencies...${RESET}"

if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Running automated installation...${RESET}"
    chmod +x install.sh
    ./install.sh
else
    echo -e "${GREEN}✓ Virtual environment (.venv) detected.${RESET}"
fi

# Ensure huggingface_hub is available for fast multi-threaded downloads
if ! .venv/bin/python -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface_hub for fast direct downloads..."
    .venv/bin/pip install -q huggingface_hub
fi

# ---------------------------------------------------------
# Step 2: Model Catalog Definition
# ---------------------------------------------------------
declare -A MODEL_NAMES
declare -A MODEL_REPOS
declare -A MODEL_FILES
declare -A MODEL_SIZES
declare -A MODEL_DESCS

# 1. Qwen1.5-MoE-A2.7B
MODEL_NAMES[1]="Qwen1.5-MoE-A2.7B-Chat (Q4_K_M)"
MODEL_REPOS[1]="Qwen/Qwen1.5-MoE-A2.7B-Chat-GGUF"
MODEL_FILES[1]="qwen1_5-moe-a2_7b-chat-q4_k_m.gguf"
MODEL_SIZES[1]="~9.2 GB (14.3B Total / 2.7B Active)"
MODEL_DESCS[1]="Ultra-fast agentic & general coding. Runs efficiently on 8GB+ RAM / 3GB+ VRAM."

# 2. DeepSeek-V2-Lite-Chat
MODEL_NAMES[2]="DeepSeek-V2-Lite-Chat (Q4_K_M)"
MODEL_REPOS[2]="bartowski/DeepSeek-V2-Lite-Chat-GGUF"
MODEL_FILES[2]="DeepSeek-V2-Lite-Chat-Q4_K_M.gguf"
MODEL_SIZES[2]="~10.1 GB (16B Total / 2.4B Active)"
MODEL_DESCS[2]="Strong technical reasoning, math, and code generation with MoE dynamic sparsity."

# 3. Cerebras Qwen3-Coder-25B-A3B (Large MoE)
MODEL_NAMES[3]="Qwen3-Coder-REAP-25B-A3B (IQ4_XS) [Large MoE]"
MODEL_REPOS[3]="Cerebras/Qwen3-Coder-REAP-25B-A3B-GGUF"
MODEL_FILES[3]="cerebras_Qwen3-Coder-REAP-25B-A3B-IQ4_XS.gguf"
MODEL_SIZES[3]="~15.0 GB (25B Total / 3B Active)"
MODEL_DESCS[3]="High-performance 48-layer, 103-expert coding model. 100% passed 700-prompt agentic test."

# 4. Mixtral-8x7B-Instruct-v0.1 (Large MoE)
MODEL_NAMES[4]="Mixtral-8x7B-Instruct-v0.1 (Q3_K_M) [Large MoE]"
MODEL_REPOS[4]="TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF"
MODEL_FILES[4]="mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf"
MODEL_SIZES[4]="~20.3 GB (46.7B Total / 12.9B Active)"
MODEL_DESCS[4]="Classic 8-expert powerhouse model. Exceptional at creative and deep reasoning."

# 5. Qwen2.5-Coder-7B-Instruct (Dense Baseline)
MODEL_NAMES[5]="Qwen2.5-Coder-7B-Instruct (Q4_K_M) [Dense]"
MODEL_REPOS[5]="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
MODEL_FILES[5]="qwen2.5-coder-7b-instruct-q4_k_m.gguf"
MODEL_SIZES[5]="~4.7 GB (7B Total / 7B Active)"
MODEL_DESCS[5]="Standard high-accuracy coding model. Excellent baseline comparison."

# ---------------------------------------------------------
# Step 3: Scan Local Models & Interactive Model Selection
# ---------------------------------------------------------
echo -e "\n${BOLD}[2/4] Model Selection & Management${RESET}"

EXISTING_MODELS=()
while IFS= read -r -d $'\0' file; do
    EXISTING_MODELS+=("$file")
done < <(find "$MODELS_DIR" /media/local-ai/Model 2>/dev/null -maxdepth 2 -type f -name "*.gguf" -print0 2>/dev/null)

CHOSEN_MODEL_PATH=""

if [ ${#EXISTING_MODELS[@]} -gt 0 ]; then
    echo -e "\n${GREEN}Found ${#EXISTING_MODELS[@]} locally available GGUF model(s):${RESET}"
    idx=1
    for m in "${EXISTING_MODELS[@]}"; do
        base="$(basename "$m")"
        sz="$(du -h "$m" | cut -f1)"
        echo -e "  [L$idx] ${BOLD}$base${RESET} (${sz}) -> $m"
        ((idx++))
    done
fi

echo -e "\n${BOLD}Catalog of Curated MoE Models (Direct HF Download):${RESET}"
for i in 1 2 3 4 5; do
    echo -e "  [${CYAN}$i${RESET}] ${BOLD}${MODEL_NAMES[$i]}${RESET}"
    echo -e "      Size: ${YELLOW}${MODEL_SIZES[$i]}${RESET} | Repo: ${MODEL_REPOS[$i]}"
    echo -e "      ${MODEL_DESCS[$i]}"
done
echo -e "  [${CYAN}C${RESET}] Custom Hugging Face model or direct file path"

echo ""
read -p "Select a model option (e.g., L1, 1, 2, 3, 4, 5, or C) [Default: 1]: " USER_CHOICE
USER_CHOICE="${USER_CHOICE:-1}"

if [[ "$USER_CHOICE" =~ ^[Ll]([0-9]+)$ ]]; then
    sel_idx=$((${BASH_REMATCH[1]} - 1))
    if [ $sel_idx -ge 0 ] && [ $sel_idx -lt ${#EXISTING_MODELS[@]} ]; then
        CHOSEN_MODEL_PATH="${EXISTING_MODELS[$sel_idx]}"
        echo -e "${GREEN}Using existing local model:${RESET} $CHOSEN_MODEL_PATH"
    else
        echo -e "${RED}Invalid local model index. Defaulting to Option 1.${RESET}"
        USER_CHOICE="1"
    fi
fi

if [ -z "$CHOSEN_MODEL_PATH" ]; then
    case "$USER_CHOICE" in
        1|2|3|4|5)
            REPO="${MODEL_REPOS[$USER_CHOICE]}"
            FILENAME="${MODEL_FILES[$USER_CHOICE]}"
            TARGET_FILE="$MODELS_DIR/$FILENAME"

            if [ -f "$TARGET_FILE" ]; then
                echo -e "${GREEN}✓ Model already downloaded in $MODELS_DIR!${RESET}"
                CHOSEN_MODEL_PATH="$TARGET_FILE"
            else
                echo -e "\n${BOLD}[3/4] Downloading ${MODEL_NAMES[$USER_CHOICE]}...${RESET}"
                echo -e "Target repository: ${CYAN}$REPO${RESET}"
                echo -e "Target filename:   ${CYAN}$FILENAME${RESET}"
                echo -e "Downloading into:  ${CYAN}$TARGET_FILE${RESET}"

                .venv/bin/python - <<EOF
from huggingface_hub import hf_hub_download
import os

repo_id = "$REPO"
filename = "$FILENAME"
local_dir = "$MODELS_DIR"

print("Initiating download (supports auto-resume if interrupted)...")
try:
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    print("Download completed successfully:", path)
except Exception as e:
    print("Error during download:", e)
    exit(1)
EOF
                CHOSEN_MODEL_PATH="$TARGET_FILE"
            fi
            ;;
        [Cc]*)
            echo -e "\n${BOLD}Custom Model Configuration:${RESET}"
            read -p "Enter Hugging Face Repo ID (e.g. bartowski/DeepSeek-V2-Lite-Chat-GGUF) or local path: " CUSTOM_INPUT
            if [ -f "$CUSTOM_INPUT" ]; then
                CHOSEN_MODEL_PATH="$CUSTOM_INPUT"
            else
                read -p "Enter GGUF Filename in repo (e.g. DeepSeek-V2-Lite-Chat-Q4_K_M.gguf): " CUSTOM_FILENAME
                TARGET_FILE="$MODELS_DIR/$CUSTOM_FILENAME"
                .venv/bin/python - <<EOF
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="$CUSTOM_INPUT", filename="$CUSTOM_FILENAME", local_dir="$MODELS_DIR")
EOF
                CHOSEN_MODEL_PATH="$TARGET_FILE"
            fi
            ;;
        *)
            echo -e "${RED}Unrecognized choice. Exiting.${RESET}"
            exit 1
            ;;
    esac
fi

if [ ! -f "$CHOSEN_MODEL_PATH" ]; then
    echo -e "${RED}Error: Selected model path does not exist: $CHOSEN_MODEL_PATH${RESET}"
    exit 1
fi

# ---------------------------------------------------------
# Step 4: Launch Options (Daemon API Server vs Interactive Chat)
# ---------------------------------------------------------
echo -e "\n${BOLD}[4/4] Launch Mode${RESET}"
echo -e "  [${CYAN}1${RESET}] Start ${BOLD}MoEBoost API Server & Web UI${RESET} (Port 4133, OpenAI compatible)"
echo -e "  [${CYAN}2${RESET}] Start ${BOLD}Interactive CLI Chat${RESET} with real-time tok/s & TTFT telemetry"
echo -e "  [${CYAN}3${RESET}] Direct In-Process Python Model Loader test"

read -p "Select launch mode [Default: 1]: " LAUNCH_MODE
LAUNCH_MODE="${LAUNCH_MODE:-1}"

# Export the chosen model folder & model path
export MOE_MODELS_DIR="$(dirname "$CHOSEN_MODEL_PATH")"
export MOE_MODEL_FILE="$(basename "$CHOSEN_MODEL_PATH")"

case "$LAUNCH_MODE" in
    1)
        echo -e "\n${GREEN}Starting MoEBoost Daemon with model: $CHOSEN_MODEL_PATH...${RESET}"
        ./start.sh
        ;;
    2)
        # Check if server is running; if not, ask to start or use model loader
        if ! curl -s http://127.0.0.1:4133/v1/models >/dev/null 2>&1; then
            echo -e "${YELLOW}MoEBoost daemon is not currently running. Starting direct in-process chat...${RESET}"
            .venv/bin/python moeboost_model_loader.py --model "$CHOSEN_MODEL_PATH"
        else
            echo -e "${GREEN}Connecting to active MoEBoost daemon...${RESET}"
            .venv/bin/python moeboost_chat.py --model "$CHOSEN_MODEL_PATH"
        fi
        ;;
    3)
        echo -e "\n${GREEN}Running in-process LlamaCppRuntime loader...${RESET}"
        .venv/bin/python moeboost_model_loader.py --model "$CHOSEN_MODEL_PATH"
        ;;
    *)
        echo -e "${GREEN}Starting default MoEBoost server...${RESET}"
        ./start.sh
        ;;
esac
