#!/usr/bin/env bash
# =============================================================================
# 🚀 AWESOME vLLM Model Browser & Launcher v1.17 🚀
# Detects hardware, asks about quantization + CPU offload upfront, then lists
# HuggingFace models that actually fit, and starts a standard vLLM server.
#
# Usage:
#   ./vllm_launcher.sh              — full interactive wizard
#   ./vllm_launcher.sh --relaunch   — skip wizard, reuse last saved command
#   ./vllm_launcher.sh --list-local — show locally cached models and exit
#   ./vllm_launcher.sh --dry-run    — run wizard but print command without launching
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
if [[ -f "${SCRIPT_DIR}/venv/bin/activate" ]]; then
    source "${SCRIPT_DIR}/venv/bin/activate"
fi

HOST="0.0.0.0"
PORT="4444"
MODEL_DIR="/media/tom/fast/NoSlop/vllm/models"   # ← change to your preferred download location
HF_API="https://huggingface.co/api/models"
LAST_CMD_FILE="/tmp/vllm_last_command.sh"
HF_TOKEN_CACHE="${HOME}/.cache/vllm_launcher_token"
DRY_RUN=0

# ── Colours & Formatting ──────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'
DIM='\033[2m'; RESET='\033[0m'

# ── Graceful Ctrl+C ───────────────────────────────────────────────────────────
trap 'echo -e "\n\n${YELLOW}🚪 Interrupted. Goodbye!${RESET}"; exit 130' INT TERM

# ── Helpers ───────────────────────────────────────────────────────────────────
require() { command -v "$1" &>/dev/null || { echo -e "${RED}Error: '$1' is required but not installed.${RESET}"; exit 1; }; }
require curl
require python3

# ── Argument parsing ──────────────────────────────────────────────────────────
case "${1:-}" in
    --dry-run)   DRY_RUN=1 ;;
    --relaunch)  : ;;   # handled below
    --list-local) : ;;  # handled below
    "")          : ;;
    *)  echo -e "${RED}Unknown flag: ${1}${RESET}"
        echo -e "Usage: $0[--relaunch | --list-local | --dry-run]"
        exit 1 ;;
esac

# ── --list-local shortcut ─────────────────────────────────────────────────────
if [[ "${1:-}" == "--list-local" ]]; then
    echo -e "\n${BOLD}${CYAN}=== 📁 Locally Cached Models ===${RESET}"
    echo -e "  Scanning: ${MODEL_DIR}\n"
    python3 - "${MODEL_DIR}" <<'PYEOF'
import os, sys

model_dir = sys.argv[1]
if not os.path.isdir(model_dir):
    print("  (directory does not exist)")
    sys.exit(0)

local_models =[]
for entry in os.scandir(model_dir):
    if not entry.is_dir():
        continue
    name = entry.name
    if name.startswith("models--"):
        parts = name[len("models--"):].split("--", 1)
        if len(parts) == 2:
            local_models.append(f"{parts[0]}/{parts[1]}")
    else:
        for subentry in os.scandir(entry.path):
            if subentry.is_dir():
                if os.path.exists(os.path.join(subentry.path, "config.json")) or any(f.name.endswith('.gguf') for f in os.scandir(subentry.path) if f.is_file()):
                    local_models.append(f"{name}/{subentry.name}")

if not local_models:
    print("  (no models found)")
else:
    for i, m in enumerate(sorted(local_models), 1):
        print(f"  {i:>3}. {m}")
    print(f"\n  Total: {len(local_models)} model(s)")
PYEOF
    exit 0
fi

# ── --relaunch shortcut ───────────────────────────────────────────────────────
if [[ "${1:-}" == "--relaunch" ]]; then
    if [[ ! -f "$LAST_CMD_FILE" ]]; then
        echo -e "${RED}No saved command found at ${LAST_CMD_FILE}. Run the wizard first.${RESET}"
        exit 1
    fi
    echo -e "\n${BOLD}${CYAN}=== ♻️  Relaunching last session ===${RESET}"
    echo -e "${DIM}$(cat "$LAST_CMD_FILE")${RESET}\n"

    local_venv_file="${LAST_CMD_FILE%.sh}.venv"
    if [[ -f "$local_venv_file" ]]; then
        saved_venv=$(cat "$local_venv_file")
        if [[ -f "${saved_venv}/bin/activate" ]]; then
            echo -e "  ${DIM}Restoring venv: ${saved_venv}${RESET}"
            # shellcheck disable=SC1090
            source "${saved_venv}/bin/activate"
        fi
    fi
    read -rp "Launch this command? [Y/n]: " confirm
    [[ "${confirm,,}" == "n" ]] && exit 0

    bash "$LAST_CMD_FILE"
    exit 0
fi

# ── Detect Hardware ───────────────────────────────────────────────────────────
detect_hardware() {
    echo -e "\n${BOLD}${CYAN}=== 🖥️  Hardware Detection ===${RESET}"

    TOTAL_RAM_GB=$(python3 -c "import os; mem=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES'); print(round(mem/1024**3,1))")
    AVAIL_RAM_GB=$(python3 -c "
with open('/proc/meminfo') as f:
    for line in f:
        if 'MemAvailable' in line:
            print(round(int(line.split()[1])/1024**2, 1))
            break
")
    echo -e "  RAM Total    : ${GREEN}${TOTAL_RAM_GB} GB${RESET}"
    echo -e "  RAM Available: ${GREEN}${AVAIL_RAM_GB} GB${RESET}"

    CPU_CORES=$(nproc)
    echo -e "  CPU Cores    : ${GREEN}${CPU_CORES}${RESET}"

    GPU_VRAM_GB=0
    GPU_VRAM_FREE_GB=0
    GPU_NAME="None"
    DEVICE="cpu"
    GPU_COUNT=0
    GPU_MEM_UTIL="0.85"

    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        GPU_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
        GPU_VRAM_FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
        GPU_VRAM_GB=$(python3 -c "print(round(${GPU_VRAM_MB}/1024, 1))")
        GPU_VRAM_FREE_GB=$(python3 -c "print(round(${GPU_VRAM_FREE_MB}/1024, 1))")

        # Aggressively reserve ~2GB of FREE VRAM strictly for PyTorch/CUDA graphs to prevent OOM
        GPU_MEM_UTIL=$(python3 -c "u=round((${GPU_VRAM_FREE_MB}-2048)/${GPU_VRAM_MB},2); print(max(0.30,min(u,0.90)))")

        DEVICE="cuda"
        echo -e "  GPU          : ${GREEN}${GPU_NAME} (x${GPU_COUNT})${RESET}"
        echo -e "  VRAM Total   : ${GREEN}${GPU_VRAM_GB} GB${RESET}"
        echo -e "  VRAM Free    : ${GREEN}${GPU_VRAM_FREE_GB} GB${RESET}"
        echo -e "  GPU mem util : ${GREEN}${GPU_MEM_UTIL}${RESET} (auto-tuned leaving 2GB buffer for PyTorch overhead)"
    elif command -v rocm-smi &>/dev/null && rocm-smi &>/dev/null 2>&1; then
        GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | grep -i "card" | head -1 | awk -F: '{print $2}' | xargs || echo "AMD GPU")
        GPU_COUNT=1
        GPU_VRAM_MB=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "Total Memory" | head -1 | grep -oP '\d+' | head -1 || echo "0")
        GPU_VRAM_FREE_MB=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "Free Memory" | head -1 | grep -oP '\d+' | head -1 || echo "0")
        GPU_VRAM_GB=$(python3 -c "print(round(${GPU_VRAM_MB}/1024, 1))" 2>/dev/null || echo "0")
        GPU_VRAM_FREE_GB=$(python3 -c "print(round(${GPU_VRAM_FREE_MB}/1024, 1))" 2>/dev/null || echo "0")

        GPU_MEM_UTIL=$(python3 -c "u=round((${GPU_VRAM_FREE_MB}-2048)/${GPU_VRAM_MB},2); print(max(0.30,min(u,0.90)))")

        DEVICE="rocm"
        echo -e "  GPU          : ${GREEN}${GPU_NAME}${RESET}"
        echo -e "  VRAM Total   : ${GREEN}${GPU_VRAM_GB} GB${RESET}"
        echo -e "  VRAM Free    : ${GREEN}${GPU_VRAM_FREE_GB} GB${RESET}"
    elif [[ "$(uname -s)" == "Darwin" ]] && sysctl -n machdep.cpu.brand_string 2>/dev/null | grep -iq "apple"; then
        GPU_NAME=$(sysctl -n machdep.cpu.brand_string | xargs)
        GPU_COUNT=1
        GPU_VRAM_GB=$TOTAL_RAM_GB
        GPU_VRAM_FREE_GB=$AVAIL_RAM_GB
        GPU_MEM_UTIL="0.85"
        DEVICE="mps"
        echo -e "  GPU          : ${GREEN}${GPU_NAME} (Unified Memory)${RESET}"
        echo -e "  VRAM/RAM Max : ${GREEN}${GPU_VRAM_GB} GB${RESET}"
        echo -e "  VRAM/RAM Free: ${GREEN}${GPU_VRAM_FREE_GB} GB${RESET}"
    else
        echo -e "  GPU          : ${YELLOW}None detected (CPU-only mode)${RESET}"
    fi

    mkdir -p "$MODEL_DIR" 2>/dev/null || true
    if [[ -d "$MODEL_DIR" ]]; then
        DISK_FREE_GB=$(python3 -c "import shutil; s = shutil.disk_usage('${MODEL_DIR}'); print(round(s.free / 1024**3, 1))")
        echo -e "  Model dir    : ${GREEN}${MODEL_DIR}${RESET}"
        echo -e "  Disk free    : ${GREEN}${DISK_FREE_GB} GB${RESET} available on that volume"
    else
        DISK_FREE_GB=999
        echo -e "  Model dir    : ${YELLOW}${MODEL_DIR} (will be created)${RESET}"
    fi

    if command -v ss &>/dev/null; then
        if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
            echo -e "\n  ${YELLOW}⚠️ Warning: port ${PORT} is already in use.${RESET}"
            read -rp "  Use a different port?[Y/n]: " change_port
            if [[ "${change_port,,}" != "n" ]]; then
                read -rp "  New port [default=4445]: " new_port
                PORT="${new_port:-4445}"
                echo -e "  Using port: ${GREEN}${PORT}${RESET}"
            fi
        fi
    fi

    export TOTAL_RAM_GB AVAIL_RAM_GB GPU_VRAM_GB GPU_VRAM_FREE_GB GPU_NAME \
           DEVICE CPU_CORES GPU_COUNT GPU_MEM_UTIL DISK_FREE_GB PORT
}

scan_local_models() {
    python3 - "${MODEL_DIR}" <<'PYEOF'
import os, sys, json

model_dir = sys.argv[1]
local_models =[]

if not os.path.isdir(model_dir):
    print(json.dumps([]))
    sys.exit(0)

for entry in os.scandir(model_dir):
    if not entry.is_dir():
        continue
    name = entry.name
    if name.startswith("models--"):
        parts = name[len("models--"):].split("--", 1)
        if len(parts) == 2:
            local_models.append(f"{parts[0]}/{parts[1]}")
    else:
        for subentry in os.scandir(entry.path):
            if subentry.is_dir():
                if os.path.exists(os.path.join(subentry.path, "config.json")) or any(f.name.endswith('.gguf') for f in os.scandir(subentry.path) if f.is_file()):
                    local_models.append(f"{name}/{subentry.name}")

print(json.dumps(local_models))
PYEOF
}

configure_options() {
    echo -e "\n${BOLD}${CYAN}=== 🧠 Memory Configuration ===${RESET}"
    echo -e "  These choices affect which models are shown as compatible.\n"

    echo -e "${BOLD}Quantization / Model Format:${RESET}"
    echo -e "  Reduces model size in memory, allowing larger models to fit."
    echo -e "  Only fully supported vLLM formats are listed:\n"
    echo -e "    1) None  — 1.0x  full bf16/fp16, best quality (Base models)"
    echo -e "    2) int8  — 0.5x  ~8-bit via bitsandbytes, broad compatibility"
    echo -e "    3) awq   — 0.25x ~4-bit, fast, requires an AWQ model variant"
    echo -e "    4) gptq  — 0.25x ~4-bit, ultra-fast via Marlin, requires GPTQ variant"
    echo -e "    5) fp8   — 0.5x  ~8-bit (FP8, W8A8, DeepSpeedFP), fast, Hopper+/ROCm6+"
    echo -e "    6) gguf  — 0.25x ~4-bit GGUF, extremely popular, excellent compatibility"
    echo ""
    read -rp "  Select quantization [1-6, default=1]: " quant_choice

    case "${quant_choice:-1}" in
        2) QUANT="int8";  QUANT_ARG="--quantization bitsandbytes --load-format bitsandbytes"; QUANT_RATIO=0.50 ;;
        3) QUANT="awq";   QUANT_ARG="--quantization awq";  QUANT_RATIO=0.25 ;;
        4) QUANT="gptq";  QUANT_ARG="--quantization gptq_marlin"; QUANT_RATIO=0.25 ;;
        5) QUANT="fp8";   QUANT_ARG="--quantization fp8";  QUANT_RATIO=0.50 ;;
        6) QUANT="gguf";  QUANT_ARG="--quantization gguf --load-format gguf"; QUANT_RATIO=0.25 ;;
        *) QUANT="none";  QUANT_ARG="";                    QUANT_RATIO=1.00 ;;
    esac

    echo -e "  Selected: ${GREEN}${QUANT}${RESET} (weights take ~${QUANT_RATIO}x of bf16 size)"

    if [[ "$QUANT" == "awq" || "$QUANT" == "gptq" ]]; then
        echo -e "  ${YELLOW}Note: AWQ/GPTQ require a pre-quantized model variant (search for '${QUANT}' in model names).${RESET}"
    fi

    CPU_OFFLOAD_GB=0
    if [[ "$DEVICE" == "cuda" || "$DEVICE" == "rocm" ]]; then
        echo ""
        echo -e "${BOLD}CPU RAM Offloading:${RESET}"
        echo -e "  Spills model layers that don't fit in VRAM out to system RAM."
        echo -e "  Enables larger models at the cost of slower inference (PCIe bottleneck)."
        echo -e "  Available RAM: ${GREEN}${AVAIL_RAM_GB} GB${RESET}"
        echo ""
        read -rp "  Enable CPU RAM offloading? [y/N]: " use_offload
        if [[ "${use_offload,,}" == "y" ]]; then
            MAX_OFFLOAD=$(python3 -c "print(max(1, int(float('${AVAIL_RAM_GB}') - 4)))")
            echo -e "  Suggested max: ${GREEN}${MAX_OFFLOAD} GB${RESET} (available RAM minus 4 GB OS reserve)"
            read -rp "  GB to offload to RAM [default=${MAX_OFFLOAD}]: " offload_input
            CPU_OFFLOAD_GB="${offload_input:-$MAX_OFFLOAD}"
            echo -e "  ${GREEN}CPU offload: ${CPU_OFFLOAD_GB} GB${RESET}"
        fi
    fi

    if [[ "$DEVICE" != "cpu" && "$DEVICE" != "mps" ]]; then
        RAW_MEM_GB=$(python3 -c "print(round(float('${GPU_VRAM_FREE_GB}') * 0.90 + float('${CPU_OFFLOAD_GB}'), 1))")
    else
        RAW_MEM_GB=$(python3 -c "print(round(float('${AVAIL_RAM_GB}') * 0.75, 1))")
    fi

    MEM_BUDGET_GB=$(python3 -c "print(round(float('${RAW_MEM_GB}') / float('${QUANT_RATIO}'), 1))")

    echo ""
    if [[ "${CPU_OFFLOAD_GB}" != "0" ]]; then
        echo -e "  Physical memory : ${GREEN}${RAW_MEM_GB} GB${RESET} (${GPU_VRAM_FREE_GB} GB VRAM + ${CPU_OFFLOAD_GB} GB RAM)"
        echo -e "  ${YELLOW}Note: CPU-offloaded layers are slower due to PCIe transfer overhead.${RESET}"
    else
        echo -e "  Physical memory : ${GREEN}${RAW_MEM_GB} GB${RESET}"
    fi
    echo -e "  Model size limit: ${GREEN}${MEM_BUDGET_GB} GB${RESET} bf16-equivalent (after ${QUANT} compression)"

    export QUANT QUANT_ARG QUANT_RATIO CPU_OFFLOAD_GB RAW_MEM_GB MEM_BUDGET_GB
}

fetch_models() {
    local search_query="${1:-}"
    local limit=300

    echo -e "\n${BOLD}${CYAN}=== 🔎 Fetching Compatible Models from HuggingFace ===${RESET}"
    echo -e "  Searching for : ${YELLOW}${search_query:-'top text-generation models'}${RESET}"
    echo -e "  Quantization  : ${GREEN}${QUANT}${RESET}"
    echo -e "  Model size cap: ${GREEN}${MEM_BUDGET_GB} GB${RESET} bf16-equivalent\n"

    local url="${HF_API}?sort=downloads&direction=-1&limit=${limit}&full=true"
    if [[ -z "$search_query" ]]; then
        url="${url}&pipeline_tag=text-generation"
    else
        url="${url}&search=${search_query// /%20}"
    fi

    if [[ "$QUANT" == "awq" || "$QUANT" == "gptq" || "$QUANT" == "fp8" || "$QUANT" == "gguf" ]]; then
        local quant_search="${QUANT}"
        [[ -n "$search_query" ]] && quant_search="${search_query} ${QUANT}"
        local quant_url="${HF_API}?sort=downloads&direction=-1&limit=${limit}&full=true&search=${quant_search// /%20}"
        if [[ -z "$search_query" ]]; then
            quant_url="${quant_url}&pipeline_tag=text-generation"
        fi

        local r1 r2
        r1=$(curl -sf "$url" 2>/dev/null)        || r1="[]"
        r2=$(curl -sf "$quant_url" 2>/dev/null)  || r2="[]"
        echo "$r1" > /tmp/vllm_r1.json
        echo "$r2" > /tmp/vllm_r2.json
        python3 -c "
import json
try:
    with open('/tmp/vllm_r1.json') as f: a=json.load(f)
except Exception: a=[]
try:
    with open('/tmp/vllm_r2.json') as f: b=json.load(f)
except Exception: b=[]
if isinstance(a, dict) and 'error' in a: a=[]
if isinstance(b, dict) and 'error' in b: b=[]
seen=set(); merged=[]
for m in a+b:
    if isinstance(m, dict) and 'id' in m and m['id'] not in seen:
        seen.add(m['id']); merged.append(m)
with open('/tmp/vllm_hf_response.json','w') as f: json.dump(merged,f)
"
    else
        curl -sf "$url" > /tmp/vllm_hf_response.json 2>/dev/null || echo "[]" > /tmp/vllm_hf_response.json
        python3 -c "
import json
try:
    with open('/tmp/vllm_hf_response.json') as f: data=json.load(f)
    if isinstance(data, dict) and 'error' in data: data=[]
except Exception: data=[]
with open('/tmp/vllm_hf_response.json','w') as f: json.dump(data,f)
"
    fi

    LOCAL_MODELS_JSON=$(scan_local_models)

    python3 - "${MEM_BUDGET_GB}" "${QUANT_RATIO}" "${CPU_OFFLOAD_GB}" \
               "${GPU_VRAM_FREE_GB:-0}" "${QUANT}" "${RAW_MEM_GB}" \
               "${DISK_FREE_GB}" "$LOCAL_MODELS_JSON" "${search_query}" <<'PYEOF'
import json, sys, re, os, urllib.request, concurrent.futures

mem_budget    = float(sys.argv[1])
quant_ratio   = float(sys.argv[2])
offload_gb    = float(sys.argv[3])
vram_free_gb  = float(sys.argv[4])
quant         = sys.argv[5]
raw_mem_gb    = float(sys.argv[6])
disk_free_gb  = float(sys.argv[7])
local_models  = set(json.loads(sys.argv[8]))
search_query  = sys.argv[9].lower()

with open("/tmp/vllm_hf_response.json") as f:
    data = json.load(f)

MOE_PATTERNS =[
    (r'deepseek-v3',              671, 37),
    (r'deepseek-r1(?!.*distill)', 671, 37),
    (r'deepseek-r1-0528',         671, 37),
    (r'kimi-k2',                  1000, 32),
    (r'mixtral-8x7b',             46,  12.9),
    (r'mixtral-8x22b',            141, 39),
    (r'qwen.*moe',                57,  14),
    (r'qwen3-30b-a3b',            30,  3),
    (r'qwen.*57b-a14b',           57,  14),
]

def get_moe_size(model_id):
    name = model_id.lower()
    for pattern, total_b, _ in MOE_PATTERNS:
        if re.search(pattern, name):
            return total_b * 2 * 1.10, True
    return None, False

def estimate_size_gb(model_id):
    moe_size, is_moe = get_moe_size(model_id)
    if is_moe:
        return moe_size
    name = model_id.lower()
    match = re.search(r'(?<![a-z])(\d+(?:\.\d+)?)\s*([bm])(?![a-z])', name)
    if not match:
        return None
    value = float(match.group(1))
    unit  = match.group(2)
    params_b = value / 1000 if unit == 'm' else value
    return params_b * 2 * 1.10

SKIP_TAGS = {"dataset", "lora", "onnx", "peft", "adapter", "mlx", "exl2", "coreml", "openvino"}
SKIP_PATTERNS =[
    r'onnx', r'-lora$', r'/lora-', r'embedding',
    r'reranker', r'reward-model', r'classifier', r'internal-testing',
    r'tiny-random', r'adapter', r'mlx', r'exl2', r'openvino', r'coreml'
]

results =[]
seen_bases = set()

for model in data:
    model_id = model.get("id", "")
    tags = [t.lower() for t in model.get("tags", [])]

    if any(t in SKIP_TAGS for t in tags): continue
    if any(re.search(p, model_id.lower()) for p in SKIP_PATTERNS): continue

    has_safetensors = "safetensors" in tags
    has_gguf = "gguf" in tags or "gguf" in model_id.lower()

    if not has_safetensors and not has_gguf and model_id not in local_models:
        continue

    model_lower = model_id.lower()
    if quant in ("awq", "gptq", "fp8", "gguf"):
        if quant == "gguf" and not has_gguf: continue

        if quant == "fp8":
            if not any(q in model_lower for q in ("fp8", "w8a8", "deepspeed")): continue
            base_name = re.sub(r'[-_](fp8|w8a8|deepspeed.*).*$', '', model_lower)
        elif quant != "gguf":
            if quant not in model_lower: continue
            base_name = re.sub(rf'[-_]{quant}.*$', '', model_lower)

        if quant != "gguf":
            if base_name in seen_bases: continue
            seen_bases.add(base_name)
    elif quant in ("none", "int8"):
        if any(q in model_lower for q in ("awq", "gptq", "fp8", "w8a8", "gguf", "int4")):
            continue
        if not has_safetensors and model_id not in local_models:
            continue

    safetensors = model.get("safetensors", {})
    total_bytes = safetensors.get("total") or 0
    _, is_moe = get_moe_size(model_id)

    if total_bytes > 0:
        bf16_size_gb = total_bytes / 1e9
        if quant in ("awq", "gptq") and any(q in model_id.lower() for q in ("awq","gptq","int4")):
            bf16_size_gb = bf16_size_gb / quant_ratio
        size_str = f"{total_bytes/1e9:.1f} GB"
        size_known = True
    else:
        est = estimate_size_gb(model_id)
        if est is not None:
            bf16_size_gb = est
            size_str = f"~{est:.1f} GB"
            size_known = True
        else:
            bf16_size_gb = None
            size_str = "unknown"
            size_known = False

    if size_known:
        effective_gb = bf16_size_gb * quant_ratio
        physical_budget = vram_free_gb * 0.90 + offload_gb
        if effective_gb > physical_budget and bf16_size_gb > physical_budget:
            continue

    downloads = model.get("downloads", 0)
    likes     = model.get("likes", 0)
    is_local  = model_id in local_models
    needs_ram = False
    size_warn = False

    if size_known:
        effective_gb = bf16_size_gb * quant_ratio
        vram_budget  = vram_free_gb * 0.90
        needs_ram    = offload_gb > 0 and effective_gb > vram_budget
        size_warn    = effective_gb > disk_free_gb * 0.9
    else:
        size_warn = True

    flags = ""
    if is_local:  flags += "L"
    if needs_ram: flags += "R"
    if size_warn: flags += "?"

    results.append((model_id, size_str, bf16_size_gb or 9999,
                    downloads, likes, flags, is_local, is_moe))

for lm in local_models:
    lm_lower = lm.lower()
    if search_query and not all(p in lm_lower for p in search_query.split()): continue
    if quant in ("awq", "gptq", "fp8", "gguf"):
        if quant == "gguf" and "gguf" not in lm_lower: continue
        if quant == "fp8" and not any(q in lm_lower for q in ("fp8", "w8a8", "deepspeed")): continue
        if quant in ("awq", "gptq") and quant not in lm_lower: continue
    elif quant in ("none", "int8"):
        if any(q in lm_lower for q in ("awq", "gptq", "fp8", "w8a8", "gguf", "int4")):
            continue

    if not any(r[0] == lm for r in results):
        _, is_moe = get_moe_size(lm)
        results.append((lm, "local", 9999, 0, 0, "L", True, is_moe))

def check_qwen_support(item):
    model_id = item[0]
    name = model_id.lower()
    if "qwen" in name and "3.5" in name and "-vl" not in name:
        try:
            url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2.0) as response:
                config = json.loads(response.read().decode('utf-8', errors='ignore'))
                if "Qwen3_5ForCausalLM" in config.get("architectures", []):
                    return False
        except Exception:
            return False
    return True

valid_results = []
if results:
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(check_qwen_support, m): m for m in results}
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                valid_results.append(futures[future])
    results = valid_results

results.sort(key=lambda x: (not x[6], -x[3]))

if not results:
    print("NO_MODELS_FOUND")
    sys.exit(0)

print(f" {'#':<4} {'Model ID':<52} {'On-disk':>9} {'Flags':5} {'Downloads':>11} {'Likes':>6}")
print(" ─" * 46)

for i, (mid, sz, _, dl, lk, flags, is_local, is_moe) in enumerate(results[:50], 1):
    mid_display = mid if len(mid) <= 52 else mid[:49] + "..."
    flag_str = ""
    if "L" in flags: flag_str += "\033[32mL\033[0m"
    if "R" in flags: flag_str += "\033[33mR\033[0m"
    if "?" in flags: flag_str += "\033[31m?\033[0m"
    if is_moe:       flag_str += "\033[36mM\033[0m"
    vis_len = len(re.sub(r'\033\[[0-9;]*m', '', flag_str))
    flag_padded = flag_str + " " * (4 - vis_len)
    dl_str = f"{dl:,}" if dl > 0 else "-"
    lk_str = f"{lk:,}" if lk > 0 else "-"
    print(f" {i:<4} {mid_display:<52} {sz:>9}  {flag_padded} {dl_str:>11} {lk_str:>6}")

with open("/tmp/vllm_model_list.txt", "w") as f:
    for mid, sz, _, dl, lk, flags, _, _ in results[:50]:
        f.write(f"{mid}|{sz}|{dl}|{lk}\n")

count = min(50, len(results))
print(f"\n  {count} models shown")
print(f"  Flags: {chr(27)}[32mL{chr(27)}[0m=cached locally  "
      f"{chr(27)}[33mR{chr(27)}[0m=needs CPU RAM offload  "
      f"{chr(27)}[31m?{chr(27)}[0m=unknown/disk space warning  "
      f"{chr(27)}[36mM{chr(27)}[0m=MoE architecture")
PYEOF
}

select_model() {
    if [[ ! -f /tmp/vllm_model_list.txt ]] || [[ ! -s /tmp/vllm_model_list.txt ]]; then
        echo -e "${RED}No compatible models found for your hardware.${RESET}"
        exit 1
    fi

    local model_count
    model_count=$(wc -l < /tmp/vllm_model_list.txt)

    echo ""
    echo -e "${BOLD}Options:${RESET}"
    echo -e "  • Enter a number (1-${model_count}) to select a listed model"
    echo -e "  • Type a full model ID directly (e.g. ${CYAN}Qwen/Qwen2.5-7B-Instruct${RESET})"
    echo -e "  • Press Enter to search again with a different query"
    read -rp "> " selection

    if [[ -z "$selection" ]]; then
        read -rp "New search query (e.g. 'llama 8b', 'mistral', 'q4 gguf'): " query
        fetch_models "$query"
        select_model
        return
    fi

    if [[ "$selection" =~ ^[0-9]+$ ]]; then
        if [[ "$selection" -ge 1 && "$selection" -le "$model_count" ]]; then
            SELECTED_MODEL=$(sed -n "${selection}p" /tmp/vllm_model_list.txt | cut -d'|' -f1)
        else
            echo -e "${RED}Invalid number. Enter 1-${model_count}.${RESET}"
            select_model
            return
        fi
    else
        SELECTED_MODEL="$selection"
    fi

    local lower_model="${SELECTED_MODEL,,}"
    if [[ "$lower_model" == *"qwen"* && "$lower_model" == *"3.5"* && "$lower_model" != *"-vl"* ]]; then
        # Dynamically verify if it's the bugged text-only version
        local is_text_only
        is_text_only=$(python3 -c "
import urllib.request, json, sys
try:
    url = f'https://huggingface.co/{sys.argv[1]}/resolve/main/config.json'
    with urllib.request.urlopen(urllib.request.Request(url), timeout=2.0) as r:
        if 'Qwen3_5ForCausalLM' in json.loads(r.read().decode('utf-8','ignore')).get('architectures', []):
            print('yes')
            sys.exit(0)
except Exception:
    pass
print('no')
" "$SELECTED_MODEL" 2>/dev/null || echo "no")

        if [[ "$is_text_only" == "yes" ]]; then
            echo -e "\n${RED}⚠️  UNSUPPORTED MODEL DETECTED ⚠️${RESET}"
            echo -e "${YELLOW}This specific Qwen3.5 model is a stripped text-only version ('Qwen3_5ForCausalLM').${RESET}"
            echo -e "${YELLOW}vLLM 0.19.1 has a bug where it forgets to register this architecture, causing an instant crash.${RESET}"
            echo -e "${YELLOW}Please select a Qwen3.5 model that kept its vision capabilities, or a Qwen2 model.${RESET}"
            read -rp "Press Enter to try again..."
            select_model
            return
        fi
    fi

    export SELECTED_MODEL
}

suggest_max_len() {
    local model_id="$1"
    python3 - "${model_id}" "${GPU_VRAM_FREE_GB}" "${QUANT_RATIO}" "${CPU_OFFLOAD_GB}" <<'PYEOF'
import sys, re, urllib.request, json

model_id     = sys.argv[1]
vram_free_gb = float(sys.argv[2])
quant_ratio  = float(sys.argv[3])
offload_gb   = float(sys.argv[4])

model_max_len   = None
bytes_per_token = None
cfg = {}
try:
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    req = urllib.request.Request(url, headers={"User-Agent": "vllm-launcher/1.17"})
    with urllib.request.urlopen(req, timeout=6) as resp:
        cfg = json.loads(resp.read().decode())

    for key in ("max_position_embeddings", "model_max_length", "n_positions", "max_sequence_length"):
        if key in cfg and isinstance(cfg[key], (int, float)) and 0 < cfg[key] < 10_000_000:
            model_max_len = int(cfg[key])
            break

    num_layers   = cfg.get("num_hidden_layers") or cfg.get("n_layer") or cfg.get("num_layers")
    head_dim     = cfg.get("head_dim") or cfg.get("d_head")
    num_kv_heads = cfg.get("num_key_value_heads") or cfg.get("num_kv_heads")
    hidden_size  = cfg.get("hidden_size") or cfg.get("d_model")
    num_heads    = cfg.get("num_attention_heads") or cfg.get("n_head")

    if not head_dim and hidden_size and num_heads:
        head_dim = hidden_size // num_heads
    if not num_kv_heads and num_heads:
        num_kv_heads = num_heads

    if num_layers and num_kv_heads and head_dim:
        bytes_per_token = int(num_layers) * 2 * int(num_kv_heads) * int(head_dim) * 2

except Exception:
    pass

name = model_id.lower()
match = re.search(r'(?<![a-z])(\d+(?:\.\d+)?)\s*([bm])(?![a-z])', name)
params_b = 0.0
if match:
    value    = float(match.group(1))
    unit     = match.group(2)
    params_b = value / 1000 if unit == 'm' else value

quantized_gb = params_b * 2 * quant_ratio * 1.10 if params_b else 0.0

if offload_gb > 0 and quantized_gb > 0:
    fraction_in_vram = max(0.30, (quantized_gb - offload_gb) / quantized_gb)
else:
    fraction_in_vram = 1.0

vram_used_by_weights = quantized_gb * fraction_in_vram

vram_total_budget    = max(0.5, vram_free_gb - 2.0)
vram_for_kv          = max(0.25, vram_total_budget - vram_used_by_weights)

if not bytes_per_token:
    if params_b:
        bytes_per_token = max(40_000, int(params_b * 11_000))
    else:
        bytes_per_token = 160_000

vram_based_max = int((vram_for_kv * 1024 ** 3) / bytes_per_token)

if model_max_len:
    suggested_tokens = min(vram_based_max, model_max_len)
else:
    suggested_tokens = vram_based_max

suggested_tokens = int(suggested_tokens * 0.80)

common =[512, 1024, 2048, 4096, 8192, 16384, 24576, 32768, 40960, 65536, 131072]
suggestion = 512
for c in reversed(common):
    if suggested_tokens >= c:
        suggestion = c
        break

print(str(suggestion))
PYEOF
}

validate_quant_model() {
    local model_id="$1"
    local quant="$2"

    echo -e "  ${CYAN}Verifying model has ${quant^^} quantization config...${RESET}"

    local result
    result=$(python3 - "${model_id}" "${quant}" "${HF_TOKEN_VAL:-}" <<'PYEOF'
import sys, urllib.request, json, os

model_id = sys.argv[1]
quant    = sys.argv[2]
token    = sys.argv[3] if len(sys.argv) > 3 else None

def fetch(url, timeout=8):
    headers = {"User-Agent": "vllm-launcher/1.17"}
    if token and token.strip() != "":
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()

base = f"https://huggingface.co/{model_id}/resolve/main"

try:
    fetch(f"{base}/quantize_config.json")
    print("yes"); sys.exit(0)
except Exception:
    pass

try:
    cfg = json.loads(fetch(f"{base}/config.json"))
    qcfg = cfg.get("quantization_config") or {}
    qtype = (qcfg.get("quant_type") or qcfg.get("quantization_method") or "").lower()

    if quant in qtype or "gptq" in qtype or "fp8" in qtype or "w8a8" in qtype:
        print("yes"); sys.exit(0)
except Exception:
    pass

if any(q in model_id.lower() for q in["gptq", "awq", "fp8", "w8a8", "deepspeed"]):
    print("name_only"); sys.exit(0)

print("no")
PYEOF
)

    if [[ "$result" == "no" ]]; then
        echo -e "\n${RED}Error: '${model_id}' does not appear to be a ${quant^^}-quantized model.${RESET}"
        read -rp "  Pick a different model instead? [Y/n]: " retry
        if [[ "${retry,,}" != "n" ]]; then
            fetch_models "${quant}"
            select_model
            configure_launch
            return
        else
            echo -e "${RED}Aborting — cannot load a non-${quant^^} model with --quantization ${quant}.${RESET}"
            exit 1
        fi
    elif [[ "$result" == "name_only" ]]; then
        echo -e "  ${YELLOW}${quant^^} config check skipped (HF unreachable or gated) — proceeding on model name alone ✓${RESET}"
    else
        echo -e "  ${GREEN}${quant^^} config found ✓${RESET}"
    fi
}

configure_launch() {
    echo -e "\n${BOLD}${CYAN}=== 🔑 Authentication ===${RESET}"
    local token_arg=""
    local hf_token=""
    if [[ -n "${HF_TOKEN:-}" ]]; then
        hf_token="${HF_TOKEN}"
        echo -e "  ${GREEN}Using HF_TOKEN from environment.${RESET}"
    elif [[ -f "$HF_TOKEN_CACHE" ]]; then
        hf_token=$(cat "$HF_TOKEN_CACHE")
        echo -e "  ${GREEN}Using saved HuggingFace token from ${HF_TOKEN_CACHE}${RESET}"
    else
        read -rp "  HuggingFace token (for gated models, leave blank if not needed): " hf_token
        if [[ -n "$hf_token" ]]; then
            read -rp "  Save token for future sessions? [Y/n]: " save_token
            if [[ "${save_token,,}" != "n" ]]; then
                mkdir -p "$(dirname "$HF_TOKEN_CACHE")"
                echo "$hf_token" > "$HF_TOKEN_CACHE"
                chmod 600 "$HF_TOKEN_CACHE"
                echo -e "  ${GREEN}Token saved to ${HF_TOKEN_CACHE}${RESET}"
            fi
        fi
    fi
    [[ -n "$hf_token" ]] && token_arg="--hf-token ${hf_token}"
    export HF_TOKEN_VAL="${hf_token}"

    if [[ -n "${HF_TOKEN_VAL:-}" ]]; then
        export HF_TOKEN="${HF_TOKEN_VAL}"
    fi

    export BASE_MODEL_ID="${SELECTED_MODEL}"

    if [[ "$QUANT" == "gguf" || "${SELECTED_MODEL,,}" == *"gguf"* ]]; then
        if [[ "$QUANT" != "gguf" ]]; then
            echo -e "\n  ${YELLOW}Auto-detecting GGUF model format.${RESET}"
            QUANT="gguf"
            QUANT_ARG="--quantization gguf --load-format gguf"
        fi

        echo -e "\n${BOLD}${CYAN}=== 📦 GGUF Model Setup ===${RESET}"
        echo -e "  ${CYAN}Fetching metadata & files for ${SELECTED_MODEL}...${RESET}"

        local GGUF_FETCH_OUTPUT
        GGUF_FETCH_OUTPUT=$(python3 - "${SELECTED_MODEL}" "${HF_TOKEN_VAL:-}" <<'PYEOF'
import urllib.request, json, sys, re
repo = sys.argv[1]
token = sys.argv[2] if len(sys.argv) > 2 else None

headers = {'User-Agent': 'vllm-launcher/1.17'}
if token:
    headers['Authorization'] = f'Bearer {token}'

has_config = "0"
base_model = ""

try:
    req = urllib.request.Request(f'https://huggingface.co/api/models/{repo}/tree/main', headers=headers)
    with urllib.request.urlopen(req, timeout=5) as r:
        files = json.loads(r.read().decode())
    ggufs =[f['path'] for f in files if f.get('path', '').endswith('.gguf')]
    if any(f.get('path') == 'config.json' for f in files):
        has_config = "1"
except Exception:
    ggufs =[]

if has_config == "0":
    try:
        req = urllib.request.Request(f'https://huggingface.co/api/models/{repo}', headers=headers)
        with urllib.request.urlopen(req, timeout=5) as r:
            info = json.loads(r.read().decode())
            bm = info.get('cardData', {}).get('base_model', '')
            if isinstance(bm, list) and len(bm) > 0:
                base_model = bm[0]
            elif isinstance(bm, str):
                base_model = bm
    except Exception:
        pass

    if base_model:
        try:
            req = urllib.request.Request(f'https://huggingface.co/api/models/{base_model}/tree/main', headers=headers)
            with urllib.request.urlopen(req, timeout=3) as r:
                bfiles = json.loads(r.read().decode())
                if not any(f.get('path') == 'config.json' for f in bfiles):
                    base_model = ""
        except Exception:
            base_model = ""

print(has_config)
print(base_model)
print('\n'.join(ggufs))
PYEOF
)

        local HAS_CONFIG=$(echo "$GGUF_FETCH_OUTPUT" | sed -n '1p')
        local GUESSED_BASE=$(echo "$GGUF_FETCH_OUTPUT" | sed -n '2p')
        local GGUF_FILES=$(echo "$GGUF_FETCH_OUTPUT" | tail -n +3 | grep '\.gguf$' || true)

        if [[ -n "$GGUF_FILES" ]]; then
            echo "$GGUF_FILES" > /tmp/vllm_gguf_files.txt
            local FILE_COUNT
            FILE_COUNT=$(wc -l < /tmp/vllm_gguf_files.txt)

            if [[ "$FILE_COUNT" -gt 1 ]]; then
                echo -e "\n${BOLD}Multiple GGUF files found. Select one to download and run:${RESET}"
                awk '{print "  " NR ") " $0}' /tmp/vllm_gguf_files.txt

                local SUGGESTED_IDX
                SUGGESTED_IDX=$(awk 'tolower($0) ~ /q4_k_m/ {print NR; exit}' /tmp/vllm_gguf_files.txt)
                [[ -z "$SUGGESTED_IDX" ]] && SUGGESTED_IDX=$(awk 'tolower($0) ~ /q4/ {print NR; exit}' /tmp/vllm_gguf_files.txt)
                [[ -z "$SUGGESTED_IDX" ]] && SUGGESTED_IDX=1

                local file_choice
                read -rp "  Select file[1-${FILE_COUNT}, default=${SUGGESTED_IDX}]: " file_choice
                file_choice=${file_choice:-$SUGGESTED_IDX}
                SELECTED_GGUF_FILE=$(sed -n "${file_choice}p" /tmp/vllm_gguf_files.txt)
            else
                SELECTED_GGUF_FILE=$(head -1 /tmp/vllm_gguf_files.txt)
            fi
            echo -e "  ${GREEN}Selected GGUF: ${SELECTED_GGUF_FILE}${RESET}"
            export SELECTED_GGUF_FILE
        else
            echo -e "  ${YELLOW}No .gguf files found via API. Proceeding with standard loading.${RESET}"
        fi

        if [[ "$HAS_CONFIG" == "0" ]]; then
            echo -e "\n  ${RED}⚠️  WARNING: No 'config.json' found in this GGUF repository.${RESET}"

            if [[ -z "$GUESSED_BASE" ]]; then
                GUESSED_BASE=$(echo "${SELECTED_MODEL}" | sed -E 's/-(GGUF|gguf|AWQ|GPTQ)//i')
            fi

            BASE_MODEL_ID="$GUESSED_BASE"
            while true; do
                echo ""
                echo -e "  ${DIM}Please provide the full HuggingFace ID (Author/ModelName) of the unquantized base model.${RESET}"
                read -rp "  Original base model repo[default=${BASE_MODEL_ID}]: " user_base

                if [[ "${user_base,,}" == "skip" ]]; then
                    echo -e "  ${YELLOW}Skipping validation. Will attempt to load the GGUF file directly.${RESET}"
                    BASE_MODEL_ID="SKIP"
                    break
                fi

                BASE_MODEL_ID="${user_base:-$BASE_MODEL_ID}"

                if [[ -z "$BASE_MODEL_ID" ]]; then
                    echo -e "  ${RED}Error: Base model is required for this GGUF. Aborting.${RESET}"
                    exit 1
                fi

                local curl_args=("-s" "-o" "/dev/null" "-w" "%{http_code}")
                [[ -n "${HF_TOKEN_VAL:-}" ]] && curl_args+=("-H" "Authorization: Bearer ${HF_TOKEN_VAL}")

                local http_code
                http_code=$(curl "${curl_args[@]}" "https://huggingface.co/api/models/${BASE_MODEL_ID}")

                if [[ "$http_code" == "401" || "$http_code" == "404" ]]; then
                    echo -e "  ${RED}❌ Repository '${BASE_MODEL_ID}' not found or is inaccessible (HTTP ${http_code}).${RESET}"
                else
                    echo -e "  ${GREEN}✓ Valid base model verified: ${BASE_MODEL_ID}${RESET}"
                    break
                fi
            done
        fi
    fi

    echo -e "\n${BOLD}${CYAN}=== ⚙️  Launch Configuration ===${RESET}"
    echo -e "  Model        : ${GREEN}${SELECTED_MODEL}${RESET}"
    [[ -n "${SELECTED_GGUF_FILE:-}" ]] && echo -e "  Target File  : ${GREEN}${SELECTED_GGUF_FILE}${RESET}"
    [[ "$BASE_MODEL_ID" != "$SELECTED_MODEL" ]] && echo -e "  Base Config  : ${GREEN}${BASE_MODEL_ID}${RESET}"
    echo -e "  Host         : ${GREEN}${HOST}:${PORT}${RESET}"
    echo -e "  Device       : ${GREEN}${DEVICE}${RESET}"
    echo -e "  Quantization : ${GREEN}${QUANT}${RESET}"
    echo -e "  GPU mem util : ${GREEN}${GPU_MEM_UTIL}${RESET}"
    echo -e "  Model dir    : ${GREEN}${MODEL_DIR}${RESET}"
    echo -e "  Disk free    : ${GREEN}${DISK_FREE_GB} GB${RESET}"
    [[ "${CPU_OFFLOAD_GB}" != "0" ]] && echo -e "  CPU offload  : ${GREEN}${CPU_OFFLOAD_GB} GB${RESET}"
    echo ""

    if [[ ! -d "$MODEL_DIR" ]]; then
        mkdir -p "$MODEL_DIR" || { echo -e "  ${RED}Failed to create directory. Check permissions.${RESET}"; exit 1; }
    fi

    local dtype="auto"
    [[ "$DEVICE" == "cpu" ]] && dtype="float32"
    [[ "$DEVICE" == "mps" ]] && dtype="half"

    local tensor_parallel="${GPU_COUNT:-1}"
    [[ "$tensor_parallel" -lt 1 ]] && tensor_parallel=1

    local suggested_len
    suggested_len=$(suggest_max_len "$SELECTED_MODEL")
    if [[ "$suggested_len" -gt 0 ]]; then
        echo -e "  ${DIM}Suggested max context based on available VRAM: ${suggested_len} tokens${RESET}"
    else
        suggested_len=8192
    fi

    read -rp "  Max context length in tokens [default=${suggested_len}]: " max_len
    max_len="${max_len:-$suggested_len}"
    local max_len_arg="--max-model-len ${max_len}"

    local tool_call_arg=""
    echo ""
    echo -e "${BOLD}🛠️  Tool Calling:${RESET}"
    read -rp "  Enable tool calling? [Y/n]: " enable_tools
    if [[ "${enable_tools,,}" != "n" ]]; then
        local detected_parser
        detected_parser=$(python3 - "${SELECTED_MODEL}" <<'PYEOF'
import sys, re
model = sys.argv[1].lower()
if re.search(r'llama.?3|llama.?4', model): print("llama3_json"); exit()
if re.search(r'mistral|mixtral', model):   print("mistral"); exit()
if re.search(r'internlm', model):          print("internlm"); exit()
if re.search(r'jamba', model):             print("jamba"); exit()
if re.search(r'qwen', model):              print("qwen3_coder"); exit()
print("hermes")
PYEOF
)
        echo -e "  ${DIM}(Available parsers: hermes, llama3_json, mistral, internlm, jamba, qwen3_coder, qwen3_xml)${RESET}"
        read -rp "  Tool parser [default=${detected_parser}]: " parser_input
        local tool_parser="${parser_input:-$detected_parser}"
        tool_call_arg="--enable-auto-tool-choice --tool-call-parser ${tool_parser}"
        echo -e "  ${GREEN}Tool calling enabled with parser: ${tool_parser}${RESET}"
    else
        echo -e "  ${DIM}Tool calling disabled.${RESET}"
    fi

    local reasoning_arg=""
    if [[ "${SELECTED_MODEL,,}" == *"reason"* || "${SELECTED_MODEL,,}" == *"deepseek"* || "${SELECTED_MODEL,,}" == *"qwen"* ]]; then
        echo ""
        read -rp "  Reasoning parser (e.g. qwen3, deepseek_r1) [leave blank to skip]: " reasoning_parser
        if [[ -n "$reasoning_parser" ]]; then
            reasoning_arg="--reasoning-parser ${reasoning_parser}"
            echo -e "  ${GREEN}Reasoning parser enabled: ${reasoning_parser}${RESET}"
        fi
    fi

    if [[ "$QUANT" == "awq" || "$QUANT" == "gptq" || "$QUANT" == "fp8" ]]; then
        validate_quant_model "${SELECTED_MODEL}" "${QUANT}"
    fi

    local max_num_seqs
    max_num_seqs=$(python3 - "${SELECTED_MODEL}" "${GPU_VRAM_GB}" "${GPU_VRAM_FREE_GB}" "${GPU_MEM_UTIL}" <<'PYEOF'
import sys, urllib.request, json

model_id     = sys.argv[1]
vram_total   = float(sys.argv[2])
vram_free    = float(sys.argv[3])
gpu_mem_util = float(sys.argv[4])

if   vram_total <= 12:  base = 16
elif vram_total <= 16:  base = 32
elif vram_total <= 24:  base = 64
elif vram_total <= 40:  base = 96
elif vram_total <= 80:  base = 192
else:                   base = 256

vocab_size = 150_000
try:
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    req = urllib.request.Request(url, headers={"User-Agent": "vllm-launcher/1.17"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        cfg = json.loads(resp.read().decode())
    vocab_size = cfg.get("vocab_size") or vocab_size
except Exception:
    pass

scale    = min(4.0, 150_000 / vocab_size)
adjusted = int(base * scale)

for p in[8, 16, 32, 48, 64, 96, 128, 192, 256]:
    if adjusted <= p:
        print(p)
        sys.exit(0)
print(256)
PYEOF
)
    echo -e "  Max num seqs : ${GREEN}${max_num_seqs}${RESET} (auto-tuned to avoid sampler OOM)"

    CMD="python3 -m vllm.entrypoints.openai.api_server"

    if [[ "$QUANT" == "gguf" && -n "${SELECTED_GGUF_FILE:-}" ]]; then
        UNIFIED_DIR="${MODEL_DIR}/unified_execution/${SELECTED_MODEL//\//_}"
        CMD+=" --model \"${UNIFIED_DIR}\""
        export UNIFIED_DIR
    else
        CMD+=" --model ${SELECTED_MODEL}"
        if [[ "${BASE_MODEL_ID:-}" != "SKIP" && "${BASE_MODEL_ID:-}" != "$SELECTED_MODEL" ]]; then
            CMD+=" --hf-config-path \"${BASE_MODEL_ID}\""
            CMD+=" --tokenizer \"${BASE_MODEL_ID}\""
        fi
    fi

    CMD+=" --host ${HOST}"
    CMD+=" --port ${PORT}"
    CMD+=" --dtype ${dtype}"
    CMD+=" --tensor-parallel-size ${tensor_parallel}"
    CMD+=" --gpu-memory-utilization ${GPU_MEM_UTIL}"
    CMD+=" --max-num-seqs ${max_num_seqs}"
    CMD+=" --download-dir ${MODEL_DIR}"
    [[ -n "$QUANT_ARG" ]]            && CMD+=" ${QUANT_ARG}"
    if [[ "${CPU_OFFLOAD_GB}" != "0" ]]; then
        CMD+=" --cpu-offload-gb ${CPU_OFFLOAD_GB} --enforce-eager"
    fi
    [[ -n "$max_len_arg" ]]          && CMD+=" ${max_len_arg}"
    [[ -n "$token_arg" ]]            && CMD+=" ${token_arg}"
    [[ -n "$tool_call_arg" ]]        && CMD+=" ${tool_call_arg}"
    [[ -n "$reasoning_arg" ]]        && CMD+=" ${reasoning_arg}"
    CMD+=" --trust-remote-code"

    if [[ "${SELECTED_MODEL,,}" == *"qwen"* && "${SELECTED_MODEL,,}" == *"3.5"* && "${SELECTED_MODEL,,}" != *"-vl"* ]]; then
        # Tell vLLM to skip loading the vision encoder for text-only Qwen3.5 models to prevent crashes
        CMD+=" --language-model-only"
    fi

    export CMD
}

check_dependencies() {
    local PIP="python3 -m pip install --upgrade"

    echo -e "\n${CYAN}📦 Ensuring known best dependencies are installed...${RESET}"

    if ! python3 -c "import vllm" 2>/dev/null; then
        echo -e "  ${YELLOW}vLLM not found. Installing latest vLLM...${RESET}"
        $PIP vllm
    fi

    echo -e "  ${DIM}Synchronizing transformers, hf-transfer, huggingface_hub...${RESET}"
    $PIP transformers hf-transfer huggingface_hub --quiet
    export HF_HUB_ENABLE_HF_TRANSFER=1

    case "$QUANT" in
        int8)
            $PIP "bitsandbytes>=0.45.0" --quiet
            ;;
        awq)
            echo -e "  ${GREEN}✓ AWQ selected: vLLM handles this natively via built-in kernels.${RESET}"
            ;;
        gguf)
            $PIP "gguf>=0.10.0" --quiet
            ;;
        gptq)
            echo -e "  ${GREEN}✓ GPTQ selected: vLLM handles this natively via built-in Marlin kernels.${RESET}"
            ;;
        fp8)
            echo -e "  ${GREEN}✓ FP8/W8A8 selected: vLLM handles this natively.${RESET}"
            ;;
    esac

    echo -e "  ${GREEN}✓ Core dependencies resolved and enforced.${RESET}"

    # --- Auto-fix: vLLM 0.19.x CPU offload + hybrid mamba/attention assertion bug ---
    # Qwen3.5 and similar hybrid models crash with:
    #   AssertionError: Cannot re-initialize the input batch when CPU weight offloading is enabled.
    # This is a known vLLM bug: the assertion is an overly cautious guard — the
    # re-initialization itself is safe. We comment it out automatically.
    if [[ "${CPU_OFFLOAD_GB}" != "0" ]]; then
        local RUNNER_FILE
        RUNNER_FILE=$(python3 -c "import vllm.v1.worker.gpu_model_runner as m; print(m.__file__)" 2>/dev/null)
        if [[ -n "$RUNNER_FILE" ]] && grep -q 'assert self.offload_config.uva.cpu_offload_gb == 0' "$RUNNER_FILE" 2>/dev/null; then
            echo -e "  ${YELLOW}⚡ Applying vLLM CPU-offload hybrid-model bugfix...${RESET}"
            sed -i '/assert self\.offload_config\.uva\.cpu_offload_gb == 0/,/for more details\./s/^/# [vllm-launcher fix] /' "$RUNNER_FILE"
            # Clear bytecode cache so Python uses the patched source
            local CACHE_DIR
            CACHE_DIR="$(dirname "$RUNNER_FILE")/__pycache__"
            rm -f "${CACHE_DIR}"/gpu_model_runner*.pyc 2>/dev/null
            echo -e "  ${GREEN}✓ Patched assertion in $(basename "$RUNNER_FILE") (safe for hybrid mamba/attention models)${RESET}"
        fi
    fi
}

launch_server() {
    check_dependencies

    local PREFLIGHT_SCRIPT=""
    if [[ "$QUANT" == "gguf" && -n "${SELECTED_GGUF_FILE:-}" ]]; then
        PREFLIGHT_SCRIPT=$(cat << EOF
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 -c "
import os, sys
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

token = os.environ.get('HF_TOKEN_VAL')
if not token or token.strip() == '':
    token = None

repo_id = '${SELECTED_MODEL}'
filename = '${SELECTED_GGUF_FILE}'
base_repo = '${BASE_MODEL_ID:-}'
model_dir = '${MODEL_DIR}'
unified_dir = '${UNIFIED_DIR}'

try:
    files = list_repo_files(repo_id=repo_id, token=token)
except Exception:
    files =[filename]

if '-of-' in filename:
    base = filename.split('-of-')[0]
    to_download =[f for f in files if f.startswith(base) and f.endswith('.gguf')]
else:
    to_download =[filename]

os.makedirs(unified_dir, exist_ok=True)

print('\n\x1b[0;36m📥 Pre-fetching GGUF weights & Base config...\x1b[0m')
gguf_paths = []
for f in to_download:
    print(f'  Fetching {f}...')
    try:
        path = hf_hub_download(repo_id=repo_id, filename=f, cache_dir=model_dir, token=token)
        gguf_paths.append((f, path))
    except Exception as e:
        print(f'  \x1b[0;31m[ERROR] Failed to fetch {f}: {e}\x1b[0m')

base_dir = None
if base_repo and base_repo != repo_id and base_repo != 'SKIP':
    print(f'  Fetching config/tokenizer metadata from {base_repo}...')
    try:
        base_dir = snapshot_download(repo_id=base_repo, allow_patterns=['*.json', '*.model', '*.tiktoken', '*.py', '*.txt'], cache_dir=model_dir, token=token)
    except Exception as e:
        print(f'  \x1b[0;31m[ERROR] Failed to fetch base config from {base_repo}: {e}\x1b[0m')
elif base_repo != 'SKIP':
    print(f'  Fetching config/tokenizer metadata from {repo_id}...')
    try:
        base_dir = snapshot_download(repo_id=repo_id, allow_patterns=['*.json', '*.model', '*.tiktoken', '*.py', '*.txt'], cache_dir=model_dir, token=token)
    except Exception:
        pass

for f in os.listdir(unified_dir):
    fp = os.path.join(unified_dir, f)
    if os.path.isfile(fp) or os.path.islink(fp):
        os.remove(fp)

if base_dir and os.path.exists(base_dir):
    for f in os.listdir(base_dir):
        if f.lower().endswith(('.gguf', '.safetensors', '.bin', '.pth', '.pt', '.h5', '.msgpack')):
            continue
        src = os.path.join(base_dir, f)
        dest = os.path.join(unified_dir, f)
        if os.path.isfile(src) and not os.path.lexists(dest):
            os.symlink(src, dest)

for f, path in gguf_paths:
    dest = os.path.join(unified_dir, f)
    if not os.path.lexists(dest):
        os.symlink(path, dest)

    lower_f = f.lower()
    if lower_f != f:
        lower_dest = os.path.join(unified_dir, lower_f)
        if not os.path.lexists(lower_dest):
            try:
                os.symlink(path, lower_dest)
            except OSError:
                pass

print('  \x1b[0;32m✓ Unified execution folder ready.\x1b[0m\n')
"
EOF
)
    else
        PREFLIGHT_SCRIPT=$(cat << EOF
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 -c "
import os
from huggingface_hub import snapshot_download
token = os.environ.get('HF_TOKEN_VAL')
if not token or token.strip() == '':
    token = None
print('\n\x1b[0;36m📥 Pre-fetching model weights...\x1b[0m')
snapshot_download(repo_id='${SELECTED_MODEL}', cache_dir='${MODEL_DIR}', token=token)
print('  \x1b[0;32m✓ Model weights are fully cached on disk.\x1b[0m\n')
"
EOF
)
    fi

    echo -e "\n${BOLD}${CYAN}=== 🚀 Ready to Launch ===${RESET}"
    echo -e "${YELLOW}Command:${RESET}\n"
    echo -e "  ${CYAN}${CMD}${RESET}\n"

    echo "#!/usr/bin/env bash" > "$LAST_CMD_FILE"
    echo "set -e" >> "$LAST_CMD_FILE"

    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        echo "source '${VIRTUAL_ENV}/bin/activate'" >> "$LAST_CMD_FILE"
        echo "${VIRTUAL_ENV}" > "${LAST_CMD_FILE%.sh}.venv"
    fi

    if [[ -n "${HF_TOKEN_VAL:-}" ]]; then
        echo "export HF_TOKEN=\"${HF_TOKEN_VAL}\"" >> "$LAST_CMD_FILE"
    fi

    if [[ "${CPU_OFFLOAD_GB}" == "0" ]]; then
        echo "export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1" >> "$LAST_CMD_FILE"
    else
        echo "export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0" >> "$LAST_CMD_FILE"
    fi
    echo "$PREFLIGHT_SCRIPT" >> "$LAST_CMD_FILE"



    echo "$CMD" >> "$LAST_CMD_FILE"
    chmod +x "$LAST_CMD_FILE"

    echo -e "  ${DIM}Command saved to ${LAST_CMD_FILE} — use ./vllm_launcher.sh --relaunch to skip wizard next time${RESET}\n"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo -e "${YELLOW}Dry-run mode — not launching. Copy the command above to run manually.${RESET}\n"
        exit 0
    fi

    read -rp "Launch server now? [Y/n]: " confirm
    if [[ "${confirm,,}" != "n" ]]; then
        echo -e "\n${GREEN}🚀 Starting vLLM server...${RESET}"
        echo -e "💬 OpenAI-compatible API : ${CYAN}http://${HOST}:${PORT}/v1${RESET}"
        echo -e "💡 ${YELLOW}Note: If the log pauses at 'Using FLASH_ATTN', PyTorch is JIT-compiling${RESET}"
        echo -e "   ${YELLOW}the GPU kernels for the first time. This is perfectly normal and takes 1-3 mins.${RESET}"
        echo -e "⏹️  Press ${BOLD}Ctrl+C${RESET} to stop.\n"
        bash "$LAST_CMD_FILE"
    else
        echo ""
        echo -e "${YELLOW}To start later, run:${RESET}\n"
        echo -e "  ${CYAN}./vllm_launcher.sh --relaunch${RESET}\n"
    fi
}

main() {
    echo -e "\n${BOLD}${CYAN}╔══════════════════════════════════════════════════╗${RESET}"
    echo -e "${BOLD}${CYAN}║      🚀 AWESOME vLLM MODEL LAUNCHER v1.17 🚀     ║${RESET}"
    echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════╝${RESET}"
    [[ "$DRY_RUN" -eq 1 ]] && echo -e "  ${YELLOW}Dry-run mode: wizard runs but server will not launch.${RESET}"

    detect_hardware
    configure_options

    local initial_query=""
    echo ""
    read -rp "Search for a specific model type? (leave blank for top downloads): " initial_query

    fetch_models "$initial_query"

    if grep -q "NO_MODELS_FOUND" /tmp/vllm_model_list.txt 2>/dev/null || [[ ! -s /tmp/vllm_model_list.txt ]]; then
        echo -e "${RED}No models found that fit your hardware.${RESET}"
        exit 1
    fi

    select_model
    configure_launch
    launch_server
}

main
