#!/usr/bin/env bash
# =============================================================================
# vLLM Model Browser & Launcher
# Detects hardware, asks about quantization + CPU offload upfront, then lists
# HuggingFace models that actually fit, and starts a vLLM server on 0.0.0.0:4444.
#
# Usage:
#   ./vllm_launcher.sh              — full interactive wizard
#   ./vllm_launcher.sh --relaunch   — skip wizard, reuse last saved command
# =============================================================================

set -euo pipefail

HOST="0.0.0.0"
PORT="4444"
MODEL_DIR="/path/to/folder"   # ← change to your preferred download location
HF_API="https://huggingface.co/api/models"
LAST_CMD_FILE="/tmp/vllm_last_command.sh"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'

# ── Helpers ───────────────────────────────────────────────────────────────────
require() { command -v "$1" &>/dev/null || { echo -e "${RED}Error: '$1' is required but not installed.${RESET}"; exit 1; }; }
require curl
require python3

# ── --relaunch shortcut ───────────────────────────────────────────────────────
if [[ "${1:-}" == "--relaunch" ]]; then
    if [[ ! -f "$LAST_CMD_FILE" ]]; then
        echo -e "${RED}No saved command found at ${LAST_CMD_FILE}. Run the wizard first.${RESET}"
        exit 1
    fi
    echo -e "\n${BOLD}${CYAN}=== Relaunching last session ===${RESET}"
    echo -e "${DIM}$(cat "$LAST_CMD_FILE")${RESET}\n"
    read -rp "Launch this command? [Y/n]: " confirm
    [[ "${confirm,,}" == "n" ]] && exit 0
    echo -e "\n${GREEN}Starting vLLM server...${RESET}"
    echo -e "OpenAI-compatible API : ${CYAN}http://${HOST}:${PORT}/v1${RESET}"
    echo -e "Press ${BOLD}Ctrl+C${RESET} to stop.\n"
    bash "$LAST_CMD_FILE"
    exit 0
fi

# ── Detect Hardware ───────────────────────────────────────────────────────────
detect_hardware() {
    echo -e "\n${BOLD}${CYAN}=== Hardware Detection ===${RESET}"

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
        GPU_MEM_UTIL=$(python3 -c "u=round((${GPU_VRAM_FREE_MB}-300)/${GPU_VRAM_MB},2); print(max(0.5,min(u,0.95)))")
        DEVICE="cuda"
        echo -e "  GPU          : ${GREEN}${GPU_NAME} (x${GPU_COUNT})${RESET}"
        echo -e "  VRAM Total   : ${GREEN}${GPU_VRAM_GB} GB${RESET}"
        echo -e "  VRAM Free    : ${GREEN}${GPU_VRAM_FREE_GB} GB${RESET}"
        echo -e "  GPU mem util : ${GREEN}${GPU_MEM_UTIL}${RESET} (auto-tuned to free VRAM)"
    elif command -v rocm-smi &>/dev/null && rocm-smi &>/dev/null 2>&1; then
        GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | grep -i "card" | head -1 | awk -F: '{print $2}' | xargs || echo "AMD GPU")
        GPU_COUNT=1
        GPU_VRAM_MB=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "Total Memory" | head -1 | grep -oP '\d+' | head -1 || echo "0")
        GPU_VRAM_FREE_MB=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "Free Memory" | head -1 | grep -oP '\d+' | head -1 || echo "0")
        GPU_VRAM_GB=$(python3 -c "print(round(${GPU_VRAM_MB}/1024, 1))" 2>/dev/null || echo "0")
        GPU_VRAM_FREE_GB=$(python3 -c "print(round(${GPU_VRAM_FREE_MB}/1024, 1))" 2>/dev/null || echo "0")
        GPU_MEM_UTIL=$(python3 -c "u=round((${GPU_VRAM_FREE_MB}-300)/${GPU_VRAM_MB},2); print(max(0.5,min(u,0.95)))")
        DEVICE="rocm"
        echo -e "  GPU          : ${GREEN}${GPU_NAME}${RESET}"
        echo -e "  VRAM Total   : ${GREEN}${GPU_VRAM_GB} GB${RESET}"
        echo -e "  VRAM Free    : ${GREEN}${GPU_VRAM_FREE_GB} GB${RESET}"
    else
        echo -e "  GPU          : ${YELLOW}None detected (CPU-only mode)${RESET}"
    fi

    # ── Check disk space on MODEL_DIR ─────────────────────────────────────────
    mkdir -p "$MODEL_DIR" 2>/dev/null || true
    if [[ -d "$MODEL_DIR" ]]; then
        DISK_FREE_GB=$(python3 -c "
import shutil
s = shutil.disk_usage('${MODEL_DIR}')
print(round(s.free / 1024**3, 1))
")
        DISK_USED_GB=$(python3 -c "
import shutil
s = shutil.disk_usage('${MODEL_DIR}')
print(round(s.used / 1024**3, 1))
")
        echo -e "  Model dir    : ${GREEN}${MODEL_DIR}${RESET}"
        echo -e "  Disk free    : ${GREEN}${DISK_FREE_GB} GB${RESET} available on that volume"
    else
        DISK_FREE_GB=999
        echo -e "  Model dir    : ${YELLOW}${MODEL_DIR} (will be created)${RESET}"
    fi

    export TOTAL_RAM_GB AVAIL_RAM_GB GPU_VRAM_GB GPU_VRAM_FREE_GB GPU_NAME \
           DEVICE CPU_CORES GPU_COUNT GPU_MEM_UTIL DISK_FREE_GB
}

# ── Scan MODEL_DIR for already-downloaded models ──────────────────────────────
scan_local_models() {
    python3 - "${MODEL_DIR}" <<'PYEOF'
import os, sys, json

model_dir = sys.argv[1]
local_models = []

if not os.path.isdir(model_dir):
    print(json.dumps([]))
    sys.exit(0)

# HF cache layout: models--org--name/snapshots/hash/
# Also support flat org/name directories
for entry in os.scandir(model_dir):
    if not entry.is_dir():
        continue
    name = entry.name
    # HF cache format: models--ORG--NAME
    if name.startswith("models--"):
        parts = name[len("models--"):].split("--", 1)
        if len(parts) == 2:
            local_models.append(f"{parts[0]}/{parts[1]}")
    # Flat format: just scan for config.json one level deep
    else:
        for subentry in os.scandir(entry.path):
            if subentry.is_dir():
                if os.path.exists(os.path.join(subentry.path, "config.json")):
                    local_models.append(f"{name}/{subentry.name}")

print(json.dumps(local_models))
PYEOF
}

# ── Ask about quantization & CPU offload BEFORE filtering models ──────────────
configure_options() {
    echo -e "\n${BOLD}${CYAN}=== Memory Configuration ===${RESET}"
    echo -e "  These choices affect which models are shown as compatible.\n"

    # ── Quantization ──────────────────────────────────────────────────────────
    echo -e "${BOLD}Quantization:${RESET}"
    echo -e "  Reduces model size in memory, allowing larger models to fit."
    echo -e "  Approximate compression vs full bf16 weights:\n"
    echo -e "    1) None  — 1.0x  full bf16/fp16, best quality"
    echo -e "    2) int8  — 0.5x  ~8-bit via bitsandbytes, broad compatibility"
    echo -e "    3) awq   — 0.25x ~4-bit, fast, requires an AWQ model variant on HF"
    echo -e "    4) gptq  — 0.25x ~4-bit, fast, requires a GPTQ model variant on HF"
    echo -e "    5) fp8   — 0.5x  ~8-bit, NVIDIA Hopper+ (H100) GPUs only"
    echo ""
    read -rp "  Select quantization [1-5, default=1]: " quant_choice

    case "${quant_choice:-1}" in
        2) QUANT="int8";  QUANT_ARG="--quantization bitsandbytes --load-format bitsandbytes"; QUANT_RATIO=0.50 ;;
        3) QUANT="awq";   QUANT_ARG="--quantization awq";  QUANT_RATIO=0.25 ;;
        4) QUANT="gptq";  QUANT_ARG="--quantization gptq"; QUANT_RATIO=0.25 ;;
        5) QUANT="fp8";   QUANT_ARG="--quantization fp8";  QUANT_RATIO=0.50 ;;
        *) QUANT="none";  QUANT_ARG="";                    QUANT_RATIO=1.00 ;;
    esac

    echo -e "  Selected: ${GREEN}${QUANT}${RESET} (model weights take ~${QUANT_RATIO}x of bf16 size)"

    if [[ "$QUANT" == "awq" || "$QUANT" == "gptq" ]]; then
        echo -e "  ${YELLOW}Note: AWQ/GPTQ require a pre-quantized model variant (search for '${QUANT}' in model names).${RESET}"
    fi

    # ── CPU offloading ────────────────────────────────────────────────────────
    CPU_OFFLOAD_GB=0
    if [[ "$DEVICE" != "cpu" ]]; then
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

    # ── Compute effective memory budget ───────────────────────────────────────
    if [[ "$DEVICE" != "cpu" ]]; then
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

# ── Fetch models from HuggingFace ─────────────────────────────────────────────
fetch_models() {
    local search_query="${1:-}"
    local limit=150

    echo -e "\n${BOLD}${CYAN}=== Fetching Compatible Models from HuggingFace ===${RESET}"
    echo -e "  Searching for : ${YELLOW}${search_query:-'top text-generation models'}${RESET}"
    echo -e "  Quantization  : ${GREEN}${QUANT}${RESET}"
    echo -e "  Model size cap: ${GREEN}${MEM_BUDGET_GB} GB${RESET} bf16-equivalent\n"

    local url="${HF_API}?pipeline_tag=text-generation&sort=downloads&direction=-1&limit=${limit}&full=true"
    if [[ -n "$search_query" ]]; then
        url="${url}&search=${search_query// /%20}"
    fi

    # For AWQ/GPTQ, merge a dedicated quantized-model search so pre-quant variants appear
    if [[ "$QUANT" == "awq" || "$QUANT" == "gptq" ]]; then
        local quant_url="${HF_API}?pipeline_tag=text-generation&sort=downloads&direction=-1&limit=${limit}&full=true&search=${QUANT}"
        local r1 r2
        r1=$(curl -sf "$url" 2>/dev/null)        || { echo -e "${RED}API fetch failed.${RESET}"; exit 1; }
        r2=$(curl -sf "$quant_url" 2>/dev/null)  || r2="[]"
        echo "$r1" > /tmp/vllm_r1.json
        echo "$r2" > /tmp/vllm_r2.json
        python3 -c "
import json
with open('/tmp/vllm_r1.json') as f: a=json.load(f)
with open('/tmp/vllm_r2.json') as f: b=json.load(f)
seen=set(); merged=[]
for m in a+b:
    if m['id'] not in seen:
        seen.add(m['id']); merged.append(m)
with open('/tmp/vllm_hf_response.json','w') as f: json.dump(merged,f)
"
    else
        curl -sf "$url" > /tmp/vllm_hf_response.json 2>/dev/null || { echo -e "${RED}API fetch failed.${RESET}"; exit 1; }
    fi

    # Scan for locally cached models
    LOCAL_MODELS_JSON=$(scan_local_models)

    python3 - "${MEM_BUDGET_GB}" "${QUANT_RATIO}" "${CPU_OFFLOAD_GB}" \
               "${GPU_VRAM_FREE_GB:-0}" "${QUANT}" "${RAW_MEM_GB}" \
               "${DISK_FREE_GB}" "$LOCAL_MODELS_JSON" <<'PYEOF'
import json, sys, re, os

mem_budget    = float(sys.argv[1])
quant_ratio   = float(sys.argv[2])
offload_gb    = float(sys.argv[3])
vram_free_gb  = float(sys.argv[4])
quant         = sys.argv[5]
raw_mem_gb    = float(sys.argv[6])
disk_free_gb  = float(sys.argv[7])
local_models  = set(json.loads(sys.argv[8]))

with open("/tmp/vllm_hf_response.json") as f:
    data = json.load(f)

# ── Known MoE model patterns and their active-param fractions ────────────────
# Format: (regex_pattern, total_params_b, active_params_b)
# We load ALL weights but only run active params — size = total * 2 bytes
MOE_PATTERNS = [
    (r'deepseek-v3',          671, 671),
    (r'deepseek-r1(?!.*distill)', 671, 671),
    (r'deepseek-r1-0528',     671, 671),
    (r'kimi-k2',              1000, 32),
    (r'mixtral-8x7b',         46,  12.9),
    (r'mixtral-8x22b',        141, 39),
    (r'qwen.*moe',            57,  14),
    (r'qwen3-30b-a3b',        30,  3),
    (r'qwen.*57b-a14b',       57,  14),
]

def get_moe_size(model_id):
    """Return (total_gb, is_moe) for known MoE models."""
    name = model_id.lower()
    for pattern, total_b, _ in MOE_PATTERNS:
        if re.search(pattern, name):
            # bf16 = 2 bytes/param, +10% overhead
            return total_b * 2 * 1.10, True
    return None, False

def estimate_size_gb(model_id):
    """Estimate bf16 model size in GB from the name."""
    # Check MoE first
    moe_size, is_moe = get_moe_size(model_id)
    if is_moe:
        return moe_size

    name = model_id.lower()
    # Match patterns like 0.5b, 1.5b, 7b, 13b, 70b, 120b etc.
    match = re.search(r'(?<![a-z])(\d+(?:\.\d+)?)\s*([bm])(?![a-z])', name)
    if not match:
        return None
    value = float(match.group(1))
    unit  = match.group(2)
    params_b = value / 1000 if unit == 'm' else value
    return params_b * 2 * 1.10  # bf16 + ~10% overhead

SKIP_TAGS = {"dataset", "lora", "gguf", "onnx"}
SKIP_PATTERNS = [
    r'gguf', r'onnx', r'-lora$', r'/lora-', r'embedding',
    r'reranker', r'reward-model', r'classifier', r'internal-testing',
    r'tiny-random',
]

# For AWQ/GPTQ: if quant is selected, exclude base (non-quantized) models
# that share a name with a quantized variant we've already found
results = []
seen_bases = set()

for model in data:
    model_id = model.get("id", "")
    tags = [t.lower() for t in model.get("tags", [])]

    if any(t in SKIP_TAGS for t in tags):
        continue
    if any(re.search(p, model_id.lower()) for p in SKIP_PATTERNS):
        continue

    # For AWQ/GPTQ: only show models that are actually pre-quantized.
    # vLLM cannot quantize on the fly, so listing base models is misleading —
    # they will pass the browser but fail at launch. Strict name-based allowlist.
    if quant in ("awq", "gptq"):
        model_lower = model_id.lower()
        is_quant_variant = quant in model_lower or "int4" in model_lower
        if not is_quant_variant:
            continue  # drop any model that isn't clearly a pre-quantized variant
        # Deduplicate: keep only the first (highest-download) variant per base name
        base_name = re.sub(r'[-_](awq|gptq|int4|quantized|4bit).*$', '', model_lower)
        if base_name in seen_bases:
            continue
        seen_bases.add(base_name)

    # Determine bf16-equivalent model size
    safetensors = model.get("safetensors", {})
    total_bytes = safetensors.get("total", 0) if safetensors else 0

    _, is_moe = get_moe_size(model_id)

    if total_bytes and total_bytes > 0:
        bf16_size_gb = total_bytes / 1e9
        # Pre-quantized AWQ/GPTQ files are already compressed — normalise to bf16
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

    # Effective post-quantization memory footprint
    if size_known:
        effective_gb = bf16_size_gb * quant_ratio
        physical_budget = vram_free_gb * 0.90 + offload_gb
        # Exclude only if too large even after quantization AND doesn't fit at bf16
        if effective_gb > physical_budget and bf16_size_gb > physical_budget:
            continue

    downloads = model.get("downloads", 0)
    likes     = model.get("likes", 0)

    # Flags
    is_local   = model_id in local_models
    needs_ram  = False
    size_warn  = False  # unknown size — flag it

    if size_known:
        effective_gb = bf16_size_gb * quant_ratio
        vram_budget  = vram_free_gb * 0.90
        needs_ram    = offload_gb > 0 and effective_gb > vram_budget
        # Warn if model is close to or exceeds disk free space
        # (rough: effective_gb is what we'd need to download)
        size_warn = effective_gb > disk_free_gb * 0.9
    else:
        size_warn = True  # unknown = potentially risky

    flags = ""
    if is_local:  flags += "L"
    if needs_ram: flags += "R"
    if size_warn: flags += "?"

    results.append((model_id, size_str, bf16_size_gb or 9999,
                    downloads, likes, flags, is_local, is_moe))

# Sort: locally cached first, then by downloads
results.sort(key=lambda x: (not x[6], -x[3]))

if not results:
    print("NO_MODELS_FOUND")
    sys.exit(0)

# ── Print table ───────────────────────────────────────────────────────────────
print(f"{'#':<4} {'Model ID':<52} {'On-disk':>9} {'Flags':5} {'Downloads':>11} {'Likes':>6}")
print("─" * 92)

for i, (mid, sz, _, dl, lk, flags, is_local, is_moe) in enumerate(results[:50], 1):
    mid_display = mid if len(mid) <= 52 else mid[:49] + "..."
    flag_str = ""
    if "L" in flags: flag_str += "\033[32mL\033[0m"   # green L = local
    if "R" in flags: flag_str += "\033[33mR\033[0m"   # yellow R = needs RAM
    if "?" in flags: flag_str += "\033[31m?\033[0m"   # red ? = unknown/disk warn
    if is_moe:       flag_str += "\033[36mM\033[0m"   # cyan M = MoE
    # Pad flag_str to consistent visual width (each escape seq = ~9 chars overhead)
    vis_len = len(re.sub(r'\033\[[0-9;]*m', '', flag_str))
    flag_padded = flag_str + " " * (4 - vis_len)
    print(f"{i:<4} {mid_display:<52} {sz:>9}  {flag_padded} {dl:>11,} {lk:>6,}")

# Write machine-readable list
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

# ── Interactive model selection ───────────────────────────────────────────────
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
        read -rp "New search query (e.g. 'llama 8b', 'mistral', 'qwen'): " query
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

    export SELECTED_MODEL
}

# ── Suggest context length based on model config + remaining VRAM ────────────
suggest_max_len() {
    local model_id="$1"
    python3 - "${model_id}" "${GPU_VRAM_FREE_GB}" "${QUANT_RATIO}" "${CPU_OFFLOAD_GB}" <<'PYEOF'
import sys, re, urllib.request, json

model_id     = sys.argv[1]
vram_free_gb = float(sys.argv[2])
quant_ratio  = float(sys.argv[3])
offload_gb   = float(sys.argv[4])

# ── Step 1: fetch model architecture from HF config.json ─────────────────────
# We need the real KV cache cost, which depends on the model's actual architecture.
# Formula: bytes_per_token = num_layers * 2(K+V) * num_kv_heads * head_dim * 2(bf16)
model_max_len  = None
bytes_per_token = None
cfg = {}
try:
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    req = urllib.request.Request(url, headers={"User-Agent": "vllm-launcher/1.0"})
    with urllib.request.urlopen(req, timeout=6) as resp:
        cfg = json.loads(resp.read().decode())

    # Hard context ceiling vLLM enforces
    for key in ("max_position_embeddings", "model_max_length", "n_positions", "max_sequence_length"):
        if key in cfg and isinstance(cfg[key], (int, float)) and 0 < cfg[key] < 10_000_000:
            model_max_len = int(cfg[key])
            break

    # KV cache architecture parameters
    num_layers  = cfg.get("num_hidden_layers") or cfg.get("n_layer") or cfg.get("num_layers")
    head_dim    = cfg.get("head_dim") or cfg.get("d_head")
    num_kv_heads = cfg.get("num_key_value_heads") or cfg.get("num_kv_heads")
    hidden_size  = cfg.get("hidden_size") or cfg.get("d_model")
    num_heads    = cfg.get("num_attention_heads") or cfg.get("n_head")

    # Derive head_dim if not explicit
    if not head_dim and hidden_size and num_heads:
        head_dim = hidden_size // num_heads

    # Fall back num_kv_heads to num_heads (MHA, no GQA)
    if not num_kv_heads and num_heads:
        num_kv_heads = num_heads

    if num_layers and num_kv_heads and head_dim:
        # 2 = K+V, 2 = bytes per bf16 element
        bytes_per_token = int(num_layers) * 2 * int(num_kv_heads) * int(head_dim) * 2

except Exception:
    pass  # fall through to rough heuristic

# ── Step 2: estimate VRAM available for KV cache ─────────────────────────────
name = model_id.lower()
match = re.search(r'(?<![a-z])(\d+(?:\.\d+)?)\s*([bm])(?![a-z])', name)
params_b = 0.0
if match:
    value    = float(match.group(1))
    unit     = match.group(2)
    params_b = value / 1000 if unit == 'm' else value

# Quantized model size (what actually lives in memory)
quantized_gb = params_b * 2 * quant_ratio * 1.10 if params_b else 0.0

# With CPU offload, vLLM splits the model across VRAM + RAM.
# We estimate the fraction that remains in VRAM conservatively at 30%
# (even with large offload_gb, PyTorch keeps activations + some layers in VRAM).
# Without offload, 100% of weights are in VRAM.
if offload_gb > 0 and quantized_gb > 0:
    fraction_in_vram = max(0.30, (quantized_gb - offload_gb) / quantized_gb)
else:
    fraction_in_vram = 1.0

vram_used_by_weights = quantized_gb * fraction_in_vram
vram_total_budget    = vram_free_gb * 0.85   # matches gpu_memory_utilization
vram_for_kv          = max(0.25, vram_total_budget - vram_used_by_weights)

# KV cache bytes per token fallback (if config fetch failed)
if not bytes_per_token:
    if params_b:
        # Empirically: scales roughly as params_b * 11,000 bytes/token
        bytes_per_token = max(40_000, int(params_b * 11_000))
    else:
        bytes_per_token = 160_000  # safe default (~14B-class model)

# ── Step 3: compute safe token budget ────────────────────────────────────────
vram_based_max = int((vram_for_kv * 1024 ** 3) / bytes_per_token)

if model_max_len:
    suggested_tokens = min(vram_based_max, model_max_len)
else:
    suggested_tokens = vram_based_max

# Apply a 20% safety haircut — our estimates are approximations and vLLM
# reserves additional memory for CUDA graphs, activations, and overhead.
suggested_tokens = int(suggested_tokens * 0.80)

# Round down to nearest sensible boundary
common = [512, 1024, 2048, 4096, 8192, 16384, 24576, 32768, 40960, 65536, 131072]
suggestion = 512
for c in reversed(common):
    if suggested_tokens >= c:
        suggestion = c
        break

print(str(suggestion))
PYEOF
}

# ── Validate that an AWQ/GPTQ model actually has its quantization config ──────
validate_quant_model() {
    local model_id="$1"
    local quant="$2"

    echo -e "  ${CYAN}Verifying model has ${quant^^} quantization config...${RESET}"

    local result
    result=$(python3 - "${model_id}" "${quant}" <<'PYEOF'
import sys, urllib.request

model_id = sys.argv[1]
quant    = sys.argv[2]

# AWQ and GPTQ both store their quantization metadata in quantize_config.json
url = f"https://huggingface.co/{model_id}/resolve/main/quantize_config.json"
try:
    req = urllib.request.Request(url, headers={"User-Agent": "vllm-launcher/1.0"})
    urllib.request.urlopen(req, timeout=8)
    print("yes")
except Exception:
    print("no")
PYEOF
)

    if [[ "$result" == "no" ]]; then
        echo -e "\n${RED}Error: '${model_id}' does not appear to be a ${quant^^}-quantized model.${RESET}"
        echo -e "${YELLOW}  vLLM requires the model to already be quantized — it cannot quantize on the fly.${RESET}"
        echo -e "${YELLOW}  Search HuggingFace for a pre-quantized variant, e.g.:${RESET}"
        echo -e "    ${CYAN}https://huggingface.co/models?search=${model_id##*/}-${quant}${RESET}"
        echo ""
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
    fi

    echo -e "  ${GREEN}${quant^^} config found ✓${RESET}"
}

# ── Final launch configuration ────────────────────────────────────────────────
configure_launch() {
    echo -e "\n${BOLD}${CYAN}=== Launch Configuration ===${RESET}"
    echo -e "  Model        : ${GREEN}${SELECTED_MODEL}${RESET}"
    echo -e "  Host         : ${GREEN}${HOST}:${PORT}${RESET}"
    echo -e "  Device       : ${GREEN}${DEVICE}${RESET}"
    echo -e "  Quantization : ${GREEN}${QUANT}${RESET}"
    echo -e "  GPU mem util : ${GREEN}${GPU_MEM_UTIL}${RESET}"
    echo -e "  Model dir    : ${GREEN}${MODEL_DIR}${RESET}"
    echo -e "  Disk free    : ${GREEN}${DISK_FREE_GB} GB${RESET}"
    [[ "${CPU_OFFLOAD_GB}" != "0" ]] && echo -e "  CPU offload  : ${GREEN}${CPU_OFFLOAD_GB} GB${RESET}"
    echo ""

    # Create model directory if needed
    if [[ ! -d "$MODEL_DIR" ]]; then
        echo -e "  ${YELLOW}Creating model directory: ${MODEL_DIR}${RESET}"
        mkdir -p "$MODEL_DIR" || { echo -e "  ${RED}Failed to create directory. Check permissions.${RESET}"; exit 1; }
        echo -e "  ${GREEN}Directory created.${RESET}\n"
    fi

    local dtype="auto"
    [[ "$DEVICE" == "cpu" ]] && dtype="float32"

    local tensor_parallel="${GPU_COUNT:-1}"
    [[ "$tensor_parallel" -lt 1 ]] && tensor_parallel=1

    # Suggest a sensible context length
    local suggested_len
    suggested_len=$(suggest_max_len "$SELECTED_MODEL")
    if [[ "$suggested_len" -gt 0 ]]; then
        echo -e "  ${DIM}Suggested max context based on available VRAM: ${suggested_len} tokens${RESET}"
    fi
    read -rp "  Max context length in tokens [suggested=${suggested_len}, leave blank to use model default]: " max_len
    local max_len_arg=""
    [[ -n "$max_len" ]] && max_len_arg="--max-model-len ${max_len}"

    local token_arg=""
    if [[ -n "${HF_TOKEN:-}" ]]; then
        token_arg="--token ${HF_TOKEN}"
        echo -e "  ${GREEN}Using HF_TOKEN from environment.${RESET}"
    else
        read -rp "  HuggingFace token (for gated models, leave blank if not needed): " hf_token
        [[ -n "$hf_token" ]] && token_arg="--token ${hf_token}"
    fi

    # ── Tool calling ──────────────────────────────────────────────────────────
    # --enable-auto-tool-choice lets the model decide when to call tools.
    # --tool-call-parser must match the model's chat template format or tool
    # calls will silently malform. We auto-detect from the model name but
    # always let the user override.
    local tool_call_arg=""
    echo ""
    echo -e "${BOLD}Tool Calling:${RESET}"
    echo -e "  Required for agentic / function-calling workloads."
    read -rp "  Enable tool calling? [Y/n]: " enable_tools
    if [[ "${enable_tools,,}" != "n" ]]; then
        # Auto-detect the best parser from the model name
        local detected_parser
        detected_parser=$(python3 - "${SELECTED_MODEL}" <<'PYEOF'
import sys, re
model = sys.argv[1].lower()
# Ordered from most-specific to least-specific
if re.search(r'qwen', model):            print("hermes");      exit()
if re.search(r'hermes|nous', model):     print("hermes");      exit()
if re.search(r'llama.?3', model):        print("llama3_json"); exit()
if re.search(r'mistral|mixtral', model): print("mistral");     exit()
if re.search(r'internlm', model):        print("internlm");    exit()
if re.search(r'jamba', model):           print("jamba");       exit()
if re.search(r'granite', model):         print("granite-20b-fc"); exit()
if re.search(r'xlam|salesforce', model): print("xlam");        exit()
# Conservative fallback — hermes is the most broadly compatible
print("hermes")
PYEOF
)
        echo ""
        echo -e "  Available parsers:"
        echo -e "    ${CYAN}hermes${RESET}        — Qwen, Nous-Hermes, and most modern instruct models ${DIM}(recommended default)${RESET}"
        echo -e "    ${CYAN}llama3_json${RESET}   — Llama 3.x series"
        echo -e "    ${CYAN}mistral${RESET}       — Mistral / Mixtral"
        echo -e "    ${CYAN}internlm${RESET}      — InternLM series"
        echo -e "    ${CYAN}jamba${RESET}         — AI21 Jamba"
        echo -e "    ${CYAN}granite-20b-fc${RESET}— IBM Granite"
        echo -e "    ${CYAN}xlam${RESET}          — Salesforce xLAM"
        echo -e "    ${CYAN}pythonic${RESET}      — Models trained with Python-style tool call syntax"
        echo ""
        echo -e "  Auto-detected parser for ${GREEN}${SELECTED_MODEL##*/}${RESET}: ${YELLOW}${detected_parser}${RESET}"
        read -rp "  Parser to use [default=${detected_parser}]: " parser_input
        local tool_parser="${parser_input:-$detected_parser}"
        tool_call_arg="--enable-auto-tool-choice --tool-call-parser ${tool_parser}"
        echo -e "  ${GREEN}Tool calling enabled with parser: ${tool_parser}${RESET}"
    else
        echo -e "  ${DIM}Tool calling disabled.${RESET}"
    fi

    # ── Validate AWQ/GPTQ model has the required quant config ─────────────────
    # vLLM does NOT quantize on the fly — the model must ship quantize_config.json.
    # Without this check the server crashes with "Cannot find the config file for awq/gptq".
    if [[ "$QUANT" == "awq" || "$QUANT" == "gptq" ]]; then
        validate_quant_model "${SELECTED_MODEL}" "${QUANT}"
    fi

    # ── Calculate safe max_num_seqs ──────────────────────────────────────────
    # The sampler warmup allocates [max_num_seqs × vocab_size] logit tensors.
    # On small GPUs with large-vocab models this OOMs at the default of 256.
    # Strategy: start from a VRAM-tier baseline, then refine using vocab_size
    # from config.json (smaller vocab → can afford more concurrent sequences).
    local max_num_seqs
    max_num_seqs=$(python3 - "${SELECTED_MODEL}" "${GPU_VRAM_GB}" "${GPU_VRAM_FREE_GB}" "${GPU_MEM_UTIL}" <<'PYEOF'
import sys, urllib.request, json

model_id     = sys.argv[1]
vram_total   = float(sys.argv[2])
vram_free    = float(sys.argv[3])
gpu_mem_util = float(sys.argv[4])

# Step 1: VRAM-tier baseline (conservative safe values empirically chosen)
# These account for the fact that CUDA graphs + KV cache + weights leave
# very little headroom on small GPUs for the sampler warmup pass.
if   vram_total <= 12:  base = 16
elif vram_total <= 16:  base = 32
elif vram_total <= 24:  base = 64
elif vram_total <= 40:  base = 96
elif vram_total <= 80:  base = 192
else:                   base = 256

# Step 2: try to fetch vocab_size from config.json to refine upward.
# Larger vocab → each seq costs more VRAM → fewer safe concurrent seqs.
# Smaller vocab (e.g. 32K) → can safely allow more.
vocab_size = 150_000  # conservative default (covers Qwen/Llama large vocabs)
try:
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    req = urllib.request.Request(url, headers={"User-Agent": "vllm-launcher/1.0"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        cfg = json.loads(resp.read().decode())
    vocab_size = cfg.get("vocab_size") or vocab_size
except Exception:
    pass

# Step 3: scale adjustment based on vocab relative to our baseline assumption.
# If vocab is smaller than assumed, we can afford proportionally more sequences.
# Cap the scale factor at 4x to stay safe.
scale = min(4.0, 150_000 / vocab_size)
adjusted = int(base * scale)

# Round to nearest power of 2 for cleanliness, clamp to [8, 256]
for p in [8, 16, 32, 48, 64, 96, 128, 192, 256]:
    if adjusted <= p:
        print(p)
        sys.exit(0)
print(256)
PYEOF
)
    echo -e "  Max num seqs : ${GREEN}${max_num_seqs}${RESET} (auto-tuned to avoid sampler OOM)"

    CMD="python3 -m vllm.entrypoints.openai.api_server"
    CMD+=" --model ${SELECTED_MODEL}"
    CMD+=" --host ${HOST}"
    CMD+=" --port ${PORT}"
    CMD+=" --dtype ${dtype}"
    CMD+=" --tensor-parallel-size ${tensor_parallel}"
    CMD+=" --gpu-memory-utilization ${GPU_MEM_UTIL}"
    CMD+=" --max-num-seqs ${max_num_seqs}"
    CMD+=" --download-dir ${MODEL_DIR}"
    [[ -n "$QUANT_ARG" ]]            && CMD+=" ${QUANT_ARG}"
    [[ "${CPU_OFFLOAD_GB}" != "0" ]] && CMD+=" --cpu-offload-gb ${CPU_OFFLOAD_GB}"
    [[ -n "$max_len_arg" ]]          && CMD+=" ${max_len_arg}"
    [[ -n "$token_arg" ]]            && CMD+=" ${token_arg}"
    [[ -n "$tool_call_arg" ]]        && CMD+=" ${tool_call_arg}"
    CMD+=" --trust-remote-code"

    export CMD
}

# ── Dependency checks ─────────────────────────────────────────────────────────
check_dependencies() {
    # vllm
    if ! python3 -c "import vllm" 2>/dev/null; then
        echo -e "${YELLOW}Warning: vllm not found in current Python environment.${RESET}"
        read -rp "Install vllm now? [Y/n]: " install_now
        if [[ "${install_now,,}" != "n" ]]; then
            pip install vllm
        else
            echo -e "${RED}Aborting — vllm not available.${RESET}"; exit 1
        fi
    fi

    # Quantization-specific packages
    case "$QUANT" in
        int8)
            if ! python3 -c "import bitsandbytes" 2>/dev/null; then
                echo -e "${YELLOW}int8 requires 'bitsandbytes'. Install now? [Y/n]: ${RESET}"
                read -rp "" bnb_install
                [[ "${bnb_install,,}" != "n" ]] && pip install "bitsandbytes>=0.46.1" || { echo -e "${RED}Aborting.${RESET}"; exit 1; }
            else
                BNB_OK=$(python3 -c "
import bitsandbytes as bnb, packaging.version as pv
print('yes' if pv.parse(bnb.__version__) >= pv.parse('0.46.1') else 'no')
" 2>/dev/null || echo "unknown")
                if [[ "$BNB_OK" == "no" ]]; then
                    echo -e "${YELLOW}bitsandbytes too old (need >=0.46.1). Upgrade? [Y/n]: ${RESET}"
                    read -rp "" bnb_upgrade
                    [[ "${bnb_upgrade,,}" != "n" ]] && pip install "bitsandbytes>=0.46.1" || { echo -e "${RED}Aborting.${RESET}"; exit 1; }
                fi
            fi
            ;;
        awq)
            if ! python3 -c "import autoawq" 2>/dev/null; then
                echo -e "${YELLOW}AWQ works best with 'autoawq'. Install now? [Y/n]: ${RESET}"
                read -rp "" awq_install
                [[ "${awq_install,,}" != "n" ]] && pip install autoawq
            fi
            ;;
        gptq)
            if ! python3 -c "import auto_gptq" 2>/dev/null && ! python3 -c "import optimum" 2>/dev/null; then
                echo -e "${YELLOW}GPTQ works best with 'auto-gptq'. Install now? [Y/n]: ${RESET}"
                read -rp "" gptq_install
                [[ "${gptq_install,,}" != "n" ]] && pip install auto-gptq
            fi
            ;;
    esac
}

# ── Launch server ─────────────────────────────────────────────────────────────
launch_server() {
    echo -e "\n${BOLD}${CYAN}=== Ready to Launch ===${RESET}"
    echo -e "${YELLOW}Command:${RESET}\n"
    echo -e "  ${CYAN}${CMD}${RESET}\n"

    check_dependencies

    # Save command for --relaunch
    echo "#!/usr/bin/env bash" > "$LAST_CMD_FILE"
    echo "$CMD" >> "$LAST_CMD_FILE"
    chmod +x "$LAST_CMD_FILE"
    echo -e "  ${DIM}Command saved to ${LAST_CMD_FILE} — use ./vllm_launcher.sh --relaunch to skip wizard next time${RESET}\n"

    read -rp "Launch server now? [Y/n]: " confirm
    if [[ "${confirm,,}" != "n" ]]; then
        echo -e "\n${GREEN}Starting vLLM server...${RESET}"
        echo -e "OpenAI-compatible API : ${CYAN}http://${HOST}:${PORT}/v1${RESET}"
        echo -e "Press ${BOLD}Ctrl+C${RESET} to stop.\n"
        eval "$CMD"
    else
        echo ""
        echo -e "${YELLOW}To start later, run:${RESET}\n"
        echo -e "  ${CYAN}${CMD}${RESET}"
        echo -e "\n  or: ${CYAN}./vllm_launcher.sh --relaunch${RESET}\n"
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
    echo -e "\n${BOLD}${GREEN}╔══════════════════════════════════════════╗${RESET}"
    echo -e "${BOLD}${GREEN}║        vLLM Model Launcher               ║${RESET}"
    echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════╝${RESET}"

    detect_hardware
    configure_options

    local initial_query="${1:-}"
    if [[ -z "$initial_query" ]]; then
        echo ""
        read -rp "Search for a specific model type? (leave blank for top downloads): " initial_query
    fi

    fetch_models "$initial_query"

    if grep -q "NO_MODELS_FOUND" /tmp/vllm_model_list.txt 2>/dev/null || [[ ! -s /tmp/vllm_model_list.txt ]]; then
        echo -e "${RED}No models found that fit your hardware.${RESET}"
        exit 1
    fi

    select_model
    configure_launch
    launch_server
}

main "${1:-}"
