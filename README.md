# 🚀 vLLM Model Launcher

A smart, interactive bash script that detects your hardware, browses HuggingFace for compatible models, and launches a local [vLLM](https://github.com/vllm-project/vllm) OpenAI-compatible API server — all in one command.

No more manually hunting for models that fit your GPU, guessing context lengths, or wrestling with launch flags. Just run it and go.

---

## ✨ Features

- **Hardware-aware model browsing** — detects your GPU VRAM, system RAM, and free disk space, then queries HuggingFace for models that actually fit
- **Quantization-first workflow** — choose int8 / AWQ / GPTQ / fp8 *before* the model list appears, so the list only shows models compatible with your chosen method
- **CPU RAM offloading** — optionally spill model layers to system RAM to run models larger than your VRAM alone
- **Smart context length suggestion** — fetches the model's architecture from HuggingFace config.json and calculates a safe `--max-model-len` based on your actual remaining VRAM after weights load
- **Auto-tuned `--max-num-seqs`** — scales concurrent request slots to your GPU tier and model vocab size to prevent sampler OOM errors
- **Auto-tuned `--gpu-memory-utilization`** — calculated from your actual free VRAM at launch time, not a hardcoded guess
- **50-model browsing table** with useful flags:
  - 🟢 `L` — model already cached locally in your model directory
  - 🟡 `R` — model needs CPU RAM offload to fit
  - 🔴 `?` — unknown size or potential disk space warning
  - 🔵 `M` — MoE (Mixture of Experts) architecture
- **MoE model awareness** — DeepSeek-V3, Mixtral, Kimi-K2 and others are correctly sized and flagged
- **AWQ/GPTQ deduplication** — when a quantized variant exists, the base model is hidden to prevent mismatches
- **Disk space check** — warns before you download a model that won't fit on your drive
- **Local model detection** — scans your model directory and sorts already-downloaded models to the top
- **Dependency auto-install** — detects missing packages (`bitsandbytes`, `autoawq`, `auto-gptq`) and offers to install them before launch
- **`--relaunch` mode** — skip the entire wizard and instantly reuse your last command
- **Saves last command** to `/tmp/vllm_last_command.sh` for easy reuse

---

## 📋 Requirements

- Linux (tested on Ubuntu 24)
- `bash`, `curl`, `python3`
- `vllm` installed in your Python environment (`pip install vllm`)
- NVIDIA GPU with CUDA drivers, or AMD GPU with ROCm (CPU-only mode also supported)

---

## ⚡ Quick Start

```bash
# Clone or download the script
chmod +x vllm_launcher.sh

# Run the interactive wizard
./vllm_launcher.sh

# Pre-filter to a specific model family
./vllm_launcher.sh "mistral 7b"

# Skip the wizard and reuse your last session
./vllm_launcher.sh --relaunch
```

Once running, your model is available as an OpenAI-compatible API at:

```
http://0.0.0.0:4444/v1
```

Drop it straight into any OpenAI-compatible client by pointing the base URL there.

---

## 🛠️ Configuration

At the top of the script, edit these three lines to match your setup:

```bash
HOST="0.0.0.0"          # interface to bind (0.0.0.0 = all interfaces)
PORT="4444"             # port to serve on
MODEL_DIR="/path/to/your/models"  # where models are downloaded
```

You can also set `HF_TOKEN` as an environment variable for gated models (Llama, Gemma, etc.):

```bash
export HF_TOKEN="hf_your_token_here"
./vllm_launcher.sh
```

---

## 🎛️ Quantization Options

| Option | Compression | Notes |
|--------|-------------|-------|
| None   | 1.0× (full bf16) | Best quality, highest VRAM use |
| int8   | ~0.5×  | Via `bitsandbytes`. Works on any bf16 model |
| awq    | ~0.25× | Requires a pre-quantized AWQ model variant |
| gptq   | ~0.25× | Requires a pre-quantized GPTQ model variant |
| fp8    | ~0.5×  | NVIDIA Hopper+ (H100/H200) only |

AWQ and GPTQ automatically search HuggingFace for pre-quantized variants and filter the model list accordingly.

---

## 💾 CPU RAM Offloading

When enabled, model layers that don't fit in VRAM are spilled to system RAM via the `--cpu-offload-gb` flag. This lets you run models significantly larger than your VRAM alone — for example, a 14B model at int8 on a 12 GB GPU with 22 GB of RAM offload.

**Trade-off:** offloaded layers are slower to access due to the PCIe bus bottleneck. Expect lower tokens/sec compared to a model that fits entirely in VRAM.

---

## 📊 Example Session

```
=== Hardware Detection ===
  GPU          : NVIDIA GeForce RTX 3060 (x1)
  VRAM Total   : 12.0 GB
  VRAM Free    : 10.5 GB
  Disk free    : 41.6 GB available on that volume

=== Memory Configuration ===
  Select quantization [1-5, default=1]: 2   ← int8
  Enable CPU RAM offloading? [y/N]: y
  GB to offload to RAM [default=22]: 22

  Model size limit: 63.0 GB bf16-equivalent

#    Model ID                          On-disk  Flags   Downloads
1    Qwen/Qwen3-14B                   ~30.8 GB   LR     1,355,433
2    Qwen/Qwen2.5-7B-Instruct         ~15.4 GB          18,725,795
...

> 1   ← select Qwen3-14B

  Suggested max context : 16384 tokens
  Max num seqs          : 16 (auto-tuned)

  Launching: python3 -m vllm.entrypoints.openai.api_server ...
  OpenAI-compatible API : http://0.0.0.0:4444/v1
```

---

## 🤖 Built With AI

This script was built collaboratively with **Claude** (Anthropic's AI assistant), which wrote and iteratively improved the code across multiple debugging sessions — including the hardware detection, HuggingFace API integration, KV cache math, MoE model handling, and sampler OOM fixes. Each error log was fed back and analysed to make the launcher progressively more robust.

It's a good example of what's possible when you pair domain knowledge with an AI that can write, test, and reason about bash and Python together.

---

## 🤝 Contributing

Issues and pull requests are welcome! If you hit a new error or have a model family that isn't handled well, open an issue with the log output and we'll get it fixed.

---

## 📄 License

MIT — see [LEGAL.md](LEGAL.md) for full terms.
