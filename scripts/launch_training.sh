#!/usr/bin/env bash
# ============================================================
# Aletheia V2.1 — Local QLoRA Training Launch Script
# Target: 2x 8GB GPUs via DDP (Distributed Data Parallel)
# ============================================================
set -euo pipefail

CONFIG="${1:-config/axolotl_2x8gb.yml}"
NUM_GPUS="${2:-2}"

echo "========================================"
echo " Aletheia V2.1 QLoRA Training"
echo " Config:  ${CONFIG}"
echo " GPUs:    ${NUM_GPUS}"
echo "========================================"

# --- Pre-flight checks ---
if ! python -c "import axolotl" 2>/dev/null; then
    echo "[ERROR] Axolotl not installed. Install via:"
    echo "  pip install axolotl[flash-attn]"
    echo "  # or for Unsloth integration:"
    echo "  pip install unsloth axolotl"
    exit 1
fi

if ! python -c "import torch; assert torch.cuda.device_count() >= ${NUM_GPUS}" 2>/dev/null; then
    echo "[WARN] Expected ${NUM_GPUS} GPUs, found $(python -c 'import torch; print(torch.cuda.device_count())')"
    echo "Falling back to single GPU mode..."
    NUM_GPUS=1
fi

# --- Show VRAM status ---
python -c "
import torch
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    total_gb = props.total_mem / (1024**3)
    print(f'  GPU {i}: {props.name} — {total_gb:.1f} GB VRAM')
"

echo ""
echo "[*] Launching training..."
echo ""

if [ "${NUM_GPUS}" -gt 1 ]; then
    # Multi-GPU: DDP via accelerate
    accelerate launch \
        --num_processes "${NUM_GPUS}" \
        --multi_gpu \
        -m axolotl.cli.train "${CONFIG}"
else
    # Single GPU: direct launch
    python -m axolotl.cli.train "${CONFIG}"
fi

echo ""
echo "[OK] Training complete. Adapter saved to: runs/aletheia-v2.1-qlora/"
echo "[*] To merge LoRA into base model:"
echo "    python -m axolotl.cli.merge_lora ${CONFIG} --lora_model_dir runs/aletheia-v2.1-qlora/"
