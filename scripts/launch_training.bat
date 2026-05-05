@echo off
REM ============================================================
REM Aletheia V2.1 — Local QLoRA Training Launch Script (Windows)
REM Target: 2x 8GB GPUs via DDP (Distributed Data Parallel)
REM ============================================================

set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=config\axolotl_2x8gb.yml

set NUM_GPUS=%2
if "%NUM_GPUS%"=="" set NUM_GPUS=2

echo ========================================
echo  Aletheia V2.1 QLoRA Training
echo  Config:  %CONFIG%
echo  GPUs:    %NUM_GPUS%
echo ========================================

REM --- Show VRAM status ---
python -c "import torch; [print(f'  GPU {i}: {torch.cuda.get_device_properties(i).name} -- {torch.cuda.get_device_properties(i).total_mem / (1024**3):.1f} GB VRAM') for i in range(torch.cuda.device_count())]"

echo.
echo [*] Launching training...
echo.

if %NUM_GPUS% GTR 1 (
    accelerate launch --num_processes %NUM_GPUS% --multi_gpu -m axolotl.cli.train %CONFIG%
) else (
    python -m axolotl.cli.train %CONFIG%
)

echo.
echo [OK] Training complete. Adapter saved to: runs\aletheia-v2.1-qlora\
echo [*] To merge LoRA into base model:
echo     python -m axolotl.cli.merge_lora %CONFIG% --lora_model_dir runs\aletheia-v2.1-qlora\
pause
