@echo off
setlocal

echo ========================================================
echo THE ALETHEIA FORGE: DRIVE-ISOLATED REBUILD (F: DRIVE)
echo ========================================================

:: 1. Force context to the current drive and directory
cd /d "%~dp0"

:: 2. Delete the broken environment
:: echo [1/4] Nuking broken .venv...
:: if exist ".venv" rd /s /q ".venv"

:: 3. Create a fresh, drive-local environment
echo [2/4] Creating fresh local environment on F: drive...
python -m venv .venv
call .venv\Scripts\activate.bat

REM Install CUDA torch FIRST (hard constraint)
%PIP% install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
REM Install training stack
%PIP% install -r requirements_training.txt
REM Re-enforce CUDA torch (Unsloth may override)
echo [PATCH] Enforcing CUDA Torch
%PIP% install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
REM Enforce PEFT invariant
%PIP% install peft==0.18.1 --no-deps --force-reinstall

echo ========================================================
echo [OK] Training Runtime Dependencies Installed.
echo ========================================================
