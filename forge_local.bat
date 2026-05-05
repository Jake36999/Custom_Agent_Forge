@echo off
setlocal
echo ========================================================
echo THE ALETHEIA FORGE: DRIVE-ISOLATED REBUILD (F: DRIVE)
echo ========================================================

:: 1. Force context to the current drive and directory
cd /d "%~dp0"

:: 2. Delete the broken environment
echo [1/4] Nuking broken .venv...
if exist ".venv" rd /s /q ".venv"

:: 3. Create a fresh, drive-local environment
echo [2/4] Creating fresh local environment on F: drive...
python -m venv .venv
call .venv\Scripts\activate.bat

:: 4. Install CUDA-specific PyTorch (The source of fbgemm.dll)


:: 4. Install CUDA-specific PyTorch (The source of fbgemm.dll)
echo [3/4] Installing CUDA 12.1 + PyTorch 2.4.1...
python -m pip install --upgrade pip
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall

:: 5. Install Unsloth (let it pull its own dependencies)
echo [5/6] Installing Unsloth and all required math kernels...

pip install --upgrade --force-reinstall unsloth
echo ========================================================
echo [PATCH] Enforcing PEFT compatibility (torch 2.4 safe)
echo ========================================================
pip uninstall peft -y
pip install peft==0.18.1 --no-deps
echo [VERIFY] Installed PEFT version:
pip show peft
python -c "import peft, torch; assert peft.__version__ == '0.18.1', 'PEFT VERSION MISMATCH'; assert torch.__version__.startswith('2.4'), 'TORCH VERSION MISMATCH'"

:: 6. Install standard requirements (no torch deps)
echo [6/6] Installing remaining requirements...
pip install --no-deps -r requirements.txt

:: FINAL PATCH: Enforce PEFT compatibility (torch 2.4 safe)
echo ========================================================
echo [PATCH] Enforcing PEFT compatibility (torch 2.4 safe)
echo ========================================================
pip uninstall peft -y
pip install peft==0.18.1 --no-deps
echo [VERIFY] Installed PEFT version:
pip show peft
python -c "import peft, torch; assert peft.__version__ == '0.18.1', 'PEFT VERSION MISMATCH'; assert torch.__version__.startswith('2.4'), 'TORCH VERSION MISMATCH'"

echo.
echo ========================================================
echo [4/4] STARTING TRAINING...
echo ========================================================
:: We use 'python' directly to ensure it uses the activated venv
python scripts/train_unsloth_local.py

pause