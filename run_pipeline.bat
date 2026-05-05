@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo ALETHEIA DAG PIPELINE (STATEFUL ORCHESTRATION)
echo ========================================================

:: Build semantic environment
call forge_semantic.bat
if %errorlevel% neq 0 (
    echo [FATAL] Semantic environment initialization failed.
    exit /b %errorlevel%
)

:: Build training environment
call forge_training.bat
if %errorlevel% neq 0 (
    echo [FATAL] Training environment initialization failed.
    exit /b %errorlevel%
)

:: Handoff to orchestrator (semantic venv)
.venv_semantic\Scripts\python.exe -m src.pipeline.Agent_Forge_orchestrator
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ========================================================
echo [SUCCESS] PIPELINE EXECUTION COMPLETE
echo ========================================================
pause
