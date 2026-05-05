@echo off
echo [SEMANTIC FORGE]
if not exist ".venv_semantic" (
    python -m venv .venv_semantic
)
if not exist ".venv_semantic\Scripts\python.exe" (
    echo [FATAL] Semantic venv creation failed
    exit /b 1
)
set PYTHON=.venv_semantic\Scripts\python.exe
set PIP=.venv_semantic\Scripts\pip.exe

:: Use python -m to avoid Windows executable locks
%PYTHON% -m pip install --upgrade pip
%PIP% install -r requirements_semantic.txt

:: Direct URL bypasses the 404 versioning bug
%PIP% install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz
echo [VERIFY] SentenceTransformer runtime:
%PYTHON% -c "from sentence_transformers import SentenceTransformer; m=SentenceTransformer('all-MiniLM-L6-v2'); m.encode('test')"
echo [OK] Semantic environment ready
