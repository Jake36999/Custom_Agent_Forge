# ───────────────────────────────────────────────────────────────
# Aletheia Knowledge Compiler Engine — Docker Image
# Multi-stage build: system deps → Python deps → application
# ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies for tree-sitter, spacy, numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Dependency layer (cached unless requirements change) ──────
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model (small, deterministic)
RUN python -m spacy download en_core_web_sm || true

# ── Application layer ────────────────────────────────────────
FROM deps AS app

COPY src/ src/
COPY scripts/ scripts/
COPY config/ config/
COPY tests/ tests/

# Ensure src is importable
ENV PYTHONPATH=/app

# Default: run the Celery worker (overridden in docker-compose per service)
CMD ["celery", "-A", "src.celery_app", "worker", "--loglevel=info"]
