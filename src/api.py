"""
Aletheia Knowledge Compiler — FastAPI Gateway
----------------------------------------------
Minimal HTTP surface for triggering ingestion pipelines and polling
run status.  Runs inside the ``api`` Docker service.

Endpoints
---------
POST /ingest          Trigger full pipeline for a GitHub repo URL
POST /score           Score pre-partitioned payloads (skip ingestion)
GET  /status/{run_id} Poll Celery AsyncResult state
GET  /health          Liveness probe (Redis reachability + worker count)
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Aletheia Knowledge Compiler", version="5.3.0")
logger = logging.getLogger("Advocate.API")


# ── Request / Response schemas ─────────────────────────────────

class IngestRequest(BaseModel):
    repo_url: str = Field(..., description="GitHub HTTPS clone URL")
    project_name: Optional[str] = Field(None, description="Override auto-detected project name")


class ScoreRequest(BaseModel):
    payloads: List[Dict[str, Any]] = Field(..., min_length=1)
    mode: str = Field("auto", pattern="^(theorist|coding_assistant|advocate|veteran|auto)$")
    project_name: str = Field("direct_score")


class RunStatus(BaseModel):
    run_id: str
    state: str
    result: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    redis: str
    worker_count: int


# ── Endpoints ──────────────────────────────────────────────────

@app.post("/ingest", response_model=Dict[str, str])
def trigger_ingest(req: IngestRequest):
    """Enqueue the full pipeline: clone → slice → score → format."""
    from src.tasks.pipeline_workflow import launch_full_pipeline

    run_id = launch_full_pipeline(req.repo_url, req.project_name or "")
    return {"run_id": run_id, "status": "dispatched"}


@app.post("/score", response_model=Dict[str, str])
def trigger_score(req: ScoreRequest):
    """Score pre-partitioned payloads without cloning a repo."""
    import uuid
    from celery import chord, group
    from src.tasks.mode_tasks import run_advocate, run_coding, run_theorist, run_veteran
    from src.tasks.pipeline_workflow import collect_and_format, _detect_mode, UnrecognizedModeError
    from src.pipeline.dag_scoring_pass import wrap_payload

    run_id = f"SCORE-{uuid.uuid4().hex[:8].upper()}"

    buckets: Dict[str, List[Dict[str, Any]]] = {
        "theorist": [],
        "coding_assistant": [],
        "advocate": [],
        "veteran": [],
    }

    for p in req.payloads:
        try:
            mode = req.mode if req.mode != "auto" else _detect_mode(p)
        except UnrecognizedModeError:
            raise HTTPException(
                status_code=422,
                detail=f"UNRECOGNIZED_MODE_ERROR: payload keys {sorted(p.keys())} "
                       f"do not match any Identity Matrix mode.",
            )
        wrapped = wrap_payload(p, mode, req.project_name)
        if wrapped:
            buckets[mode].append(wrapped)

    task_map = {
        "theorist": run_theorist,
        "coding_assistant": run_coding,
        "advocate": run_advocate,
        "veteran": run_veteran,
    }

    tasks = [
        task_map[m].s(items, f"{run_id}-{m[:3].upper()}")
        for m, items in buckets.items()
        if items
    ]

    if not tasks:
        raise HTTPException(status_code=400, detail="No valid payloads after wrapping")

    callback = collect_and_format.s(project_name=req.project_name, run_id=run_id)
    chord(group(tasks))(callback)

    return {"run_id": run_id, "status": "dispatched"}


@app.get("/status/{run_id}", response_model=RunStatus)
def get_status(run_id: str):
    """Poll the state of a Celery task / workflow by run_id."""
    from celery.result import AsyncResult
    from src.celery_app import app as celery_app

    res = AsyncResult(run_id, app=celery_app)
    result_data = None
    if res.ready():
        try:
            result_data = res.result if isinstance(res.result, dict) else {"raw": str(res.result)}
        except Exception:
            result_data = {"error": "Result not serializable"}

    return RunStatus(run_id=run_id, state=res.state, result=result_data)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Liveness probe — verify Redis is reachable and at least one worker is up."""
    from src.celery_app import app as celery_app

    # Redis ping
    redis_status = "unreachable"
    try:
        conn = celery_app.connection()
        conn.ensure_connection(max_retries=1, timeout=2)
        conn.close()
        redis_status = "ok"
    except Exception:
        pass

    # Worker count
    worker_count = 0
    try:
        inspect = celery_app.control.inspect(timeout=2)
        ping = inspect.ping() or {}
        worker_count = len(ping)
    except Exception:
        pass

    return HealthResponse(redis=redis_status, worker_count=worker_count)
