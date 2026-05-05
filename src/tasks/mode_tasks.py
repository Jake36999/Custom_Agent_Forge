"""
Celery tasks — one per Identity-Matrix mode.

Each task receives a list of raw payload dicts (already partitioned by mode
upstream), instantiates an isolated DAGRuntime with the correct mode, runs
the scoring state machine, and returns scored + rejected results as plain
dicts (JSON-serializable).

Queues:  theorist | coding | advocate | veteran
"""

import logging
import uuid
from typing import Any, Dict, List

from celery.exceptions import SoftTimeLimitExceeded

from src.celery_app import app

logger = logging.getLogger("Advocate.Tasks.Mode")


# ── helpers ────────────────────────────────────────────────────
def _build_runtime(mode: str, nodes: List[Dict[str, Any]]):
    """Construct a fresh DAGRuntime + graph for the given mode."""
    from src.pipeline.dag_runtime import DAGRuntime, InMemoryEpistemicGraph
    from src.pipeline.acs_engine import ACSExecutionGovernor
    from src.pipeline.dag_scoring_pass import sanitize_scored_node

    graph = InMemoryEpistemicGraph(nodes)
    acs = ACSExecutionGovernor()
    runtime = DAGRuntime(graph=graph, acs=acs, mode=mode)
    return runtime, graph, sanitize_scored_node


def _execute_and_collect(mode: str, payloads: List[Dict[str, Any]], run_id: str) -> Dict[str, Any]:
    """Shared execution logic for all four mode tasks."""
    runtime, graph, sanitize = _build_runtime(mode, payloads)

    logger.info(
        "[%s] run_id=%s  nodes=%d  quarantined=%d",
        mode, run_id, len(graph.nodes), len(graph.quarantine),
    )

    runtime.run()

    scored: List[Dict[str, Any]] = []
    accepted = rejected = 0

    for node in graph.nodes.values():
        d = node.model_dump(exclude_unset=False)
        d = sanitize(d)
        if node.epistemic.state in ("ACCEPTED", "terminal"):
            accepted += 1
        else:
            rejected += 1
        scored.append(d)

    telemetry_snapshot = runtime.telemetry.snapshot() if runtime.telemetry else {}

    return {
        "run_id": run_id,
        "mode": mode,
        "scored_nodes": scored,
        "rejected_traces": runtime.rejected_traces,
        "telemetry": telemetry_snapshot,
        "accepted": accepted,
        "rejected": rejected,
        "quarantined": len(graph.quarantine),
    }


# ── per-mode tasks ────────────────────────────────────────────

@app.task(bind=True, queue="theorist", max_retries=2, acks_late=True,
          time_limit=14460, soft_time_limit=14400, name="mode.run_theorist")
def run_theorist(self, payloads: List[Dict[str, Any]], run_id: str = "") -> Dict[str, Any]:
    """Score theory-mode payloads through the DAG state machine."""
    run_id = run_id or f"THEORIST-{uuid.uuid4().hex[:8].upper()}"
    try:
        return _execute_and_collect("theorist", payloads, run_id)
    except SoftTimeLimitExceeded:
        logger.error("[theorist] Soft time limit exceeded for run_id=%s — failing without retry", run_id)
        raise
    except Exception as exc:
        logger.exception("[theorist] Task failed: %s", exc)
        raise self.retry(exc=exc, countdown=10 * (self.request.retries + 1))


@app.task(bind=True, queue="coding", max_retries=2, acks_late=True,
          time_limit=14460, soft_time_limit=14400, name="mode.run_coding")
def run_coding(self, payloads: List[Dict[str, Any]], run_id: str = "") -> Dict[str, Any]:
    """Score coding-assistant payloads through the DAG state machine."""
    run_id = run_id or f"CODING-{uuid.uuid4().hex[:8].upper()}"
    try:
        return _execute_and_collect("coding_assistant", payloads, run_id)
    except SoftTimeLimitExceeded:
        logger.error("[coding] Soft time limit exceeded for run_id=%s — failing without retry", run_id)
        raise
    except Exception as exc:
        logger.exception("[coding] Task failed: %s", exc)
        raise self.retry(exc=exc, countdown=10 * (self.request.retries + 1))


@app.task(bind=True, queue="advocate", max_retries=2, acks_late=True,
          time_limit=14460, soft_time_limit=14400, name="mode.run_advocate")
def run_advocate(self, payloads: List[Dict[str, Any]], run_id: str = "") -> Dict[str, Any]:
    """Score advocate-mode payloads through the DAG state machine."""
    run_id = run_id or f"ADVOCATE-{uuid.uuid4().hex[:8].upper()}"
    try:
        return _execute_and_collect("advocate", payloads, run_id)
    except SoftTimeLimitExceeded:
        logger.error("[advocate] Soft time limit exceeded for run_id=%s — failing without retry", run_id)
        raise
    except Exception as exc:
        logger.exception("[advocate] Task failed: %s", exc)
        raise self.retry(exc=exc, countdown=10 * (self.request.retries + 1))


@app.task(bind=True, queue="veteran", max_retries=2, acks_late=True,
          time_limit=14460, soft_time_limit=14400, name="mode.run_veteran")
def run_veteran(self, payloads: List[Dict[str, Any]], run_id: str = "") -> Dict[str, Any]:
    """Score veteran-mode payloads through the DAG state machine."""
    run_id = run_id or f"VETERAN-{uuid.uuid4().hex[:8].upper()}"
    try:
        return _execute_and_collect("veteran", payloads, run_id)
    except SoftTimeLimitExceeded:
        logger.error("[veteran] Soft time limit exceeded for run_id=%s — failing without retry", run_id)
        raise
    except Exception as exc:
        logger.exception("[veteran] Task failed: %s", exc)
        raise self.retry(exc=exc, countdown=10 * (self.request.retries + 1))
