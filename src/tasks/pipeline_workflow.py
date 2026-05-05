"""
Celery pipeline workflows — orchestrates the full ingestion chain.

Workflows
---------
ingest_and_score : chain
    1. ``ingest_repo`` — clone, AST-slice, cognitive-process a single repo
    2. ``partition_and_fan_out`` — partition nodes by mode, dispatch a ``group``
       of mode-specific scoring tasks (theorist | coding | advocate | veteran)
    3. ``collect_and_format`` — chord callback that merges scored outputs and
       runs the dataset formatter to produce QLoRA JSONL

score_partitioned : chord
    Direct fan-out for pre-partitioned payloads (skip ingestion).
"""

import json
import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List

from celery import chord, group
from celery.exceptions import SoftTimeLimitExceeded

from src.celery_app import app
from src.tasks.mode_tasks import run_advocate, run_coding, run_theorist, run_veteran

logger = logging.getLogger("Advocate.Tasks.Pipeline")

WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", Path(__file__).resolve().parent.parent.parent))
STAGING_DIR = WORKSPACE_ROOT / "staging_point"
OUTPUT_DIR = WORKSPACE_ROOT / "output"
KNOWLEDGE_DIR = OUTPUT_DIR / "knowledge_complexes"
MANIFEST_DIR = KNOWLEDGE_DIR / "manifests"


# ── Step 1: Ingest a single repository ────────────────────────

@app.task(bind=True, queue="pipeline", max_retries=1, acks_late=True,
          time_limit=14460, soft_time_limit=14400, name="pipeline.ingest_repo")
def ingest_repo(self, repo_url: str, project_name: str = "") -> Dict[str, Any]:
    """
    Clone *repo_url*, run the semantic slicer and cognitive processor,
    return the list of extracted payload dicts ready for scoring.
    """
    project_name = project_name or repo_url.rstrip("/").rsplit("/", 1)[-1].replace(".git", "")
    run_id = f"INGEST-{uuid.uuid4().hex[:8].upper()}"
    target_folder = STAGING_DIR / project_name

    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKSPACE_ROOT)

    slicer_script = WORKSPACE_ROOT / "src" / "pipeline" / "semantic_slicer_AG.py"
    processor_script = WORKSPACE_ROOT / "src" / "pipeline" / "cognitive_processor.py"

    try:
        # 1. Clone
        target_folder.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", repo_url, str(target_folder)],
            check=True, capture_output=True, text=True,
            timeout=120,
        )

        # 2. AST Slice
        ast_bundle = target_folder / "raw_ast_bundle.json"
        subprocess.run(
            [
                "python", str(slicer_script), str(target_folder),
                "--base-dir", str(target_folder),
                "--format", "json", "--agent-role", "architecture",
                "--workers", str(max(1, (os.cpu_count() or 2) - 1)),
                "--deterministic", "-o", str(ast_bundle),
            ],
            check=True, capture_output=True, text=True, env=env,
            timeout=14400,
        )

        # 3. Cognitive Processing → Unified YAML
        bundle_parts = sorted(target_folder.glob("raw_ast_bundle*.json"))
        compiled_yamls: List[Path] = []
        for part in bundle_parts:
            suffix = part.stem.replace("raw_ast_bundle", "")
            final_yaml = KNOWLEDGE_DIR / f"KNOWLEDGE_MATRIX_{project_name}{suffix}_UNIFIED.yaml"
            subprocess.run(
                [
                    "python", str(processor_script), str(part),
                    "--output", str(final_yaml),
                    "--agent-prefix", "LogosAgent",
                ],
                check=True, capture_output=True, text=True, env=env,
                timeout=14400,
            )
            compiled_yamls.append(final_yaml)

        # 4. Collect payloads from the generated YAML files
        import yaml as _yaml

        all_payloads: List[Dict[str, Any]] = []
        for ypath in compiled_yamls:
            with open(ypath, "r", encoding="utf-8") as f:
                data = _yaml.safe_load(f)
            if isinstance(data, list):
                all_payloads.extend(d for d in data if isinstance(d, dict))
            elif isinstance(data, dict):
                skills = data.get("capability_injection", {}).get("compiled_skills", [])
                if skills:
                    all_payloads.extend(skills)
                else:
                    all_payloads.append(data)

        # 5. Cleanup staging
        shutil.rmtree(target_folder, ignore_errors=True)

        # 6. Write manifest
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        manifest = {
            "project_name": project_name,
            "repo_url": repo_url,
            "ingestion_status": "Completed",
            "schema_version": "aletheia_skill_dossier_v1",
            "chunks_processed": len(compiled_yamls),
            "run_id": run_id,
        }
        with open(MANIFEST_DIR / f"{project_name}_INGESTION_MANIFEST.json", "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info("[ingest] %s complete — %d payloads extracted", project_name, len(all_payloads))
        return {
            "run_id": run_id,
            "project_name": project_name,
            "payloads": all_payloads,
            "chunks": len(compiled_yamls),
        }

    except SoftTimeLimitExceeded:
        logger.error("[ingest] Soft time limit exceeded for %s (run_id=%s) — aborting without retry", project_name, run_id)
        if target_folder.exists():
            failed_dest = STAGING_DIR / "failed_workspaces" / f"{project_name}_{run_id}_timeout"
            failed_dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(target_folder), str(failed_dest))
        raise
    except Exception as exc:
        logger.exception("[ingest] Failed for %s: %s", project_name, exc)
        if target_folder.exists():
            failed_dest = STAGING_DIR / "failed_workspaces" / f"{project_name}_{run_id}"
            failed_dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(target_folder), str(failed_dest))
        raise self.retry(exc=exc, countdown=30)


# ── Step 2: Partition payloads by mode and fan out ─────────────

# HD4: Hard-stop sentinel for unrecognized mode payloads.
# Replaces the silent coding_assistant fallback to prevent propagation of
# empty-message-array "ghost" nodes through the pipeline.
class UnrecognizedModeError(RuntimeError):
    """Raised when a payload cannot be mapped to any Identity Matrix mode."""


VALID_MODES = frozenset({"theorist", "coding_assistant", "advocate", "veteran"})


def _detect_mode(payload: Dict[str, Any]) -> str:
    """Mirror of dag_scoring_pass._detect_payload_mode."""
    if "traceback" in payload or "error_traceback" in payload:
        return "veteran"
    if "theory_text" in payload and "implementation_code" in payload:
        return "advocate"
    sem = payload.get("semantics") or {}
    mode = sem.get("orchestration_mode", "")
    if mode in VALID_MODES:
        return mode
    if "code_snippet" in payload:
        return "coding_assistant"
    raise UnrecognizedModeError(
        f"UNRECOGNIZED_MODE_ERROR: Payload does not match any Identity Matrix "
        f"mode.  Keys present: {sorted(payload.keys())}"
    )


@app.task(queue="pipeline", acks_late=True,
          time_limit=300, soft_time_limit=270, name="pipeline.partition_and_fan_out")
def partition_and_fan_out(ingest_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Partition ingested payloads by detected mode and dispatch a chord
    that fans out to mode-specific workers, collecting into
    ``collect_and_format``.
    """
    from src.pipeline.dag_scoring_pass import wrap_payload

    payloads = ingest_result.get("payloads", [])
    run_id = ingest_result.get("run_id", uuid.uuid4().hex[:8])
    project_name = ingest_result.get("project_name", "unknown")

    buckets: Dict[str, List[Dict[str, Any]]] = {
        "theorist": [],
        "coding_assistant": [],
        "advocate": [],
        "veteran": [],
    }

    for p in payloads:
        try:
            mode = _detect_mode(p)
        except UnrecognizedModeError:
            logger.error(
                "[partition] UNRECOGNIZED_MODE_ERROR — quarantining payload "
                "with keys=%s for run_id=%s",
                sorted(p.keys()), run_id,
            )
            continue
        wrapped = wrap_payload(p, mode, project_name)
        if wrapped:
            buckets[mode].append(wrapped)

    task_map = {
        "theorist": run_theorist,
        "coding_assistant": run_coding,
        "advocate": run_advocate,
        "veteran": run_veteran,
    }

    tasks = []
    for mode, items in buckets.items():
        if items:
            tasks.append(task_map[mode].s(items, f"{run_id}-{mode[:3].upper()}"))

    if not tasks:
        logger.warning("[partition] No payloads to score for %s", project_name)
        return {"run_id": run_id, "status": "empty", "project_name": project_name}

    callback = collect_and_format.s(project_name=project_name, run_id=run_id)
    chord(group(tasks))(callback)

    return {"run_id": run_id, "status": "dispatched", "project_name": project_name}


# ── Step 3: Collect scored results and format dataset ──────────

@app.task(queue="pipeline", acks_late=True,
          time_limit=600, soft_time_limit=540, name="pipeline.collect_and_format")
def collect_and_format(mode_results: List[Dict[str, Any]], project_name: str = "", run_id: str = "") -> Dict[str, Any]:
    """
    Chord callback — receives scored outputs from all mode workers,
    merges them, writes QLoRA JSONL via dataset_formatter logic.
    """
    all_scored: List[Dict[str, Any]] = []
    total_accepted = 0
    total_rejected = 0
    merged_telemetry: List[Dict[str, Any]] = []

    for result in mode_results:
        all_scored.extend(result.get("scored_nodes", []))
        total_accepted += result.get("accepted", 0)
        total_rejected += result.get("rejected", 0)
        snap = result.get("telemetry")
        if snap:
            merged_telemetry.append(snap)

    # Persist scored nodes as JSON (input to dataset formatter)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scored_file = OUTPUT_DIR / f"dag_scored_{project_name}_{run_id}.json"
    with open(scored_file, "w", encoding="utf-8") as f:
        json.dump(all_scored, f, indent=2, default=str)

    # Persist merged telemetry
    if merged_telemetry:
        tel_file = OUTPUT_DIR / f"telemetry_{project_name}_{run_id}.json"
        with open(tel_file, "w", encoding="utf-8") as f:
            json.dump(merged_telemetry, f, indent=2, default=str)

    logger.info(
        "[collect] %s run=%s  accepted=%d  rejected=%d  total=%d",
        project_name, run_id, total_accepted, total_rejected, len(all_scored),
    )

    return {
        "run_id": run_id,
        "project_name": project_name,
        "status": "completed",
        "total_nodes": len(all_scored),
        "accepted": total_accepted,
        "rejected": total_rejected,
        "scored_file": str(scored_file),
    }


# ── Convenience: full chain (ingest → partition → score → format) ──

def launch_full_pipeline(repo_url: str, project_name: str = "") -> str:
    """
    Kick off the complete ingestion → scoring → formatting pipeline.
    Returns the ``run_id`` for status polling.
    """
    from celery import chain as celery_chain

    run_id = f"PIPE-{uuid.uuid4().hex[:8].upper()}"

    workflow = celery_chain(
        ingest_repo.s(repo_url, project_name),
        partition_and_fan_out.s(),
    )
    workflow.apply_async()
    return run_id
