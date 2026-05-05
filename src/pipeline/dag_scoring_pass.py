"""
Aletheia DAG Scoring Pass v3.0
-------------------------------
Out-of-core streaming DAG compilation with OS-level subprocess isolation.

Architectural contract: host RAM must stay below 32 GB across the full run.
v2.0 enforced this via per-chunk del + gc.collect(), which prevents Python-level
reference accumulation but cannot return CPython arena memory to the OS.
v3.0 adds a second isolation tier: a supervisor/worker split where each batch
of files is processed in a dedicated child process that exits after its work,
allowing the OS to fully reclaim all arena memory — including fragmented blocks
that gc.collect() cannot recover.

Execution model:
  Supervisor (run_scoring_pass) — lightweight coordinator; never loads node data.
    1. Collects all input files and partitions into batches of _SCORE_BATCH_SIZE.
    2. For each batch: writes a temp manifest, spawns an isolated worker subprocess,
       reads the worker's JSON summary from stdout, aggregates totals.
    3. Returns aggregated summary when all batches complete.

  Worker (--worker flag) — ephemeral child process; handles exactly one batch.
    Runs the same four-phase scoring logic as v2.0 (contract → sparsity → PageRank
    → JSONL flush), appends results to the shared master JSONL sinks, then exits.
    OS reclaims all memory. Parent RSS stays flat for the entire run.

Four phases applied per-file inside each worker:
  Phase 1 — Streaming ingestion + contract enforcement
  Phase 2 — Early rejection gate (topological sparsity check)
  Phase 3 — Adaptive statistical thresholding (scale-invariant z-score)
  Phase 4 — Incremental JSONL I/O + aggressive memory teardown

Env knobs:
  ALETHEIA_SCORE_BATCH  — files per worker batch (default: 50)
  ALETHEIA_MIN_NODES    — Phase 2 sparsity floor (default: 5)
  ALETHEIA_PAGERANK_K   — Phase 3 z-score multiplier K (default: 1.5)

Called by Agent_Forge_orchestrator between Step 2 (extraction) and
Step 3 (formatting). Output: dag_scored_nodes.jsonl / dag_rejected_nodes.jsonl
"""

import gc
import json
import logging
import os
import statistics
import subprocess
import sys
import tempfile
import uuid
import yaml
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("Advocate.DAGScoringPass")

try:
    import argparse
    from src.core.models import AletheiaSkill  # noqa: F401 — kept for wrap_payload consumers
    from src.pipeline.contracts import MODE_CONTRACTS
except ImportError as e:
    logger.error(f"Failed to import core modules: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Mirror of dag_runtime.py:108 — single source of truth lives there
SENTINEL_WARN_QUARANTINE = "[DAG_WARN_NODE_QUARANTINED]"

# Derived from MODE_CONTRACTS (contracts.py) — the canonical list of valid modes.
_VALID_MODES: frozenset = frozenset(MODE_CONTRACTS.keys())

# Minimum intra-chunk node count to justify nx.DiGraph construction (Phase 2)
_MIN_VALID_NODES: int = int(os.environ.get("ALETHEIA_MIN_NODES", "5"))

# Z-score multiplier K for adaptive PageRank threshold (Phase 3)
_PAGERANK_K: float = float(os.environ.get("ALETHEIA_PAGERANK_K", "1.5"))

# PageRank damping factor
_PAGERANK_ALPHA: float = 0.85

# Files processed per worker subprocess before the child exits and OS reclaims RAM.
# Lower = more subprocess overhead but better memory isolation.
# Higher = fewer spawns but larger peak RSS per child.
_SCORE_BATCH_SIZE: int = int(os.environ.get("ALETHEIA_SCORE_BATCH", "50"))


# ---------------------------------------------------------------------------
# Trajectory Builder
# ---------------------------------------------------------------------------

def _build_veteran_trajectory(
    execution_pattern: List[str],
    traceback_text: str,
    diff: str,
    resolved_code: str,
) -> Dict[str, Any]:
    """Build a reasoning_trajectory dict with confusion_matrix and response_vector."""
    has_tb = 1.0 if traceback_text else 0.0
    has_diff = 1.0 if diff else 0.0
    has_res = 1.0 if resolved_code else 0.0

    confusion_matrix: List[List[float]] = []
    response_vector: List[float] = []
    steps: List[Dict[str, Any]] = []

    for step_label in execution_pattern:
        sl = step_label.lower()
        if sl in ("fix", "resolution", "resolved"):
            row = [has_tb, has_diff, has_res, 1.0]
            rv = 1.0
        elif sl in ("error", "traceback", "failure"):
            row = [has_tb, 0.0, 0.0, 0.0]
            rv = 0.0
        elif sl in ("diagnosis", "analysis", "diagnostic"):
            row = [has_tb, 0.0, 0.0, 0.5]
            rv = 0.0
        else:
            row = [0.0, 0.0, 0.0, 0.5]
            rv = 0.0
        confusion_matrix.append(row)
        response_vector.append(rv)
        steps.append({
            "attempt_code": "",
            "error": "" if sl not in ("error", "traceback", "failure") else "diagnostic step",
            "diagnosis": step_label if sl in ("diagnosis", "analysis", "diagnostic") else "",
            "fix": "" if rv == 0.0 else "resolution applied",
        })

    return {
        "steps": steps,
        "confusion_matrix": confusion_matrix,
        "response_vector": response_vector,
        "labels": execution_pattern,
        "metadata": {
            "has_traceback": bool(traceback_text),
            "has_diff": bool(diff),
            "has_resolution": bool(resolved_code),
        },
    }


# ---------------------------------------------------------------------------
# Payload → AletheiaSkill converters (one per extraction mode)
# ---------------------------------------------------------------------------

def _wrap_veteran_payload(payload: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    """Convert veteran extraction JSON into an AletheiaSkill-shaped dict."""
    node_id = payload.get("node_id", f"vet_{uuid.uuid4().hex[:8]}")
    code = payload.get("resolved_code", payload.get("attempt_code", ""))
    traceback_text = payload.get("error_traceback", payload.get("traceback", ""))
    diff = payload.get("successful_diff", payload.get("diff", ""))

    execution_pattern = payload.get("execution_pattern", ["attempt", "error", "diagnosis", "fix"])
    trajectory = _build_veteran_trajectory(execution_pattern, traceback_text, diff, code)

    return {
        "node_id": node_id,
        "name": f"veteran_diagnostic_{node_id}",
        "file": source_file,
        "code_snippet": code,
        "imports": [],
        "operator_type": "veteran_diagnostic",
        "source_type": "veteran_diagnostic",
        "skill_type": "execution",
        "teaching_layer": {
            "skill_identity": {"name": f"veteran_diagnostic_{node_id}"},
            "method_metadata": {"name": node_id, "language": "python"},
            "reasoning_vectors": {
                "intent": "Diagnose and resolve execution failure",
                "strategy": "Error trace analysis → root cause → corrective diff",
                "constraints": [f"traceback_present={bool(traceback_text)}"],
                "execution_pattern": execution_pattern,
                "failure_modes": ["incomplete_traceback", "empty_diff"],
            },
            "implementation_template": {"code": code},
        },
        "semantics": {
            "traceback": traceback_text,
            "diff": diff,
            "resolved_code": code,
            "orchestration_mode": "veteran",
            "reasoning_trajectory": trajectory,
        },
        "epistemic": {"state": "CREATED", "c_node": 0.5},
    }


def _wrap_advocate_payload(payload: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    """Convert advocate extraction JSON into an AletheiaSkill-shaped dict."""
    node_id = payload.get("node_id", f"adv_{uuid.uuid4().hex[:8]}")
    theory = payload.get("theory_text", "")
    impl = payload.get("implementation_code", "")
    ratio = payload.get("text_to_code_ratio", 0.0)

    return {
        "node_id": node_id,
        "name": f"advocate_theory_{node_id}",
        "file": source_file,
        "code_snippet": impl,
        "imports": [],
        "operator_type": "advocate_theory",
        "source_type": "advocate_theory",
        "skill_type": "execution",
        "teaching_layer": {
            "skill_identity": {"name": f"advocate_theory_{node_id}"},
            "method_metadata": {"name": node_id, "language": "python"},
            "reasoning_vectors": {
                "intent": "Architectural critique with implementation",
                "strategy": "Dialectical adversarial review",
                "constraints": [f"text_to_code_ratio={ratio:.3f}"],
                "execution_pattern": ["theory", "critique", "implementation"],
                "failure_modes": ["insufficient_theory", "low_semantic_density"],
            },
            "implementation_template": {"code": impl},
        },
        "semantics": {
            "theory": theory,
            "implementation": impl,
            "orchestration_mode": "advocate",
        },
        "epistemic": {"state": "CREATED", "c_node": 0.5},
    }


def _wrap_coding_payload(payload: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    """Ensure coding_assistant payloads have all required AletheiaSkill fields."""
    if "node_id" not in payload:
        payload["node_id"] = f"code_{uuid.uuid4().hex[:8]}"
    if "operator_type" not in payload or not payload["operator_type"]:
        payload["operator_type"] = "function"
    if "semantics" not in payload or not isinstance(payload.get("semantics"), dict):
        payload["semantics"] = payload.get("semantics") or {}
    if "orchestration_mode" not in payload.get("semantics", {}):
        payload.setdefault("semantics", {})["orchestration_mode"] = "coding_assistant"
    sem = payload.get("semantics", {})
    if "code_snippet" in sem and "implementation" not in sem:
        sem["implementation"] = sem.pop("code_snippet")
    for _k in ("name", "extracted_patterns"):
        sem.pop(_k, None)
    if "epistemic" not in payload:
        payload["epistemic"] = {"state": "CREATED", "c_node": 0.5}
    elif isinstance(payload["epistemic"], dict) and "state" not in payload["epistemic"]:
        payload["epistemic"]["state"] = "CREATED"
    return payload


def _detect_payload_mode(payload: Dict[str, Any]) -> str:
    """Infer extraction mode from payload structure."""
    if "traceback" in payload or "error_traceback" in payload:
        return "veteran"
    if "theory_text" in payload and "implementation_code" in payload:
        return "advocate"
    if "node_id" in payload and "code_snippet" in payload:
        return "coding_assistant"
    return "unknown"


# ---------------------------------------------------------------------------
# Output sanitization
# ---------------------------------------------------------------------------

_INTERNAL_SEMANTICS_KEYS = frozenset({
    "system_centrality_blast_radius",
    "_acs_trajectory",
    "_governance_directive",
    "_acs_structured_constraints",
    "_reroll_context",
})

_INTERNAL_NODE_KEYS = frozenset({
    "sie_node",
    "v_score",
    "acs_handshake_sid",
    "acs_violations",
    "acs_audited",
})


def sanitize_scored_node(node_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Strip internal scoring mechanics so the LLM learns domain logic, not heuristics."""
    for k in _INTERNAL_NODE_KEYS:
        node_dict.pop(k, None)

    sem = node_dict.get("semantics")
    if isinstance(sem, dict):
        for k in _INTERNAL_SEMANTICS_KEYS:
            sem.pop(k, None)

    ep = node_dict.get("epistemic")
    if isinstance(ep, dict):
        ep.pop("confidence", None)

    return node_dict


def wrap_payload(payload: Dict[str, Any], mode: str, source_file: str) -> Optional[Dict[str, Any]]:
    """Convert a raw extraction payload to AletheiaSkill-shaped dict."""
    detected = mode if mode != "auto" else _detect_payload_mode(payload)

    if detected == "veteran":
        return _wrap_veteran_payload(payload, source_file)
    elif detected == "advocate":
        return _wrap_advocate_payload(payload, source_file)
    elif detected in ("coding_assistant", "unknown"):
        return _wrap_coding_payload(payload, source_file)
    return _wrap_coding_payload(payload, source_file)


# ---------------------------------------------------------------------------
# Phase helpers — shared by both worker and any direct callers
# ---------------------------------------------------------------------------

def _iter_subgraph_payloads(input_dir: Path):
    """Generator: yield (file_path, raw_dict) one file at a time."""
    for file_path in sorted(input_dir.glob("*.*")):
        if file_path.suffix not in (".json", ".yaml", ".yml"):
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) if file_path.suffix in (".yaml", ".yml") else json.load(fh)
        except Exception as exc:
            logger.warning(f"Failed to parse {file_path.name}: {exc}")
            continue
        if raw is not None:
            yield file_path, raw


def _extract_raw_nodes(data: Any) -> List[Dict[str, Any]]:
    """Flatten the various file-format shapes into a flat list of node dicts."""
    nodes: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, list):
                nodes.extend(n for n in item if isinstance(n, dict))
            elif isinstance(item, dict):
                nodes.append(item)
    elif isinstance(data, dict):
        skills = data.get("capability_injection", {}).get("compiled_skills", [])
        if skills:
            nodes.extend(skills)
        elif "validated_nodes" in data or "rejected_nodes" in data:
            for key in ("validated_nodes", "rejected_nodes"):
                batch = data.get(key) or []
                if isinstance(batch, list):
                    nodes.extend(n for n in batch if isinstance(n, dict))
        else:
            nodes.append(data)
    return nodes


def _enforce_contract(wrapped: Dict[str, Any], valid_modes: frozenset) -> Tuple[bool, str]:
    """Phase 1 hard-rejection gate: check node mode against system contracts."""
    sem = wrapped.get("semantics") or {}
    node_mode = sem.get("orchestration_mode", "")
    if not node_mode:
        return False, "missing orchestration_mode"
    if node_mode not in valid_modes:
        return False, f"mode '{node_mode}' not in valid_modes={sorted(valid_modes)}"
    return True, ""


def _derive_raw_edges(nodes: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Build an intra-chunk edge list from topology_cluster and dependencies."""
    edges: List[Tuple[str, str]] = []
    for node in nodes:
        nid = node.get("node_id")
        if not nid:
            continue
        tc = node.get("topology_cluster") or {}
        for cid in (tc.get("child_ids") or []):
            if isinstance(cid, str):
                edges.append((nid, cid))
        deps = node.get("dependencies") or {}
        if isinstance(deps, dict):
            for did in (deps.get("downstream_calls") or []):
                if isinstance(did, str):
                    edges.append((nid, did))
        for uid in (node.get("upstream_callers") or []):
            if isinstance(uid, str):
                edges.append((uid, nid))
    return edges


def _resolve_chunk_mode(raw_nodes: List[Dict[str, Any]], global_mode: str) -> str:
    """Return the effective processing mode for this chunk."""
    if global_mode not in ("auto", "hybrid"):
        return global_mode
    counts = Counter(_detect_payload_mode(n) for n in raw_nodes)
    dominant = counts.most_common(1)[0][0] if counts else "coding_assistant"
    if dominant not in _VALID_MODES:
        dominant = "coding_assistant"
    return dominant


# ---------------------------------------------------------------------------
# Per-file four-phase scoring kernel — called inside worker child processes
# ---------------------------------------------------------------------------

def _process_single_file(
    file_path: Path,
    raw_data: Any,
    f_acc,
    f_rej,
    mode: str,
) -> Dict[str, int]:
    """Run the four-phase DAG scoring pipeline on one pre-loaded file.

    Appends accepted/rejected nodes to the open JSONL file handles and returns
    per-file counters. Designed to be called from a worker child process so
    that the OS reclaims all allocations when the child exits.
    """
    # Phase 1: contract enforcement
    raw_nodes = _extract_raw_nodes(raw_data)
    if not raw_nodes:
        del raw_data, raw_nodes
        return {"skipped_empty": 1}

    chunk_mode = _resolve_chunk_mode(raw_nodes, mode)

    valid_nodes: List[Dict[str, Any]] = []
    contract_rej: List[Dict[str, Any]] = []

    for raw_node in raw_nodes:
        wrapped = wrap_payload(raw_node, chunk_mode, file_path.name)
        if wrapped is None:
            continue
        ok, reason = _enforce_contract(wrapped, _VALID_MODES)
        if not ok:
            logger.warning(
                f"{SENTINEL_WARN_QUARANTINE} Evicting node "
                f"{wrapped.get('node_id')} from {file_path.name}: {reason}"
            )
            wrapped["_rejection_reason"] = f"contract_violation:{reason}"
            contract_rej.append(wrapped)
        else:
            valid_nodes.append(wrapped)

    for node in contract_rej:
        f_rej.write(json.dumps(node, default=str) + "\n")
    n_contract_rejected = len(contract_rej)

    # Phase 2: topological sparsity gate
    # TRUSTED_MODE bypass: standalone extraction modes (veteran, advocate) produce
    # isolated nodes with no intra-file edges. Skip the graph gate and treat every
    # valid node as uniformly scored so none are silently dropped.
    _trusted = os.environ.get("ALETHEIA_TRUSTED_MODE", "0") == "1"

    n_valid = len(valid_nodes)
    edges = _derive_raw_edges(valid_nodes)
    n_edges = len(edges)

    if (n_valid < _MIN_VALID_NODES or n_edges == 0) and not _trusted:
        for node in valid_nodes:
            node["_rejection_reason"] = (
                f"topological_sparsity(n_valid={n_valid},n_edges={n_edges})"
            )
            f_rej.write(json.dumps(node, default=str) + "\n")
        del raw_data, raw_nodes, valid_nodes, contract_rej, edges
        gc.collect()
        return {"contract_rejected": n_contract_rejected, "sparsity_rejected": n_valid, "processed": 1}

    # Phase 3: nx.DiGraph construction + adaptive statistical scoring
    # TRUSTED_MODE with no edges: assign uniform score; all nodes accepted.
    if _trusted and n_edges == 0:
        chunk_accepted = 0
        for node in valid_nodes:
            node["local_score"] = 1.0
            node["z_score"] = 0.0
            ep = node.get("epistemic")
            if isinstance(ep, dict):
                ep["state"] = "ACCEPTED"
            node_out = sanitize_scored_node(dict(node))
            f_acc.write(json.dumps(node_out, default=str) + "\n")
            chunk_accepted += 1
        f_acc.flush()
        f_rej.flush()
        logger.info(
            f"[file] {file_path.name} (trusted, no-edge): "
            f"accepted={chunk_accepted} via uniform scoring"
        )
        del raw_data, raw_nodes, valid_nodes, contract_rej, edges
        gc.collect()
        return {"contract_rejected": n_contract_rejected, "accepted": chunk_accepted, "processed": 1}

    node_id_set = {n["node_id"] for n in valid_nodes}
    local_graph = nx.DiGraph()

    for node in valid_nodes:
        local_graph.add_node(node["node_id"])
    for src, dst in edges:
        if src in node_id_set and dst in node_id_set:
            local_graph.add_edge(src, dst)

    try:
        raw_scores: Dict[str, float] = nx.pagerank(local_graph, alpha=_PAGERANK_ALPHA)
    except Exception as exc:
        logger.warning(
            f"{file_path.name}: PageRank failed ({exc}), falling back to degree_centrality"
        )
        raw_scores = nx.degree_centrality(local_graph)

    score_values = list(raw_scores.values())
    mu: float = statistics.mean(score_values)
    sigma: float = statistics.stdev(score_values) if len(score_values) > 1 else 0.0
    threshold: float = mu + (_PAGERANK_K * sigma)

    chunk_accepted = 0
    chunk_rejected = 0

    for node in valid_nodes:
        nid = node["node_id"]
        local_score = raw_scores.get(nid, 0.0)
        z_score = (local_score - mu) / sigma if sigma > 0.0 else 0.0
        node["local_score"] = local_score
        node["z_score"] = z_score

        if local_score >= threshold:
            ep = node.get("epistemic")
            if isinstance(ep, dict):
                ep["state"] = "ACCEPTED"
            node_out = sanitize_scored_node(dict(node))
            f_acc.write(json.dumps(node_out, default=str) + "\n")
            chunk_accepted += 1
        else:
            node_out = sanitize_scored_node(dict(node))
            node_out["_rejection_reason"] = (
                f"below_adaptive_threshold("
                f"score={local_score:.6f},"
                f"mu={mu:.6f},"
                f"sigma={sigma:.6f},"
                f"K={_PAGERANK_K})"
            )
            f_rej.write(json.dumps(node_out, default=str) + "\n")
            chunk_rejected += 1

    f_acc.flush()
    f_rej.flush()

    logger.info(
        f"[file] {file_path.name}: "
        f"accepted={chunk_accepted} rejected={chunk_rejected} "
        f"mu={mu:.4f} sigma={sigma:.4f} threshold={threshold:.4f}"
    )

    # Phase 4: aggressive memory teardown
    local_graph.clear()
    del (
        local_graph, raw_data, raw_nodes,
        valid_nodes, contract_rej, edges,
        raw_scores, score_values, node_id_set,
    )
    gc.collect()

    return {
        "accepted": chunk_accepted,
        "rejected": chunk_rejected,
        "contract_rejected": n_contract_rejected,
        "processed": 1,
    }


# ---------------------------------------------------------------------------
# Worker entry point — runs inside an isolated child process
# ---------------------------------------------------------------------------

def _worker_main(batch_manifest: str, output_dir_str: str, mode: str) -> None:
    """Process a batch of files and append results to the shared master JSONL sinks.

    This function is the entire body of an isolated child process. When it returns,
    the child calls sys.exit(0) and the OS reclaims all arena memory — including
    fragmented CPython allocator blocks that gc.collect() cannot recover.

    Output is appended (not written) so the parent's master sinks accumulate
    results across all batches without rewriting the file.
    """
    output_dir = Path(output_dir_str)
    accepted_path = output_dir / "dag_scored_nodes.jsonl"
    rejected_path = output_dir / "dag_rejected_nodes.jsonl"

    batch_files = [
        Path(p.strip())
        for p in Path(batch_manifest).read_text(encoding="utf-8").splitlines()
        if p.strip()
    ]

    totals: Dict[str, int] = {
        "accepted": 0, "rejected": 0,
        "contract_rejected": 0, "sparsity_rejected": 0,
        "processed": 0, "skipped_empty": 0,
    }

    with open(accepted_path, "a", encoding="utf-8") as f_acc, \
         open(rejected_path, "a", encoding="utf-8") as f_rej:

        for file_path in batch_files:
            if file_path.suffix not in (".json", ".yaml", ".yml"):
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as fh:
                    raw = (
                        yaml.safe_load(fh)
                        if file_path.suffix in (".yaml", ".yml")
                        else json.load(fh)
                    )
            except Exception as exc:
                logger.warning(f"Failed to parse {file_path.name}: {exc}")
                continue
            if raw is None:
                totals["skipped_empty"] += 1
                continue

            stats = _process_single_file(file_path, raw, f_acc, f_rej, mode)
            for k, v in stats.items():
                totals[k] = totals.get(k, 0) + v

    # Print summary as the last stdout line so the supervisor can parse it.
    print(json.dumps(totals))


# ---------------------------------------------------------------------------
# Supervisor entry point — spawns isolated workers and aggregates results
# ---------------------------------------------------------------------------

def run_scoring_pass(
    input_dir: Path,
    output_dir: Path,
    mode: str = "auto",
    config_path: str = "config/engine_config.yaml",
) -> Dict[str, Any]:
    """Partition input files into batches, spawn one isolated worker per batch.

    The supervisor process never loads node data. Its RSS stays flat throughout
    the entire run. Each worker child loads its batch, scores it, flushes to
    the shared master JSONL sinks, and exits — returning all arena memory to
    the OS before the next batch begins.
    """
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return {"status": "error", "reason": "input_dir_missing"}

    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = output_dir / "dag_scored_nodes.jsonl"
    rejected_path = output_dir / "dag_rejected_nodes.jsonl"

    # Initialize empty master sinks before any worker appends to them.
    accepted_path.write_text("", encoding="utf-8")
    rejected_path.write_text("", encoding="utf-8")

    all_files = sorted(
        f for f in input_dir.glob("*.*")
        if f.suffix in (".json", ".yaml", ".yml")
    )
    if not all_files:
        logger.warning("No input files found.")
        return {
            "status": "completed",
            "batches": 0,
            "total_files": 0,
            "chunks_processed": 0,
            "chunks_skipped_empty": 0,
            "nodes_accepted": 0,
            "nodes_rejected": 0,
            "contract_rejected": 0,
            "sparsity_rejected": 0,
            "output_accepted": str(accepted_path),
            "output_rejected": str(rejected_path),
        }

    batches = [
        all_files[i : i + _SCORE_BATCH_SIZE]
        for i in range(0, len(all_files), _SCORE_BATCH_SIZE)
    ]
    logger.info(
        f"Partitioned {len(all_files)} files into {len(batches)} batches "
        f"of ≤{_SCORE_BATCH_SIZE} (ALETHEIA_SCORE_BATCH={_SCORE_BATCH_SIZE})"
    )

    totals: Dict[str, int] = {
        "accepted": 0, "rejected": 0,
        "contract_rejected": 0, "sparsity_rejected": 0,
        "processed": 0, "skipped_empty": 0,
    }

    script_path = str(Path(__file__).resolve())

    for batch_idx, batch in enumerate(batches):
        batch_manifest_path: Optional[str] = None
        try:
            # Write the batch manifest to a temp file that the worker reads.
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write("\n".join(str(f) for f in batch))
                batch_manifest_path = tmp.name

            logger.info(
                f"[batch {batch_idx + 1}/{len(batches)}] "
                f"Spawning worker for {len(batch)} files..."
            )

            worker_cmd = [
                sys.executable, script_path,
                "--worker",
                "--batch-manifest", batch_manifest_path,
                "--output-dir", str(output_dir),
                "--mode", mode,
            ]

            result = subprocess.run(
                worker_cmd,
                capture_output=True,
                text=True,
                timeout=None,
                errors="replace",
            )

            if result.returncode != 0:
                logger.error(
                    f"[batch {batch_idx + 1}] Worker exited with code "
                    f"{result.returncode}: {result.stderr[:500]}"
                )
            else:
                # Forward worker's INFO logs to the supervisor's logger.
                if result.stderr:
                    for line in result.stderr.splitlines():
                        logger.debug(f"  [worker] {line}")
                # Parse the summary JSON from the last stdout line.
                stdout_lines = [l for l in result.stdout.splitlines() if l.strip()]
                if stdout_lines:
                    try:
                        stats = json.loads(stdout_lines[-1])
                        for k, v in stats.items():
                            totals[k] = totals.get(k, 0) + v
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"[batch {batch_idx + 1}] Could not parse worker summary: {e}"
                        )

        except Exception as exc:
            logger.error(f"[batch {batch_idx + 1}] Worker spawn failed: {exc}")

        finally:
            if batch_manifest_path:
                try:
                    Path(batch_manifest_path).unlink(missing_ok=True)
                except Exception:
                    pass

    summary = {
        "status": "completed",
        "batches": len(batches),
        "total_files": len(all_files),
        "chunks_processed": totals.get("processed", 0),
        "chunks_skipped_empty": totals.get("skipped_empty", 0),
        "nodes_accepted": totals.get("accepted", 0),
        "nodes_rejected": totals.get("rejected", 0),
        "contract_rejected": totals.get("contract_rejected", 0),
        "sparsity_rejected": totals.get("sparsity_rejected", 0),
        "output_accepted": str(accepted_path),
        "output_rejected": str(rejected_path),
    }
    logger.info(f"DAG scoring pass complete: {summary}")
    return summary


# ---------------------------------------------------------------------------
# CLI entry point — routes to worker or supervisor based on --worker flag
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse  # noqa: F811

    parser = argparse.ArgumentParser(
        description="Aletheia DAG Scoring Pass v3.0 (subprocess-isolated, 18k+ subgraphs)"
    )

    # Supervisor args
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=None,
        help="[Supervisor] Path to knowledge_complexes directory",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (defaults to input_dir in supervisor mode)",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "theorist", "coding_assistant", "advocate", "veteran", "hybrid"],
        default="auto",
    )
    parser.add_argument(
        "--config",
        default="config/engine_config.yaml",
    )

    # Worker args (used internally by the supervisor — not for direct invocation)
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Run as an isolated batch worker (spawned by the supervisor).",
    )
    parser.add_argument(
        "--batch-manifest",
        default=None,
        help="[Worker] Path to temp file listing batch file paths.",
    )

    args = parser.parse_args()

    if args.worker:
        if not args.batch_manifest or not args.output_dir:
            logger.error("--worker requires --batch-manifest and --output-dir")
            sys.exit(1)
        _worker_main(args.batch_manifest, args.output_dir, args.mode or "auto")
        sys.exit(0)
    else:
        if not args.input_dir:
            logger.error("input_dir is required in supervisor mode")
            sys.exit(1)
        input_path = Path(args.input_dir)
        output_path = Path(args.output_dir) if args.output_dir else input_path
        result = run_scoring_pass(
            input_path, output_path,
            mode=args.mode,
            config_path=args.config,
        )
        sys.exit(0 if result.get("status") == "completed" else 1)
