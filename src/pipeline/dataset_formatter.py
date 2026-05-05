
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import yaml
import argparse
import hashlib
import math
import uuid
import traceback
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import jsonlines
from pydantic import ValidationError as PydanticValidationError
from src.core.models import AletheiaSkill
from src.validation.pipeline_firewall import validate_skill, enforce_semantic_firewall, DriftViolation


DEFAULT_ADVOCATE_AUDITS_DIR = Path("output") / "advocate_audits"

def _format_dataset_legacy(input_dir, output_file):
    import gc
    input_path = Path(input_dir)
    yaml_files = list(input_path.glob("*.yaml"))
    print(f"[*] Found {len(yaml_files)} YAML matrices in {input_dir}")

    rejected_path = "output/qlora_rejected_skills.jsonl"
    hasher = hashlib.sha256()
    accepted_count = 0
    rejected_count = 0

    with open(output_file, 'w', encoding='utf-8') as f_acc, \
         open(rejected_path, 'w', encoding='utf-8') as f_rej:

        for yml_file in yaml_files:
            with open(yml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            if isinstance(data, list):
                skills = data
            elif isinstance(data, dict):
                if data.get("status") == "failed":
                    f_rej.write(json.dumps({
                        "node_id": "dossier_failed",
                        "reason": data.get("reason", "Unknown failure"),
                    }) + "\n")
                    rejected_count += 1
                    print(f"[LEGACY-SENTINEL-REJECT] Dossier-level failure in {yml_file.name}")
                    del data
                    continue
                skills = data.get("capability_injection", {}).get("compiled_skills", [])
            else:
                del data
                continue

            # Defensive default: bind `output` BEFORE the try block so that any
            # exception inside validation, code_snippet access, or f-string
            # evaluation cannot leave the name unbound.
            output = ""
            for skill_dict in skills:
                output = ""
                try:
                    # 1. Pydantic Strict Validation
                    skill_obj = AletheiaSkill.model_validate(skill_dict)

                    # 2. Epistemic & Canonical Firewall
                    diagnostic = validate_skill(skill_obj)

                    if diagnostic["status"] == "rejected":
                        f_rej.write(json.dumps({
                            "node_id": skill_obj.node_id if hasattr(skill_obj, 'node_id') else getattr(skill_obj, 'id', 'unknown'),
                            "reason": diagnostic.get('reason', 'Validation failed'),
                            "trace": diagnostic.get('trace', 'N/A'),
                        }) + "\n")
                        rejected_count += 1
                        print(f"[LEGACY-SENTINEL-REJECT] Node '{skill_obj.node_id}' rejected by firewall")
                        continue

                    # 3. Format for QLoRA
                    code_text = skill_obj.code_snippet or ""
                    semantics = getattr(skill_obj, "semantics", None) or {}
                    if not isinstance(semantics, dict):
                        semantics = {}
                    output = semantics.get("output", "") or ""
                    if not output:
                        output = (
                            f"```python\n{code_text}\n```"
                            if code_text
                            else "```python\n# <empty code slice>\n```"
                        )

                    vectors = skill_obj.teaching_layer.reasoning_vectors
                    if vectors:
                        intent = vectors.intent or "Derived execution intent"
                        strategy = vectors.strategy or "AST Translation"
                        constraints = ', '.join(vectors.constraints) if vectors.constraints else "None"
                        exec_pattern = ', '.join(vectors.execution_pattern) if vectors.execution_pattern else "None"
                        fail_modes = ', '.join(vectors.failure_modes) if vectors.failure_modes else "None"
                    else:
                        intent, strategy, constraints = "Derived execution intent", "AST Translation", "None"
                        exec_pattern, fail_modes = "None", "None"

                    instruction = (
                        "Implement the following logic based on this reasoning:\n"
                        f"- Intent: {intent}\n"
                        f"- Strategy: {strategy}\n"
                        f"- Constraints: {constraints}\n"
                        f"- Execution Steps: {exec_pattern}\n"
                        f"- Failure Modes: {fail_modes}\n"
                    )
                    line = json.dumps({"instruction": instruction, "output": output}) + "\n"
                    f_acc.write(line)
                    hasher.update(line.encode())
                    accepted_count += 1

                except (PydanticValidationError, ValueError, AttributeError, KeyError, TypeError) as e:
                    f_rej.write(json.dumps({
                        "node_id": skill_dict.get("node_id", "unknown"),
                        "reason": f"{type(e).__name__}: {e}",
                        "output_state": "bound" if output else "empty",
                    }) + "\n")
                    rejected_count += 1
                    print(f"[LEGACY-SENTINEL-VAL] Validation error in node "
                          f"'{skill_dict.get('node_id', 'unknown')}': {type(e).__name__}: {e}")
                except NameError as e:
                    # NameError here is a programmer bug, not a data-validation failure.
                    # The defensive `output = ""` hoist above should make this path
                    # unreachable for the historical "name 'output' is not defined"
                    # crash; if it ever fires again, log loudly with SENTINEL-EXEC-ERR
                    # so it can never again be silently buried in the rejection log.
                    tb = traceback.format_exc()
                    print(f"[SENTINEL-EXEC-ERR] NameError in node '{skill_dict.get('node_id', 'unknown')}': {e}")
                    print(f"[SENTINEL-EXEC-ERR] Traceback:\n{tb}")
                    f_rej.write(json.dumps({
                        "node_id": skill_dict.get("node_id", "unknown"),
                        "reason": f"SENTINEL-EXEC-ERR: NameError: {e}",
                        "traceback": tb,
                        "output_state": "bound" if output else "empty",
                    }) + "\n")
                    rejected_count += 1

            del data, skills
            gc.collect()

    print(f"\n[+] DATASET COMPILATION COMPLETE")
    print(f"    Raw Accepted Flashcards     : {accepted_count}")
    print(f"    Total Rejected Skills       : {rejected_count}")
    print(f"    Dataset Fingerprint (SHA256): {hasher.hexdigest()}")
        
"""
Aletheia Dataset Factory: QLoRA JSONL Generator (Agent A V11 Expanded Blueprint)
---------------------------------------------------------------------
Transforms structural knowledge complexes into mathematically verifiable
QLoRA training flashcards. Synthesizes three core architectural pillars:

1. Strict Pydantic Invariant Layer (Schema Locks for AST/DAG Integrity)
2. Polymorphic Conversational SFT Schema (HuggingFace TRL Compliant Loss Masking)
3. FIFO Rolling Buffer (Streaming Diversity/Token Entropy & Memory Bounds)

* Hardened with Process Identifiers, Sentinel Error Codes, and Sidecar Telemetry.
"""

# Optional dependency for highly accurate token-level entropy tracking
try:
    import tiktoken
    ENCODER = tiktoken.get_encoding("cl100k_base")
except ImportError:
    ENCODER = None


# ==========================================
# SENTINEL TAXONOMY
# ==========================================
# All sentinel codes emitted by the formatter, grouped by severity.
#
# FATAL  — unrecoverable; halts execution or drops output
#   SENTINEL-FATAL-DIR          Input directory missing / inaccessible
#   SENTINEL-IO-FATAL           Primary output write failure
#   SENTINEL-DIVERSITY-CRIT     Entropy below collapse threshold; output deleted
#
# ERROR  — node / record rejected
#   SENTINEL-VAL-ERR            Pydantic schema violation
#   SENTINEL-SURVIVAL           Epistemic survival gate rejection
#   SENTINEL-DRIFT              Semantic drift firewall rejection
#   SENTINEL-ROUTE-ERR          Unrecognised payload topology
#   SENTINEL-PARSE-JSON         JSON syntax corruption
#   SENTINEL-PARSE-YAML         YAML syntax corruption
#   SENTINEL-IO-READ            Generic file read failure
#   SENTINEL-BALANCE-ERR        random.sample failure in balance_dataset
#   SENTINEL-HASH-ERR           Dataset fingerprint computation failed
#   SENTINEL-DPO-IO-ERR         DPO preference-pair file write failure
#   SENTINEL-DPO-EMPTY          No rejected traces provided for DPO, OR
#                               a per-trace preference pair is incomplete
#                               (missing 'rejected' or 'chosen'/'corrected')
#   ALL-CONFLICT                Adversarial lens major disagreement with primary
#
# WARN   — non-fatal degradation
#   ALL-SUSPICIOUS              Primary-overscoring pattern detected by adversarial lens
#   LENS-PRESSURE               Specific lens accumulating disproportionate conflicts
#   SENTINEL-WARN-TOKENIZER     tiktoken unavailable; word-level entropy fallback
#   SENTINEL-WARN-BALANCER      Empty mode or zero-item balancer edge case
#   SENTINEL-WARN-WAVEFORM      SIE node present but all waveform fields are
#                               zero. Triggers Containment Rule: the node's
#                               _sample_weight is demoted by 0.5x and the
#                               item is tagged with _waveform_demoted=True.
#   SENTINEL-MEM-CAPACITY       Rolling buffer reached max capacity
#
# INFO / PROCESS — observability breadcrumbs
#   SENTINEL-SUBSAMPLE          balance_dataset drew fewer items than available
#   SENTINEL-HASH               Dataset fingerprint computed
#   SENTINEL-ENTROPY-PRECHECK   Entropy computed before disk commit
#   SENTINEL-DPO-WRITE          DPO preference pairs written
#   ALL-VALIDATED               Both primary and adversarial evaluators agree
#   PROCESS-START / PROCESS-CHECKPOINT / INFO-BALANCER
#
# Legacy formatter codes (prefixed with LEGACY-):
#   LEGACY-SENTINEL-VAL         Skill validation failure in legacy path
#   LEGACY-SENTINEL-REJECT      Skill rejected by firewall in legacy path
# ==========================================


# ==========================================
# 1. PIPELINE ENRICHMENT KEYS
# ==========================================
# Keys injected by the DAG pipeline that are NOT on the canonical AletheiaSkill
# schema (which uses extra="forbid").  These MUST be stripped from payloads
# before Pydantic validation to avoid spurious ValidationErrors.
_PIPELINE_ENRICHMENT_KEYS = {
    "system_centrality_blast_radius", "s_sie", "s_acs", "s_topology",
    "s_validation", "c_final", "_acs_structured_constraints",
    "_acs_trajectory", "_governance_directive", "_reroll_context",
    "orchestration_mode", "_branch_degraded", "_confidence_decomposition",
    "_audit_trail", "_sie_coherence_score", "_invariant_results",
    "tikhonov_slr", "_reroll_count", "_reroll_violation_type",
    "_adversarial_verdict", "_adversarial_conflict",
}

# Keys stripped from payloads before SFT array construction.
# Hoisted to module level so both passes share one definition.
_INTERNAL_METRIC_KEYS_SET: frozenset = frozenset({
    "system_centrality_blast_radius", "s_sie", "s_acs", "s_topology",
    "s_validation", "c_final", "sie_node", "acs_handshake_sid",
    "acs_violations", "acs_audited", "_acs_structured_constraints",
    "_acs_trajectory", "_governance_directive", "_reroll_context",
    "orchestration_mode", "_branch_degraded", "_confidence_decomposition",
    "_audit_trail", "_sie_coherence_score", "_invariant_results",
    "tikhonov_slr", "_reroll_count", "_reroll_violation_type",
    "_adversarial_verdict", "_adversarial_conflict",
    "content_density", "alignment_vector", "composite_quality_score",
    "mode_scaling_factor", "system_backpressure",
    "rho", "phase_gradient", "J_info", "kappa", "field_pressure",
    "v_score", "validation_pass", "sie_score", "sie_pass", "_sie_metadata",
})


# ==========================================
# 2. FIFO ROLLING BUFFER (STREAM METRICS)
# ==========================================

class FIFORollingBuffer:
    """
    Manages a sliding window of dataset artifacts to prevent OOM errors 
    on large monorepos while calculating running entropy and mode-collapse.
    Uses Token-level distribution if tiktoken is available, else word N-grams.
    """
    def __init__(self, capacity: int = 5000, process_id: str = "UNKNOWN"):
        self.capacity = capacity
        self.buffer: deque[str] = deque(maxlen=capacity)
        self.total_processed = 0
        self.process_id = process_id
        self._capacity_warning_emitted = False

    def append(self, text_corpus: str):
        self.buffer.append(text_corpus)
        self.total_processed += 1
        
        if self.total_processed >= self.capacity and not self._capacity_warning_emitted:
            print(f"[{self.process_id}] [SENTINEL-MEM-CAPACITY] Rolling buffer reached max capacity ({self.capacity}). Oldest traces will now be evicted.")
            self._capacity_warning_emitted = True

    def calculate_rolling_entropy(self) -> float:
        """Calculates Shannon Entropy over the recent FIFO window."""
        if not self.buffer:
            return 0.0
        
        combined_text = " ".join(self.buffer)
        
        if ENCODER:
            # High-fidelity token entropy
            tokens = ENCODER.encode(combined_text)
            if not tokens:
                return 0.0
            counts = Counter(tokens)
            total = len(tokens)
        else:
            # Fallback word-level approximation
            words = combined_text.split()
            if not words:
                return 0.0
            counts = Counter(words)
            total = len(words)
        
        return -sum((count / total) * math.log2(count / total) for count in counts.values())


# ==========================================
# 3. POLYMORPHIC CONVERSATIONAL SFT SCHEMA
# ==========================================

def build_conversational_array(payload: Dict[str, Any], pydantic_skill: Optional[AletheiaSkill] = None) -> List[Dict]:
    """Polymorphic routing: Maps heterogeneous payloads into a unified QLoRA multi-turn format.
    
    Returns a list of message dicts.  Callers should set ``item["_mode"]`` after
    receiving the result — the mode is deterministic from the branch taken here.
    """
    
    # MODE A: VETERAN (Diagnostic Debugging Loop)
    # P5: Accept variant field names for traceback and diff
    _veteran_keys = ("traceback", "traceback_text", "error_trace")
    _diff_keys = ("diff", "patch", "fix_diff")
    if any(k in payload for k in _veteran_keys) and any(k in payload for k in _diff_keys):
        tb = next((payload[k] for k in _veteran_keys if k in payload), "")
        diff = next((payload[k] for k in _diff_keys if k in payload), "")
        return [
            {
                "role": "user",
                "content": (
                    f"The following code failed to execute:\n"
                    f"```python\n{payload.get('attempt_code', '')}\n```\n\n"
                    f"Traceback:\n```\n{tb}\n```\n\n"
                    f"Please diagnose the issue and provide a fix."
                )
            },
            {
                "role": "assistant",
                "content": (
                    f"### Diagnosis & Analysis\n{payload.get('analysis_text', '')}\n\n"
                    f"### Resolution Diff\n```diff\n{diff}\n```\n\n"
                    f"### Fully Resolved Code\n```python\n{payload.get('resolved_code', '')}\n```"
                )
            }
        ]
        
    # MODE B: ADVOCATE (Architectural Theory) — accept both old and new field names
    elif ("theory" in payload and "implementation" in payload) or \
         ("theory_text" in payload and "implementation_code" in payload):
        theory = payload.get("theory") or payload.get("theory_text", "")
        impl = payload.get("implementation") or payload.get("implementation_code", "")
        return [
            {
                "role": "user",
                "content": "Please provide the architectural reasoning and the implementation code for this component."
            },
            {
                "role": "assistant",
                "content": (
                    f"### Architectural Theory\n{theory}\n\n"
                    f"### Implementation Code\n```python\n{impl}\n```"
                )
            }
        ]
        
    # MODE C: CODING ASSISTANT — accept both old and new field names
    elif pydantic_skill or ("implementation" in payload) or ("code_snippet" in payload and "name" in payload):
        name = payload.get("name", "unknown")
        code = payload.get("implementation") or payload.get("code_snippet", "")
        
        if pydantic_skill:
            mm_name = pydantic_skill.teaching_layer.method_metadata.get("name", "unknown") if isinstance(pydantic_skill.teaching_layer.method_metadata, dict) else "unknown"
            if mm_name != "unknown":
                name = mm_name
            impl_code = pydantic_skill.teaching_layer.implementation_template.get("code", "") if isinstance(pydantic_skill.teaching_layer.implementation_template, dict) else ""
            if not code and impl_code:
                code = impl_code

        return [
            {
                "role": "user",
                "content": f"Implement the function `{name}`."
            },
            {
                "role": "assistant",
                "content": f"```python\n{code}\n```"
            }
        ]
        
    # MODE D: THEORIST — accept both old and new field names
    elif "constraints" in payload or "chunk_text" in payload:
        if "constraints" in payload:
            constraints = payload["constraints"]
            text = "\n".join(constraints) if isinstance(constraints, list) else str(constraints)
        else:
            text = payload.get('chunk_text', '')
        return [
            {
                "role": "user",
                "content": "Please provide the theoretical context."
            },
            {
                "role": "assistant",
                "content": text
            }
        ]

    # Failsafe
    return []


def apply_loss_masking_schema(messages: List[Dict], identity: Optional[str] = None) -> Dict:
    """
    Enforces mathematical boundaries for the prompt-loss masking function.
    Aligns with TRL / Axolotl paradigms (only learning on 'assistant' weight).
    """
    if identity:
        # P7: Do not insert a duplicate system message if one already exists
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": f"Identity Kernel: {identity}"})
        
    formatted_item: Dict[str, Any] = {"messages": []}
    for msg in messages:
        masked_msg = {
            "role": msg["role"],
            "content": msg["content"],
            # Loss calculation is STRICTLY restricted to Assistant's generative output
            "train_loss": (msg["role"] == "assistant")
        }
        formatted_item["messages"].append(masked_msg)
        
    return formatted_item


# ==========================================
# MASTER FORMATTER ENGINE
# ==========================================

def get_hash(dataset: list, process_id: str = "UNKNOWN") -> str:
    """Computes a SHA256 fingerprint of the finalized JSONL dataset."""
    try:
        fingerprint = hashlib.sha256(
            json.dumps(dataset, sort_keys=True, default=str).encode()
        ).hexdigest()
    except (TypeError, ValueError) as e:
        print(f"[{process_id}] [SENTINEL-HASH-ERR] Dataset fingerprint computation failed: {e}")
        return "FINGERPRINT_UNAVAILABLE"
    print(f"[{process_id}] [SENTINEL-HASH] Dataset fingerprint computed: {fingerprint[:16]}... ({len(dataset)} items)")
    return fingerprint


def balance_dataset(valid_items: list, process_id: str) -> list:
    """Balances dataset samples across all identified modes."""
    
    print(f"[{process_id}] [PROCESS-CHECKPOINT] Initiating dataset balancing across {len(valid_items)} valid items...")
    
    # Analyze the mode distribution using the explicit _mode tag (P3)
    modes: Counter[str] = Counter()
    for item in valid_items:
        mode = item.get("_mode", "assistant")
        modes[mode] += 1
            
    print(f"[{process_id}] [INFO-BALANCER] Pre-balance distribution: {dict(modes)}")

    # Find the smallest mode size
    if not modes:
        print(f"[{process_id}] [SENTINEL-WARN-BALANCER] No valid modes detected. Returning empty dataset.")
        return []
        
    target_count = min(modes.values())
    if target_count == 0:
         print(f"[{process_id}] [SENTINEL-WARN-BALANCER] A detected mode has 0 items. Balancing to 0. Inspect node distribution.")
         return []
         
    # We will need a way to sample equally from all modes
    import random
    
    # Group items by mode using the explicit tag
    grouped_items: Dict[str, list] = {}
    for item in valid_items:
        mode = item.get("_mode", "assistant")
        grouped_items.setdefault(mode, []).append(item)
            
    # Sample and construct balanced dataset
    balanced_dataset = []
    for mode_name, mode_items in grouped_items.items():
        if mode_items:
            try:
                sampled_items = random.sample(mode_items, target_count)
            except ValueError as e:
                print(f"[{process_id}] [SENTINEL-BALANCE-ERR] Sampling failed for mode '{mode_name}': {e}")
                sampled_items = mode_items  # degrade gracefully
            if len(sampled_items) < len(mode_items):
                print(f"[{process_id}] [SENTINEL-SUBSAMPLE] Mode '{mode_name}': sampled {target_count}/{len(mode_items)} items")
            balanced_dataset.extend(sampled_items)
            print(f"[{process_id}] [INFO-BALANCER] Sampled {target_count} items for mode '{mode_name}'.")
            
    random.shuffle(balanced_dataset)
    print(f"[{process_id}] [PROCESS-CHECKPOINT] Dataset balancing complete. Final size: {len(balanced_dataset)}")
    return balanced_dataset


def compute_mode_balance_factors(valid_items: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute bounded mode-balance multipliers from observed mode distribution.

    Underrepresented modes receive an up-weight, overrepresented modes a
    down-weight. Factors are clipped for stability and deterministic across
    runs because inputs are already traversed in deterministic order.
    """
    counts: Counter[str] = Counter(
        str(item.get("_mode", "assistant")) for item in valid_items if isinstance(item, dict)
    )
    if not counts:
        return {}

    target = len(valid_items) / float(len(counts))
    factors: Dict[str, float] = {}
    for mode_name, count in counts.items():
        if count <= 0:
            factors[mode_name] = 1.0
            continue
        raw = target / float(count)
        factors[mode_name] = round(min(1.25, max(0.75, raw)), 4)
    return factors


def apply_mode_balance_weights(valid_items: List[Dict[str, Any]], process_id: str) -> None:
    """Apply mode-balance multipliers to per-item sample weights in-place."""
    factors = compute_mode_balance_factors(valid_items)
    if not factors:
        return

    print(f"[{process_id}] [MODE-BALANCE] Applying bounded mode factors: {factors}")
    for item in valid_items:
        if not isinstance(item, dict):
            continue
        mode_name = str(item.get("_mode", "assistant"))
        factor = float(factors.get(mode_name, 1.0))
        base_weight = float(item.get("_sample_weight", 1.0))
        item["_sample_weight"] = max(0.01, round(base_weight * factor, 4))
        item["_mode_balance_factor"] = factor


# ==========================================
# 6. REJECTED TRACE ISOLATION (DPO-ALIGNED)
# ==========================================

def format_rejected_traces(rejected_traces: List[Dict[str, Any]], output_file: Path) -> int:
    """
    Writes rejected traces as DPO preference pairs to a JSONL file.

    Each trace emits a DPO triplet:
      - prompt: the node_id + reason for failure (context)
      - chosen: the corrected output after reroll (the preferred response)
      - rejected: the original failing output (the dispreferred response)

    Traces without both a rejected AND chosen/corrected output are skipped —
    DPO requires explicit preference pairs. Every skip emits a
    SENTINEL-DPO-EMPTY warning (per-trace, not just list-level) so that a
    slicer starvation can never silently produce a broken GGUF — if the
    negative-constraint stream dries up, the log stream screams first.

    Returns the count of valid DPO pairs written.
    """
    if not rejected_traces:
        print(f"[SENTINEL-DPO-EMPTY] No rejected traces provided — DPO file will be empty.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_missing_rejected = 0
    skipped_missing_chosen = 0

    try:
        with jsonlines.open(output_file, mode='w') as writer:
            for trace in rejected_traces:
                trace_id = trace.get("node_id", trace.get("trace_id", "unknown"))

                rejected = trace.get("rejected")
                if not rejected:
                    skipped_missing_rejected += 1
                    print(
                        f"[SENTINEL-DPO-EMPTY] Trace '{trace_id}' missing "
                        f"'rejected' field — skipping incomplete preference pair"
                    )
                    continue

                corrected = trace.get("corrected") or trace.get("chosen")
                if not corrected:
                    # DPO requires both chosen and rejected — skip incomplete pairs
                    skipped_missing_chosen += 1
                    print(
                        f"[SENTINEL-DPO-EMPTY] Trace '{trace_id}' missing "
                        f"'chosen'/'corrected' field — skipping incomplete "
                        f"preference pair"
                    )
                    continue

                prompt = f"Node: {trace.get('node_id', 'unknown')} | Reason: {trace.get('reason', 'unknown')}"

                # Serialize dicts to stable JSON strings
                rejected_text = json.dumps(rejected, sort_keys=True) if isinstance(rejected, dict) else str(rejected)
                chosen_text = json.dumps(corrected, sort_keys=True) if isinstance(corrected, dict) else str(corrected)

                entry = {
                    "prompt": prompt,
                    "chosen": chosen_text,
                    "rejected": rejected_text,
                }

                writer.write(entry)
                written += 1
    except IOError as e:
        print(f"[SENTINEL-DPO-IO-ERR] Failed to write DPO file {output_file}: {e}")
        return 0

    total_skipped = skipped_missing_rejected + skipped_missing_chosen
    if total_skipped:
        print(
            f"[SENTINEL-DPO-EMPTY] Summary: {total_skipped} incomplete trace(s) "
            f"skipped ({skipped_missing_rejected} missing rejected, "
            f"{skipped_missing_chosen} missing chosen) out of "
            f"{len(rejected_traces)} submitted"
        )
    print(f"[DPO] [SENTINEL-DPO-WRITE] Wrote {written} preference pairs to {output_file}")
    return written


def _load_advocate_audits_by_node(advocate_artifacts_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load Advocate JSONL artifacts and group reports by frozen node_id."""
    audits_by_node: Dict[str, List[Dict[str, Any]]] = {}
    if not advocate_artifacts_dir.exists() or not advocate_artifacts_dir.is_dir():
        return audits_by_node

    for audit_file in sorted(advocate_artifacts_dir.glob("*.jsonl")):
        try:
            with open(audit_file, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    node_id = str(rec.get("frozen_node") or "").strip()
                    if not node_id:
                        continue
                    rec["_artifact_file"] = str(audit_file)
                    audits_by_node.setdefault(node_id, []).append(rec)
        except OSError:
            continue

    for node_id, records in audits_by_node.items():
        audits_by_node[node_id] = sorted(
            records,
            key=lambda r: (
                str(r.get("branch_id", "")),
                str(r.get("frozen_node", "")),
                str(r.get("root_cause", "")),
                str(r.get("recommendation", "")),
            ),
        )
    return audits_by_node


def _trajectory_from_failure(
    node_id: str,
    rejection_reason: str,
    failure_type: str,
    advocate_report: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a Trajectory-compatible object from rejection + Advocate diagnostics."""
    root_cause = (advocate_report or {}).get("root_cause", "unknown")
    recommendation = (advocate_report or {}).get("recommendation", "manual_review")
    dominant_mode = (advocate_report or {}).get("dominant_breach_mode")
    slr_sequence = (advocate_report or {}).get("slr_breach_sequence", [])
    trajectory_id = f"traj_{hashlib.sha256(f'{node_id}:{rejection_reason}:{root_cause}'.encode('utf-8')).hexdigest()[:16]}"

    return {
        "trajectory_id": trajectory_id,
        "steps": [
            {
                "attempt_code": "",
                "error": rejection_reason,
                "diagnosis": root_cause,
                "fix": recommendation,
            }
        ],
        "labels": ["failure_matrix", failure_type],
        "metadata": {
            "node_id": node_id,
            "dominant_breach_mode": dominant_mode,
            "slr_breach_sequence": slr_sequence,
        },
        "audit_context": {
            "audit_type": (advocate_report or {}).get("audit_type", "unknown"),
            "branch_id": (advocate_report or {}).get("branch_id"),
            "frozen_node": (advocate_report or {}).get("frozen_node", node_id),
            "root_cause": root_cause,
            "dominant_breach_mode": dominant_mode,
            "recommendation": recommendation,
            "slr_breach_count": (advocate_report or {}).get("slr_breach_count"),
            "slr_breach_sequence": slr_sequence if isinstance(slr_sequence, list) else [],
            "mode_weight_skew": (advocate_report or {}).get("mode_weight_skew", {}),
            "system_backpressure": (advocate_report or {}).get("system_backpressure"),
            "adversarial_cross_check": (advocate_report or {}).get("adversarial_cross_check"),
            "artifact_path": (advocate_report or {}).get("_persisted_to") or (advocate_report or {}).get("_artifact_file"),
        },
    }


def build_failure_matrix(
    rejected_payloads: List[Dict[str, Any]],
    output_file: Path,
    advocate_artifacts_dir: Path = DEFAULT_ADVOCATE_AUDITS_DIR,
    rejected_traces: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """
    Emit deterministic FAILURE_MATRIX JSONL by joining rejected payloads with Advocate audits.

    Join key priority:
      1) rejected payload `node_id`
      2) optional rejected trace `node_id` (if provided)
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    audits_by_node = _load_advocate_audits_by_node(advocate_artifacts_dir)

    traces_by_node: Dict[str, List[Dict[str, Any]]] = {}
    for trace_item in (rejected_traces or []):
        node_id = str(trace_item.get("node_id") or "").strip()
        if not node_id:
            continue
        traces_by_node.setdefault(node_id, []).append(trace_item)

    rows: List[Dict[str, Any]] = []
    for payload in rejected_payloads:
        node_id = str(payload.get("node_id") or "UNKNOWN_NODE")
        rejection_reason = str(
            payload.get("_rejection_reason") or payload.get("reason") or "unknown_rejection"
        )
        failure_type = str(payload.get("_failure_type") or "unknown")
        raw_sie_metadata = payload.get("_sie_metadata")
        normalized_sie_metadata: Dict[str, Any] = {}
        if isinstance(raw_sie_metadata, dict):
            normalized_sie_metadata = dict(raw_sie_metadata)
        acs_violations = payload.get("_acs_violations")
        if not isinstance(acs_violations, list):
            acs_violations = []

        advocate_report = None
        if node_id in audits_by_node and audits_by_node[node_id]:
            advocate_report = audits_by_node[node_id][-1]

        matched_trace: Optional[Dict[str, Any]] = None
        if node_id in traces_by_node and traces_by_node[node_id]:
            matched_trace = traces_by_node[node_id][-1]

        failure_key = f"{node_id}|{rejection_reason}|{(advocate_report or {}).get('root_cause', 'none')}"
        failure_id = hashlib.sha256(failure_key.encode("utf-8")).hexdigest()[:16]
        trajectory = _trajectory_from_failure(node_id, rejection_reason, failure_type, advocate_report)
        trajectory["failure_matrix_reference"] = failure_id

        row = {
            "failure_id": failure_id,
            "node_id": node_id,
            "failure_type": failure_type,
            "rejection_reason": rejection_reason,
            "constraint_violations": acs_violations,
            "final_status": payload.get("_final_status"),
            "source_file": payload.get("_source_file"),
            "sie_snapshot": {
                "content_density": float(normalized_sie_metadata.get("content_density", 0.0) or 0.0),
                "s_sie": float(normalized_sie_metadata.get("s_sie", 0.0) or 0.0),
                "composite_quality_score": float(normalized_sie_metadata.get("composite_quality_score", 0.0) or 0.0),
            },
            "advocate_audit": {
                "root_cause": (advocate_report or {}).get("root_cause", "unknown"),
                "dominant_breach_mode": (advocate_report or {}).get("dominant_breach_mode"),
                "recommendation": (advocate_report or {}).get("recommendation", "manual_review"),
                "slr_breach_count": (advocate_report or {}).get("slr_breach_count"),
                "slr_breach_sequence": (advocate_report or {}).get("slr_breach_sequence", []),
                "mode_weight_skew": (advocate_report or {}).get("mode_weight_skew", {}),
                "adversarial_cross_check": (advocate_report or {}).get("adversarial_cross_check"),
                "artifact_path": (advocate_report or {}).get("_persisted_to") or (advocate_report or {}).get("_artifact_file"),
            },
            "trajectory": trajectory,
            "remediation_status": "rerolled" if matched_trace and matched_trace.get("corrected") else "abandoned",
        }

        if matched_trace:
            row["trace_prompt"] = f"Node: {node_id} | Reason: {rejection_reason}"
        rows.append(row)

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            str(r.get("node_id", "")),
            str(r.get("failure_type", "")),
            str(r.get("rejection_reason", "")),
            str(r.get("failure_id", "")),
        ),
    )

    with open(output_file, "w", encoding="utf-8") as out_f:
        for row in rows_sorted:
            out_f.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    print(
        f"[FAILURE-MATRIX] Wrote {len(rows_sorted)} entries to {output_file} "
        f"(audits_dir={advocate_artifacts_dir})"
    )
    return len(rows_sorted)


# ---------------------------------------------------------------------------
# Streaming helpers — used by the two-pass format_dataset implementation
# ---------------------------------------------------------------------------

def _stream_payloads_from_dir(input_dir: Path):
    """Generator: yield (payload_dict, file_path) one node at a time.

    Handles .jsonl (one object per line), .json, and .yaml/.yml without
    ever holding more than one file's worth of parsed data simultaneously.
    """
    for file_path in sorted(input_dir.glob("*.*")):
        suffix = file_path.suffix.lower()

        # .rejected.yaml files are multi-GB OOM traps; they hold discarded nodes,
        # not training payloads. Skip them unconditionally.
        if suffix in (".yaml", ".yml") and file_path.stem.endswith(".rejected"):
            continue

        if suffix == ".jsonl":
            try:
                with open(file_path, "r", encoding="utf-8") as fh:
                    for raw_line in fh:
                        stripped = raw_line.strip()
                        if not stripped:
                            continue
                        try:
                            obj = json.loads(stripped)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(obj, dict):
                            yield obj, file_path
            except OSError as exc:
                print(f"[SENTINEL-IO-READ] Failed to open {file_path.name}: {exc}")
            continue

        if suffix not in (".json", ".yaml", ".yml"):
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) if suffix in (".yaml", ".yml") else json.load(fh)
        except json.JSONDecodeError as exc:
            print(f"[SENTINEL-PARSE-JSON] JSON syntax corruption in {file_path.name}: {exc}")
            continue
        except yaml.YAMLError as exc:
            print(f"[SENTINEL-PARSE-YAML] YAML syntax corruption in {file_path.name}: {exc}")
            continue
        except OSError as exc:
            print(f"[SENTINEL-IO-READ] Failed to open {file_path.name}: {exc}")
            continue

        if data is None:
            continue

        if isinstance(data, list):
            for item in data:
                if isinstance(item, list):
                    for n in item:
                        if isinstance(n, dict):
                            yield n, file_path
                elif isinstance(item, dict):
                    yield item, file_path
        elif isinstance(data, dict):
            skills = data.get("capability_injection", {}).get("compiled_skills", [])
            if skills:
                for s in skills:
                    if isinstance(s, dict):
                        yield s, file_path
            elif "validated_nodes" in data or "rejected_nodes" in data:
                for key in ("validated_nodes", "rejected_nodes"):
                    for n in (data.get(key) or []):
                        if isinstance(n, dict):
                            yield n, file_path
            else:
                yield data, file_path


def _detect_payload_mode_fast(sanitized: Dict[str, Any]) -> str:
    """Detect mode without constructing the full messages array."""
    _vet = ("traceback", "traceback_text", "error_trace")
    _dif = ("diff", "patch", "fix_diff")
    if any(k in sanitized for k in _vet) and any(k in sanitized for k in _dif):
        return "veteran"
    if ("theory" in sanitized and "implementation" in sanitized) or \
       ("theory_text" in sanitized and "implementation_code" in sanitized):
        return "advocate"
    if "constraints" in sanitized or "chunk_text" in sanitized:
        return "theorist"
    return "assistant"


def _would_produce_messages(sanitized: Dict[str, Any], pydantic_skill) -> bool:
    """Lightweight check that mirrors build_conversational_array's routing logic."""
    _vet = ("traceback", "traceback_text", "error_trace")
    _dif = ("diff", "patch", "fix_diff")
    if any(k in sanitized for k in _vet) and any(k in sanitized for k in _dif):
        return True
    if ("theory" in sanitized and "implementation" in sanitized) or \
       ("theory_text" in sanitized and "implementation_code" in sanitized):
        return True
    if pydantic_skill or "implementation" in sanitized or \
       ("code_snippet" in sanitized and "name" in sanitized):
        return True
    if "constraints" in sanitized or "chunk_text" in sanitized:
        return True
    return False


def _make_compact_rejection(
    payload: Dict[str, Any],
    source_file: str,
    reason: str,
    failure_type: str,
    sie_raw: Any = None,
    acs_violations: Optional[List] = None,
    final_status: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the minimal rejection record needed by build_failure_matrix.

    Stores only the fields that build_failure_matrix reads — avoiding the
    full payload copy that was the original memory hog.
    """
    compact: Dict[str, Any] = {
        "node_id": payload.get("node_id", "UNKNOWN_NODE"),
        "_rejection_reason": reason,
        "_failure_type": failure_type,
        "_source_file": source_file,
        "_final_status": final_status,
        "_acs_violations": acs_violations or [],
    }
    if isinstance(sie_raw, dict):
        compact["_sie_metadata"] = {
            "content_density": sie_raw.get("content_density", 0.0),
            "s_sie": sie_raw.get("s_sie", 0.0),
            "composite_quality_score": sie_raw.get("composite_quality_score", 0.0),
        }
    return compact


def format_dataset(input_dir: Path, output_file: Path, identity: Optional[str] = None, apply_loss_masking: bool = False, mode: Optional[str] = None):
    """Two-pass streaming formatter — O(1) payload memory for 100k+ node corpora.

    Pass 1 streams all input to count the mode distribution and write
    rejections immediately to the sidecar file. No payload is retained.

    Pass 2 streams again, formats each accepted node into ChatML, and writes
    it directly to the output JSONL using systematic subsampling to produce
    an exactly balanced dataset. The accepted items are never accumulated in
    a list.
    """
    
    import random as _random

    process_id = f"PROC-{uuid.uuid4().hex[:8].upper()}"
    active_mode_label = (mode or "auto").upper()
    print(f"\n{'='*60}")
    print(f"  ALETHEIA DATASET FORMATTER — ACTIVE MODE: {active_mode_label}")
    print(f"{'='*60}")
    print(f"[{process_id}] [PROCESS-START] Initializing Aletheia Formatter V12 (two-pass streaming)")
    print(f"[{process_id}] [INFO] Target Directory: {input_dir}")
    print(f"[{process_id}] [INFO] Loss Masking Enforcement: {'ACTIVE' if apply_loss_masking else 'INACTIVE'}")

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[{process_id}] [SENTINEL-FATAL-DIR] Input directory not found or inaccessible: {input_dir}")
        sys.exit(1)

    if not ENCODER:
        print(f"[{process_id}] [SENTINEL-WARN-TOKENIZER] 'tiktoken' not found. Defaulting to word-level entropy approximations.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    rejections_file = output_file.parent / f"{output_file.stem}_rejections.jsonl"
    failure_matrix_file = output_file.parent / f"{output_file.stem}_failure_matrix.jsonl"

    # =========================================================================
    # PASS 1 — mode distribution scan
    # Streams every payload through validation. Rejects are written to the
    # sidecar file immediately and stored as compact metadata (~300 B each).
    # No accepted payload is retained in memory.
    # =========================================================================
    mode_counts: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    dropped_artifacts: List[Dict[str, Any]] = []  # compact metadata only — not full payloads
    n_pass1_valid = 0
    n_pass1_rejected = 0

    print(f"[{process_id}] [PROCESS-CHECKPOINT] Pass 1: counting mode distribution...")

    with open(rejections_file, "w", encoding="utf-8") as f_rej:
        for payload, file_path in _stream_payloads_from_dir(input_dir):
            node_id = payload.get("node_id", "UNKNOWN_NODE")
            pydantic_skill = None

            if "node_id" in payload and "epistemic" in payload:
                try:
                    vp = {k: v for k, v in payload.items() if k not in _PIPELINE_ENRICHMENT_KEYS}
                    pydantic_skill = AletheiaSkill.model_validate(vp)
                except PydanticValidationError as exc:
                    error_type = exc.errors()[0]["type"]
                    rejection_reasons[error_type] += 1
                    n_pass1_rejected += 1
                    print(f"[{process_id}] [SENTINEL-VAL-ERR] Schema violation in '{node_id}': {error_type}")
                    compact = _make_compact_rejection(payload, file_path.name,
                                                      f"Pydantic Validation Failed: {error_type}", "pydantic_error")
                    dropped_artifacts.append(compact)
                    f_rej.write(json.dumps(compact, default=str) + "\n")
                    continue

                ep_obj = pydantic_skill.epistemic
                ep_state = getattr(ep_obj, "state", None) if ep_obj else None
                if not ep_state or ep_state not in ("ACCEPTED", "terminal"):
                    rejection_reasons["epistemic_survival_gate"] += 1
                    n_pass1_rejected += 1
                    print(f"[{process_id}] [SENTINEL-SURVIVAL] Node '{node_id}' state='{ep_state}' — epistemic gate")
                    compact = _make_compact_rejection(
                        payload, file_path.name,
                        f"Epistemic Survival Gate: state={ep_state}", "epistemic_gate",
                        sie_raw=payload.get("sie_node"),
                        acs_violations=payload.get("acs_violations", []),
                        final_status=getattr(ep_obj, "final_status", None) if ep_obj else None,
                    )
                    dropped_artifacts.append(compact)
                    f_rej.write(json.dumps(compact, default=str) + "\n")
                    continue

            sanitized = {k: v for k, v in payload.items() if k not in _INTERNAL_METRIC_KEYS_SET}

            if not _would_produce_messages(sanitized, pydantic_skill):
                rejection_reasons["unrecognized_topology"] += 1
                n_pass1_rejected += 1
                print(f"[{process_id}] [SENTINEL-ROUTE-ERR] Unrecognized topology in '{node_id}'")
                compact = _make_compact_rejection(payload, file_path.name,
                                                  "Unrecognized Payload Format for SFT Array", "route_error")
                dropped_artifacts.append(compact)
                f_rej.write(json.dumps(compact, default=str) + "\n")
                continue

            mode = _detect_payload_mode_fast(sanitized)
            mode_counts[mode] += 1
            n_pass1_valid += 1

    # Balancing parameters — computed once from Pass 1 counts
    if not mode_counts:
        print(f"[{process_id}] [SENTINEL-WARN-BALANCER] No valid nodes found after Pass 1. Aborting.")
        return

    target_count = min(mode_counts.values())
    total_valid = sum(mode_counts.values())
    n_modes = len(mode_counts)
    target_per_mode = total_valid / n_modes if n_modes > 0 else 0.0

    mode_balance_factors: Dict[str, float] = {}
    for m, cnt in mode_counts.items():
        raw = (target_per_mode / float(cnt)) if cnt > 0 else 1.0
        mode_balance_factors[m] = round(min(1.25, max(0.75, raw)), 4)

    print(f"[{process_id}] [INFO-BALANCER] Pass 1 distribution: {dict(mode_counts)}")
    print(f"[{process_id}] [INFO-BALANCER] Target count per mode: {target_count}")
    print(f"[{process_id}] [INFO-BALANCER] Mode balance factors: {mode_balance_factors}")

    # =========================================================================
    # PASS 2 — format + write with systematic subsampling
    #
    # Systematic subsampling: for mode M with total_M items and target_count,
    # accept item i (1-indexed) when:
    #   floor(i * target_count / total_M) > floor((i-1) * target_count / total_M)
    # This gives exactly target_count evenly-spaced samples without storing
    # any items in memory.
    # =========================================================================
    fifo_buffer = FIFORollingBuffer(capacity=5000, process_id=process_id)
    mode_seen: Counter[str] = Counter()
    mode_written: Counter[str] = Counter()
    n_written = 0
    hasher = hashlib.sha256()

    print(f"[{process_id}] [PROCESS-CHECKPOINT] Pass 2: formatting and writing balanced dataset...")

    with open(output_file, "w", encoding="utf-8") as f_out:
        for payload, file_path in _stream_payloads_from_dir(input_dir):
            node_id = payload.get("node_id", "UNKNOWN_NODE")
            pydantic_skill = None

            # Re-validate (same gates as Pass 1, silent — rejections already logged)
            if "node_id" in payload and "epistemic" in payload:
                try:
                    vp = {k: v for k, v in payload.items() if k not in _PIPELINE_ENRICHMENT_KEYS}
                    pydantic_skill = AletheiaSkill.model_validate(vp)
                except PydanticValidationError:
                    continue

                ep_obj = pydantic_skill.epistemic
                ep_state = getattr(ep_obj, "state", None) if ep_obj else None
                if not ep_state or ep_state not in ("ACCEPTED", "terminal"):
                    continue

            sanitized = {k: v for k, v in payload.items() if k not in _INTERNAL_METRIC_KEYS_SET}

            if not _would_produce_messages(sanitized, pydantic_skill):
                continue

            mode = _detect_payload_mode_fast(sanitized)
            total_M = mode_counts.get(mode, 0)
            if total_M == 0 or target_count == 0:
                continue

            mode_seen[mode] += 1
            i = mode_seen[mode]

            # Systematic subsampling gate
            if (i * target_count // total_M) <= ((i - 1) * target_count // total_M):
                continue

            # Build the full ChatML messages array
            messages = build_conversational_array(sanitized, pydantic_skill=pydantic_skill)
            if not messages:
                continue

            # Semantic drift firewall — check the fully formatted item
            if apply_loss_masking:
                item_to_write = apply_loss_masking_schema(list(messages), identity)
            else:
                msgs = list(messages)
                if identity and (not msgs or msgs[0].get("role") != "system"):
                    msgs.insert(0, {"role": "system", "content": f"Identity Kernel: {identity}"})
                item_to_write = {"messages": msgs}

            try:
                enforce_semantic_firewall(
                    json.dumps(item_to_write, sort_keys=True, default=str),
                    context=f"export:{node_id}",
                )
            except DriftViolation as dv:
                print(f"[{process_id}] [SENTINEL-DRIFT] Node '{node_id}' rejected: forbidden token '{dv.token}'")
                mode_seen[mode] -= 1  # undo the counter so systematic sampling stays aligned
                continue

            # Provenance metadata
            item_to_write["_aletheia_metadata"] = {
                "source_file": file_path.name,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "node_id": pydantic_skill.node_id if pydantic_skill else f"legacy_{uuid.uuid4().hex[:8]}",
            }

            # SIE training metadata (small subset — not stripped)
            sie_training_meta: Dict[str, Any] = {}
            if pydantic_skill:
                ep = pydantic_skill.epistemic
                sie_training_meta["final_confidence"] = round(getattr(ep, "c_node", 0.0), 4) if ep else 0.0
                sie_training_meta["mode"] = payload.get("orchestration_mode", "unknown")
                sie_training_meta["violations"] = payload.get("acs_violations", [])
            _adv_verdict = payload.get("_adversarial_verdict", {})
            if _adv_verdict:
                sie_training_meta["adversarial_conflict"] = payload.get("_adversarial_conflict", False)
                sie_training_meta["conflict_magnitude"] = _adv_verdict.get("conflict", 0.0)
            if sie_training_meta:
                item_to_write["_sie_training_metadata"] = sie_training_meta

            # Sample weight: c_final × complexity × mode_balance_factor
            c_final_val = sie_training_meta.get("final_confidence", 0.5)
            complexity_factor = min(1.0, len(messages) / 4.0)
            mode_factor = float(mode_balance_factors.get(mode, 1.0))
            item_to_write["_sample_weight"] = max(0.01, round(c_final_val * complexity_factor * mode_factor, 4))
            item_to_write["_mode_balance_factor"] = mode_factor
            item_to_write["_mode"] = mode

            # Write directly to disk — never accumulate
            line = json.dumps(item_to_write, default=str) + "\n"
            f_out.write(line)
            hasher.update(line.encode())

            assistant_content = " ".join(m["content"] for m in messages if m["role"] == "assistant")
            fifo_buffer.append(assistant_content)
            mode_written[mode] += 1
            n_written += 1

            if n_written % 5000 == 0:
                f_out.flush()
                print(f"[{process_id}] [PROCESS-CHECKPOINT] {n_written} items written so far...")

    # Entropy gate — post-write (delete output if collapse detected)
    rolling_entropy = fifo_buffer.calculate_rolling_entropy()
    print(f"[{process_id}] [SENTINEL-ENTROPY-PRECHECK] Post-write entropy: {rolling_entropy:.2f} "
          f"(threshold: 3.5, items: {n_written})")

    if rolling_entropy < 3.5 and n_written > 10:
        print(f"[{process_id}] [SENTINEL-DIVERSITY-CRIT] HALTING: entropy={rolling_entropy:.2f} < 3.5. "
              f"Deleting output file.")
        try:
            output_file.unlink(missing_ok=True)
        except Exception:
            pass
        return

    # Failure matrix (uses compact dropped_artifacts — no full payloads needed)
    try:
        build_failure_matrix(
            rejected_payloads=dropped_artifacts,
            output_file=failure_matrix_file,
            advocate_artifacts_dir=DEFAULT_ADVOCATE_AUDITS_DIR,
            rejected_traces=None,
        )
    except Exception as exc:
        print(f"[{process_id}] [SENTINEL-FAILURE-MATRIX-ERR] Failed to build failure matrix: {exc}")

    dataset_fingerprint = hasher.hexdigest()
    print(f"\n[{process_id}] [+] DATASET COMPILATION COMPLETE")
    print(f"    Pass 1 Valid Nodes          : {n_pass1_valid}")
    print(f"    Balanced Final Dataset      : {n_written}")
    print(f"    Dataset Fingerprint (SHA256): {dataset_fingerprint}")
    print(f"    Invariant Violations Dropped: {n_pass1_rejected}")

    if rejection_reasons:
        print(f"\n[{process_id}] [INFO] --- Rejection Breakdown ---")
        for reason, count in rejection_reasons.most_common():
            print(f"        - {reason}: {count}")

    print(f"\n[{process_id}] [METRIC] Rolling Diversity Score  : {rolling_entropy:.2f} (Collapse Threshold: 3.5)")
    print(f"[{process_id}] [INFO-BALANCER] Final mode distribution: {dict(mode_written)}")

    _MODE_METRIC_LABELS = {
        "theorist":         "Causal Axioms Extracted",
        "veteran":          "Trajectories/Recoveries Parsed",
        "advocate":         "Alignment Constraints Enforced",
        "coding_assistant": "API Patterns Compiled",
    }
    metric_label = _MODE_METRIC_LABELS.get(mode or "", "Knowledge Units Compiled")
    print(f"[{process_id}] [METRIC] {metric_label:<35}: {n_written}")


# ==========================================
# 7. FLASHCARD GENERATION API
# ==========================================

# Sentinel constants — skills whose intent/heuristics match these generic
# placeholders are rejected as low-signal template artifacts.
_GENERIC_INTENT_PROBLEM = "Generic problem statement"
_GENERIC_INTENT_GOAL = "Generic goal statement"
_GENERIC_USE_WHEN = "General purpose use"

# Placeholder patterns that indicate unresolved template code
_TEMPLATE_PLACEHOLDER_RE = r"\{[a-z_]+\}"


def _stable_json_blob(obj: Any) -> str:
    """Deterministic JSON serialisation — sort keys, no trailing whitespace."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def _strip_markdown(text: str) -> str:
    """Remove markdown fencing, headings, and emphasis from output text."""
    import re
    text = re.sub(r"```[a-z]*\n?", "", text)
    text = re.sub(r"```", "", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = text.replace("**", "")
    return text


def _extract_skills_from_yaml(data: dict) -> List[dict]:
    """Walk nested wrappers to find capability_injection.compiled_skills."""
    if "capability_injection" in data:
        return data["capability_injection"].get("compiled_skills", [])
    # Support wrapped schemas (e.g. YAML_Knowledge_Pipelines.Compiled_Intelligence.…)
    for _key, val in data.items():
        if isinstance(val, dict):
            result = _extract_skills_from_yaml(val)
            if result:
                return result
    return []


def _has_populated_intent(teaching: dict) -> bool:
    """Return True when the intent block contains non-empty problem AND goal."""
    intent = teaching.get("intent", {})
    problem = intent.get("problem", "")
    goal = intent.get("goal", "")
    return bool(problem) and bool(goal)


def _build_flashcard(skill: dict) -> dict:
    """Convert a single compiled skill dict into an {instruction, input, output} card."""
    teaching = skill.get("teaching_layer", {})
    identity = teaching.get("skill_identity", {})
    category = identity.get("category", "unknown")
    canonical_id = identity.get("canonical_id", "UNKNOWN")

    semantics = skill.get("semantics", {})
    deps = skill.get("dependencies", {})
    state_exec = skill.get("state_and_execution", {})
    heuristics_raw = teaching.get("heuristics", {})
    epistemic = skill.get("epistemic", {})
    conf_raw = epistemic.get("confidence")
    confidence_val = 0.0
    if isinstance(conf_raw, dict):
        confidence_val = float(conf_raw.get("global", 0.0) or 0.0)
    elif isinstance(conf_raw, (int, float)):
        confidence_val = float(conf_raw)

    operator_types = semantics.get("operator_types", [])
    pipeline_sig = " -> ".join(operator_types)

    # --- instruction ---
    if _has_populated_intent(teaching):
        instruction = (
            f"Synthesize an optimized execution method for "
            f"Category: {category} | "
            f"Canonical ID: {canonical_id} | "
            f"Pipeline Signature: {pipeline_sig} | "
            f"System Prior Confidence: {confidence_val:.3f}"
        )
    else:
        instruction = (
            f"Extract and formalize the structural execution pattern for "
            f"Category: {category} | "
            f"Canonical ID: {canonical_id} | "
            f"Pipeline Signature: {pipeline_sig} | "
            f"System Prior Confidence: {confidence_val:.3f}"
        )

    # --- input payload (constraints + graph) ---
    mutation_tracking = sorted(state_exec.get("mutation_tracking", []))
    upstream = sorted(deps.get("upstream_callers", []))
    downstream = deps.get("downstream_calls", [])

    input_payload: Dict[str, Any] = {
        "constraints": {
            "operator_types": operator_types,
            "mutation_tracking": mutation_tracking,
            "heuristic_conditions": heuristics_raw.get("conditions", {}),
        },
        "graph": {
            "upstream_callers": upstream,
            "downstream_calls": downstream,
        },
    }

    # --- output payload (heuristics + transformation + template + optional intent) ---
    output_heuristics: Dict[str, Any] = {}
    if heuristics_raw.get("use_when"):
        output_heuristics["use_when"] = heuristics_raw["use_when"]
    if heuristics_raw.get("avoid_when"):
        output_heuristics["avoid_when"] = heuristics_raw["avoid_when"]
    # Machine conditions stay in input, NOT in output heuristics

    # Resolve transformation — prefer teaching_layer.transformation, fallback to transformation_delta
    transformation = teaching.get("transformation") or teaching.get("transformation_delta")

    # Resolve template — prefer teaching_layer.template, fallback to implementation_template
    template_raw = teaching.get("template") or teaching.get("implementation_template", {})
    template_out: Dict[str, Any] = {}
    if isinstance(template_raw, dict):
        if "code" in template_raw:
            template_out["code"] = template_raw["code"]
        if "inputs" in template_raw:
            template_out["inputs"] = template_raw["inputs"]

    output_payload: Dict[str, Any] = {}
    if _has_populated_intent(teaching):
        output_payload["intent"] = teaching["intent"]
    output_payload["heuristics"] = output_heuristics
    if transformation:
        output_payload["transformation"] = transformation
    if template_out:
        output_payload["template"] = template_out

    return {
        "instruction": instruction,
        "input": _stable_json_blob(input_payload),
        "output": _strip_markdown(_stable_json_blob(output_payload)),
    }


def generate_flashcards(yaml_path: Path) -> List[dict]:
    """Load a YAML knowledge matrix and produce flashcard dicts."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    skills = _extract_skills_from_yaml(data) if isinstance(data, dict) else []
    # Sort by node_id for determinism; structural-only cards first
    skills_sorted = sorted(
        skills,
        key=lambda s: (
            _has_populated_intent(s.get("teaching_layer", {})),
            s.get("node_id", ""),
        ),
    )
    return [_build_flashcard(s) for s in skills_sorted]


# ==========================================
# 8. STRICT DATASET FILTER & WRITER
# ==========================================

import re as _re


class ValidationError(Exception):
    """Raised by StrictDatasetFilter when a flashcard fails validation."""

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


class StrictDatasetFilter:
    """Validates flashcard entries against structural quality rules."""

    MAX_GRAPH_EDGES = 50

    def __init__(self):
        self.stats: Dict[str, int] = {"accepted": 0, "rejected": 0}
        self._last_rejection_reason: Optional[str] = None

    def validate_entry(self, card: dict) -> None:
        """Raise ValidationError if the card violates any quality rule."""
        # 1. Input JSON must parse
        try:
            input_obj = json.loads(card["input"])
        except (json.JSONDecodeError, KeyError):
            raise ValidationError("malformed_input_json", "input field is not valid JSON")

        # 2. Output JSON must parse
        try:
            json.loads(card["output"])
        except (json.JSONDecodeError, KeyError):
            raise ValidationError("malformed_output_json", "output field is not valid JSON")

        # 3. Graph complexity ceiling
        graph = input_obj.get("graph", {})
        total_edges = len(graph.get("downstream_calls", [])) + len(graph.get("upstream_callers", []))
        if total_edges > self.MAX_GRAPH_EDGES:
            raise ValidationError(
                "graph_complexity_exceeded",
                f"total edges {total_edges} > {self.MAX_GRAPH_EDGES}",
            )

        # 4. Zero-density AST
        ops = input_obj.get("constraints", {}).get("operator_types", [])
        if not ops:
            raise ValidationError("zero_density_ast", "operator_types is empty")

    def mark_accepted(self) -> None:
        self.stats["accepted"] += 1

    def mark_rejected(self, reason: str) -> None:
        self.stats["rejected"] += 1
        self._last_rejection_reason = reason


def _detect_template_placeholders(card: dict) -> Optional[str]:
    """Return a rejection reason if the card contains unresolved template placeholders."""
    output_text = card.get("output", "")
    if _re.search(_TEMPLATE_PLACEHOLDER_RE, output_text):
        return "template_code_placeholder"
    return None


def write_flashcards(
    cards: List[dict], accepted_path: Path, rejected_path: Path
) -> StrictDatasetFilter:
    """Validate cards, split into accepted/rejected JSONL, return the filter with stats."""
    filt = StrictDatasetFilter()
    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)

    with open(accepted_path, "w", encoding="utf-8") as acc_f, \
         open(rejected_path, "w", encoding="utf-8") as rej_f:
        for card in cards:
            rejection_reason: Optional[str] = None
            try:
                filt.validate_entry(card)
            except ValidationError as exc:
                rejection_reason = exc.reason

            if rejection_reason is None:
                rejection_reason = _detect_template_placeholders(card)

            if rejection_reason is None:
                filt.mark_accepted()
                acc_f.write(_stable_json_blob(card) + "\n")
            else:
                filt.mark_rejected(rejection_reason)
                rejected_record = dict(card)
                rejected_record["rejection_reason"] = rejection_reason
                rej_f.write(_stable_json_blob(rejected_record) + "\n")

    return filt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aletheia V11 Polymorphic SFT Formatter")
    parser.add_argument("input_dir", help="Path to knowledge_complexes directory")
    parser.add_argument("-o", "--output", required=True, help="Path to output .jsonl dataset")
    parser.add_argument("--apply-loss-masking", action="store_true", help="Apply strict <|mask_loss|> schema mapped to -100 labels")
    parser.add_argument("--split", type=float, default=1.0, help="Train/test split ratio (reserved)")
    parser.add_argument("--identity", type=str, default=None, help="System Identity Kernel string")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["coding_assistant", "theorist", "veteran", "advocate"],
                        help="Active pipeline mode — drives telemetry labels and output naming")

    args = parser.parse_args()

    format_dataset(
        input_dir=Path(args.input_dir),
        output_file=Path(args.output),
        identity=args.identity,
        apply_loss_masking=args.apply_loss_masking,
        mode=args.mode,
    )