import argparse
import hashlib
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Accumulates SIE lens tags across a single format_complexes() run.
# Cleared at the entry of each run to prevent cross-run state bleed.
_SIE_LENS_COUNTS: dict = {}


def is_high_quality_concept(name: str, content: str) -> bool:
    """Concept Qualification Module (CQM) — rejects structural and semantic debris.

    Structural filters (OCR noise, numeric garbage) are applied first.
    Semantic filters (dangling fragments, misaligned pairs) follow.
    """
    name_s = str(name).strip()
    content_s = str(content).strip()
    name_lower = name_s.lower()
    content_lower = content_s.lower()

    if not name_lower or not content_lower:
        return False

    # --- Structural gates ---
    # Pure numeric / decimal / symbolic garbage
    if re.fullmatch(r"[0-9\.\-\s\,\+\[\]\(\)°%]+", name_lower):
        return False
    # No real alphabetic word in the name
    if not re.search(r"[a-zA-Z]{3,}", name_s):
        return False
    # Structural markers
    if re.search(r"\b(table|fig|figure|page|edu)\b", name_lower):
        return False

    # --- Semantic gates ---
    # Reject dangling linguistic fragments (articles, prepositions, comparatives, indefinite pronouns)
    if re.match(
        r"^\s*(this|that|these|those|than|and|or|of|from|to|a|an|the|"
        r"such|similar|with|without|in|on|at|by|for|as|"
        r"some|many|all|other|their|its|any|most|another)\b",
        name_lower,
    ):
        return False
    # Meaningless academic debris labels
    if re.search(r"\b(descriptive|example|term|see also|listed in)\b", name_lower):
        return False

    # Content must have enough substance (15+ words)
    if len(content_lower.split()) < 15:
        return False

    # Concept alignment: at least one 4+ letter word from the name must appear in content
    name_words = re.findall(r"\b[a-z]{4,}\b", name_lower)
    if name_words and not any(w in content_lower for w in name_words):
        return False

    return True


def get_dynamic_prompt(name: str) -> str:
    """Generate a context-aware prompt instead of a single generic template."""
    name_lower = name.lower()
    if any(w in name_lower for w in ["process", "synthesis", "reaction", "mechanism", "pathway", "cycle"]):
        return f"Explain the chemical mechanism and pathway of {name}."
    if any(w in name_lower for w in ["law", "theory", "effect", "principle", "model", "theorem"]):
        return f"Describe the theoretical framework and implications of {name}."
    if any(w in name_lower for w in ["acid", "base", "metal", "polymer", "catalyst", "oxide", "isotope"]):
        return f"Describe the chemical properties and physical characteristics of {name}."
    return f"Analyze the following chemical concept and explain its properties:\n\n{name}"


def apply_sie_filters(text: str) -> dict:
    """Semantic Identity Embedding filter stack (SSDO-lite + BASG-lite + ICLM + MRA).

    Returns {"pass": True, "text": ..., "lens": ...} or {"pass": False, "reason": ...}.
    MRA masking is deterministic: driven by hashlib.md5 of the text, not random().
    """
    if not text or len(text.split()) < 20:
        return {"pass": False, "reason": "short_output"}

    text_lower = text.lower()

    # BASG-lite: reject sycophantic / validation-first phrasing
    _SYCOPHANCY = ["as an ai", "you are absolutely right", "i completely agree", "you're correct"]
    if any(p in text_lower for p in _SYCOPHANCY):
        return {"pass": False, "reason": "sycophancy_detected"}

    # SSDO-lite: lexical density guard (only applied when text is long enough to avoid
    # falsely penalising concise chemistry reasoning that legitimately repeats terms)
    words = re.findall(r"\w+", text_lower)
    if len(words) > 50 and (len(set(words)) / len(words)) < 0.4:
        return {"pass": False, "reason": "low_lexical_density"}

    # SSDO-lite: require at least one causal AND one transformation operator
    _CAUSAL    = ["because", "therefore", "results in", "consequently", "due to"]
    _TRANSFORM = [
        "leads to", "lead to", "causes", "changes", "shifts", "transforms", "becomes",
        "allow", "enhances", "enable", "increases", "facilitates", "forms",
        "reacts", "influences", "reduces", "promotes", "converts", "prevents",
    ]

    def _has_causal_and_transform(t: str) -> bool:
        tl = t.lower()
        return any(op in tl for op in _CAUSAL) and any(op in tl for op in _TRANSFORM)

    if not _has_causal_and_transform(text):
        return {"pass": False, "reason": "missing_causal_structure"}

    # ICLM: assign diagnostic lens tag
    lens = "theoretical"
    if any(k in text_lower for k in ["why", "failure", "breakdown", "error"]):
        lens = "diagnostic"
    elif any(k in text_lower for k in ["compare", "difference", "ratio"]):
        lens = "comparative"
    elif any(k in text_lower for k in ["mechanism", "process", "step"]):
        lens = "mechanistic"

    # MRA: Masked Reasoning Augmentation — deterministic via hashlib (not random())
    final_text = text
    if int(hashlib.md5(text.encode()).hexdigest(), 16) % 100 < 15:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) >= 4:
            mid_idx         = len(sentences) // 2
            masked_sentences = sentences[:mid_idx] + sentences[mid_idx + 1:]
            masked_text      = " ".join(masked_sentences)
            # Safety gate: revert if post-mask text loses causal structure or becomes too short
            if len(masked_sentences) >= 2 and _has_causal_and_transform(masked_text):
                final_text = masked_text

    return {"pass": True, "text": final_text, "lens": lens}


DEFAULT_KNOWLEDGE_SYSTEM = (
    "You are Aletheia in compiler mode. Preserve epistemic discipline, "
    "derive answers from validated context only, and produce concise, "
    "deterministic reasoning outputs."
)

# Theorist mode system prompt
DEFAULT_THEORIST_SYSTEM = (
    "You are Aletheia in theorist mode. Explain theoretical concepts, synthesize knowledge, "
    "and map causal relationships based on the provided constraints."
)

DEFAULT_VETERAN_SYSTEM = (
    "Veteran Mode: Analyze the reasoning path, identify logical drift, "
    "and output a corrective trajectory grounded in Advocate diagnostics."
)

_INTERNAL_KEYS = {
    "node_id",
    "execution_eligible",
    "_reroll_count",
    "_reroll_violation_type",
    "_branch_degraded",
    "_confidence_decomposition",
    "_audit_trail",
    "_sie_coherence_score",
    "_invariant_results",
    "_acs_structured_constraints",
    "_acs_trajectory",
    "_governance_directive",
    "_reroll_context",
    "_adversarial_verdict",
    "_adversarial_conflict",
    # SIE / scoring internals — must not leak into SFT output
    "sie_node", "acs_score", "c_final", "sie_score", "sie_pass",
    "system_backpressure", "field_pressure",
    "content_density", "alignment_vector", "composite_quality_score",
    "mode_scaling_factor", "s_sie",
    "rho", "phase_gradient", "J_info", "kappa",  # legacy names
    "topology_cluster", "acs_handshake_sid", "acs_violations", "acs_audited",
    "v_score", "validation_pass",
    "_sie_metadata",
}


def _stable_json_blob(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)


def downsample_mode_records(
    records: List[Dict[str, Any]],
    target_mode: str = "veteran",
    max_target_ratio: float = 1.0,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Deterministically cap one mode relative to all other records.

    This is distribution control for SFT compilation, not semantic rejection:
    source failure matrices remain intact, and the cap is opt-in at the
    formatter boundary.
    """
    if max_target_ratio < 0:
        raise ValueError("max_target_ratio must be >= 0")

    target_records = [
        rec for rec in records
        if isinstance(rec, dict) and str(rec.get("_mode", "")) == target_mode
    ]
    non_target_records = [
        rec for rec in records
        if not (isinstance(rec, dict) and str(rec.get("_mode", "")) == target_mode)
    ]

    if not target_records or not non_target_records:
        return list(records)

    allowed_target_count = math.floor(len(non_target_records) * max_target_ratio)
    if len(target_records) <= allowed_target_count:
        return list(records)

    ordered_targets = sorted(target_records, key=_stable_json_blob)
    rng = random.Random(seed)
    selected_targets = rng.sample(ordered_targets, allowed_target_count)
    return non_target_records + selected_targets


def _scrub_internal_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        clean: Dict[str, Any] = {}
        for key in sorted(value.keys()):
            if key in _INTERNAL_KEYS or key.startswith("_"):
                continue
            clean[key] = _scrub_internal_metadata(value[key])
        return clean
    if isinstance(value, list):
        return [_scrub_internal_metadata(item) for item in value]
    return value


def _apply_loss_masking_schema(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    return {
        "messages": [
            {
                "role": msg["role"],
                "content": msg["content"],
                "train_loss": (msg["role"] == "assistant"),
            }
            for msg in messages
        ]
    }


# Placeholder strings emitted by the DAG runtime's Procedural_Mapping lens.
# These carry zero epistemic value and must never reach the training set.
_PLACEHOLDER_OUTPUTS = frozenset({
    "Generated_via_Procedural_Mapping",
    "generated_via_procedural_mapping",
    "",
})


def _sanitize_field(text: Any) -> str:
    """Remove angle brackets and normalize whitespace for safe embedding inside <think> blocks."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.replace("<", "").replace(">", "").replace("\n", " ")
    return " ".join(text.split())


def _render_branch_trace(bt_dict: Dict[str, Any], final_answer: str) -> str:
    """Render a serialized BranchTrace dict into a deterministic <think>...</think> block.

    Accepts the raw dict (not the Pydantic object) so this function has no Pydantic dependency.
    All field values are sanitized to prevent nested angle brackets inside the XML block.
    Output is byte-stable: same input → identical output, no random() calls.
    """
    baseline  = _sanitize_field(bt_dict.get("baseline_state", ""))
    synthesis = _sanitize_field(bt_dict.get("synthesis", ""))
    branches  = bt_dict.get("branches") or []

    lines: List[str] = ["<think>", f"Baseline: {baseline}", ""]
    for b in branches:
        bid        = _sanitize_field(b.get("branch_id", ""))
        condition  = _sanitize_field(b.get("condition", ""))
        mechanism  = _sanitize_field(b.get("mechanism", ""))
        first_ord  = _sanitize_field(b.get("first_order_effect", ""))
        second_ord = _sanitize_field(b.get("second_order_effect", ""))
        failure    = _sanitize_field(b.get("failure_boundary", ""))
        risk       = _sanitize_field(b.get("assumption_risk", ""))
        refs       = b.get("evidence_refs") or []
        refs_str   = ", ".join(_sanitize_field(r) for r in refs) if refs else "none"
        lines += [
            f"Branch: {bid}",
            f"  Condition: {condition}",
            f"  Mechanism: {mechanism}",
            f"  First-order: {first_ord}",
            f"  Second-order: {second_ord}",
            f"  Failure boundary: {failure}",
            f"  Risk: {risk}",
            f"  Evidence refs: {refs_str}",
            "",
        ]
    lines += [f"Synthesis: {synthesis}", "</think>"]
    return "\n".join(lines) + "\n" + _sanitize_field(final_answer)


def _knowledge_chatml_record(payload: Dict[str, Any], identity: Optional[str]) -> Dict[str, Any]:

    # Detect theorist nodes
    skill_type = payload.get("skill_type", "") or payload.get("type", "")
    source_type = payload.get("source_type", "")
    mode = payload.get("mode", "")
    is_theorist = (
        skill_type == "theoretical_reasoning"
        or source_type == "ocr_document"
        or mode == "theorist"
    )

    _sie_lens = None  # populated only for theorist records that clear the SIE gate

    if is_theorist:
        system_prompt = identity or DEFAULT_THEORIST_SYSTEM
        concept = str(payload.get("name") or "").strip()

        # --- Hard guard: theorist records must carry valid RAL dual-channel fields. ---
        # Records missing reasoning_trace + final_answer are dropped unconditionally.
        # This closes the OCR-as-assistant-output path for any theorist record that
        # bypassed RAL or failed amplification.
        reasoning_trace = payload.get("reasoning_trace")
        final_answer    = payload.get("final_answer", "")
        schema_version  = payload.get("schema_version", "1.0")

        # v2 branch schema path — BranchTrace render (Phase 3).
        if schema_version == "2.0":
            branch_trace_dict = payload.get("branch_trace")
            if not isinstance(branch_trace_dict, dict) or not final_answer:
                return {}
            # Validate the stored branch_trace against the contract before rendering.
            try:
                from src.pipeline.branching_contracts import BranchTrace, evaluate_branch_topology
                bt       = BranchTrace.model_validate(branch_trace_dict)
                topology = evaluate_branch_topology(bt)
            except Exception:
                return {}
            output_text  = _render_branch_trace(branch_trace_dict, final_answer)
            user_content = get_dynamic_prompt(concept)
            messages = [
                {"role": "system",    "content": system_prompt},
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": output_text},
            ]
            record = _apply_loss_masking_schema(messages)
            record["_mode"]          = "knowledge"
            record["schema_version"] = "2.0"
            record["reasoning_type"] = payload.get("reasoning_type", "standard")
            # Prefer upstream RAL/evaluator values; derive from validated BranchTrace as fallback
            # so the formatted row is always auditable even if upstream omitted the scalar metrics.
            record["branch_count"] = payload.get("branch_count", topology.get("branch_count", len(branch_trace_dict.get("branches") or [])))
            record["failure_boundary_coverage"] = (
                payload["failure_boundary_coverage"]
                if payload.get("failure_boundary_coverage") is not None
                else topology.get("failure_boundary_coverage", 0.0)
            )
            record["max_mechanism_similarity"] = (
                payload["max_mechanism_similarity"]
                if payload.get("max_mechanism_similarity") is not None
                else topology.get("max_mechanism_similarity", 0.0)
            )
            record["condition_coverage"] = (
                payload["condition_coverage"]
                if payload.get("condition_coverage") is not None
                else topology.get("condition_coverage", 0.0)
            )
            return record

        # v1 dual-channel guard — both fields required, reasoning_trace must be a non-empty list.
        if not (isinstance(reasoning_trace, list) and len(reasoning_trace) > 0 and final_answer):
            return {}

        # --- Dual-channel fast-path (RAL v1: reasoning_trace + final_answer) ---
        if isinstance(reasoning_trace, list) and final_answer:
            safe_steps   = [
                step.replace("<", "").replace(">", "").replace("\n", " ").strip()
                for step in reasoning_trace
            ]
            think_block  = "<think>\n" + "\n".join(f"- {s}" for s in safe_steps) + "\n</think>"
            output_text  = f"{think_block}\n{final_answer}"
            user_content = get_dynamic_prompt(concept)
            messages = [
                {"role": "system",    "content": system_prompt},
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": output_text},
            ]
            record = _apply_loss_masking_schema(messages)
            record["_mode"]          = "knowledge"
            record["reasoning_type"] = payload.get("reasoning_type", "standard")
            return record

    elif mode == "veteran":
        system_prompt = "Veteran Mode: Analyze the reasoning path, identify logical drift, and output a corrective trajectory grounded in Advocate diagnostics."
        user_content = payload.get("instruction", "")
        output_text = str(payload.get("output", "")).strip()
        if not user_content or not output_text:
            print(f"DEBUG: Dropped veteran node {payload.get('name')} because instruction or output_text is empty.")
            return {}
    elif mode == "advocate":
        system_prompt = "Advocate Mode: Evaluate architectural integrity, balance theoretical constraints with implementation logic, and validate YAML configurations."
        user_content = payload.get("instruction", "Analyze the following configuration:")
        output_text = str(payload.get("output", "")).strip()
        if not user_content or not output_text:
            print(f"DEBUG: Dropped advocate node {payload.get('name')} because instruction or output_text is empty.")
            return {}
    else:
        system_prompt = identity or DEFAULT_KNOWLEDGE_SYSTEM
        instruction = str(payload.get("instruction", "")).strip()
        input_text = str(payload.get("input", "")).strip()
        output_text = str(payload.get("output", "")).strip()

        if output_text in _PLACEHOLDER_OUTPUTS:
            code_snippet = str(payload.get("code_snippet", "")).strip()
            if code_snippet:
                output_text = f"```python\n{code_snippet}\n```"
            else:
                print(f"DEBUG: Dropped coding node {payload.get('name')} because output and code_snippet are empty or placeholder.")
                return {}

        if not instruction or not output_text:
            print(f"DEBUG: Dropped coding node {payload.get('name')} because instruction or output_text is empty.")
            return {}

        user_content = instruction if not input_text else f"{instruction}\n\nContext:\n{input_text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output_text},
    ]
    record = _apply_loss_masking_schema(messages)
    record["_mode"] = "knowledge"
    if _sie_lens is not None:
        record["sie_lens"]       = _sie_lens
        record["reasoning_type"] = payload.get("_ral_mode", "standard")
    return record


def _knowledge_from_existing_messages(payload: Dict[str, Any], identity: Optional[str]) -> Dict[str, Any]:
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        return {}

    cleaned: List[Dict[str, str]] = []
    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip()
        if role not in ("system", "user", "assistant"):
            continue
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        cleaned.append({"role": role, "content": content})

    if not cleaned:
        return {}

    if cleaned[0]["role"] != "system":
        cleaned.insert(0, {"role": "system", "content": identity or DEFAULT_KNOWLEDGE_SYSTEM})
    elif identity:
        cleaned[0]["content"] = identity

    has_user = any(m["role"] == "user" for m in cleaned)
    has_assistant = any(m["role"] == "assistant" for m in cleaned)
    if not has_user or not has_assistant:
        return {}

    record = _apply_loss_masking_schema(cleaned)
    record["_mode"] = "knowledge"
    return record


# Orchestration-only failure types that carry no genuine reasoning signal.
# Training on these teaches the LLM about internal pipeline timeouts and
# threshold mechanics rather than actual code/logic errors.
_ORCHESTRATION_ONLY_FAILURES = frozenset({
    "depth_limit",
    "below_instability_band",
    "omega_orphan_state",
    "topological_cycle_break",
    "cycle_quarantine_emptied_graph",
    "parent_chain_failed",
    "cascade_rejection",
})


def _is_orchestration_only_failure(payload: Dict[str, Any]) -> bool:
    """Return True if the failure record is a pipeline-internal artifact
    rather than a genuine reasoning or schema error worth training on."""
    failure_type = str(payload.get("failure_type", "")).strip().lower()
    rejection_reason = str(payload.get("rejection_reason", "")).strip().lower()

    for tag in _ORCHESTRATION_ONLY_FAILURES:
        if tag in failure_type or tag in rejection_reason:
            return True

    # Also filter cascade rejections (e.g. "cascade_from_starlette/...")
    if rejection_reason.startswith("cascade_from_"):
        return True

    return False


def _failure_chatml_record(payload: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = _scrub_internal_metadata(payload)
    rejection_reason = str(cleaned.get("rejection_reason", "unknown_rejection"))
    failure_type = str(cleaned.get("failure_type", "unknown"))
    sie_snapshot = cleaned.get("sie_snapshot", {})
    trajectory = cleaned.get("trajectory", {})
    advocate_audit = cleaned.get("advocate_audit", {})

    user_payload = {
        "failure_type": failure_type,
        "rejection_reason": rejection_reason,
        "sie_snapshot": sie_snapshot,
    }
    assistant_payload = {
        "corrective_trajectory": trajectory,
        "advocate_audit": advocate_audit,
        "remediation_status": cleaned.get("remediation_status", "abandoned"),
    }

    messages = [
        {"role": "system", "content": DEFAULT_VETERAN_SYSTEM},
        {"role": "user", "content": _stable_json_blob(user_payload)},
        {"role": "assistant", "content": _stable_json_blob(assistant_payload)},
    ]
    record = _apply_loss_masking_schema(messages)
    record["_mode"] = "veteran"
    return record


def _iter_json_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    records: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    records.append(obj)
        return records

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            try:
                obj = json.load(f)
            except json.JSONDecodeError:
                return []
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return [item for item in obj if isinstance(item, dict)]
    return []


def _looks_like_formatter_output(path: Path) -> bool:
    """Return True if ``path`` appears to be a prior sft_formatter artifact.

    We sniff the first non-empty record and check for the internal ``_mode``
    marker that only this formatter writes. This is a deterministic state
    reset: re-running ``format_complexes`` into a directory that already
    contains prior outputs must never re-ingest those outputs, otherwise
    repeat invocations double-count rows and byte stability breaks.
    """
    try:
        if not path.exists() or not path.is_file():
            return False
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        return False
                    return isinstance(obj, dict) and "_mode" in obj
        elif suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                try:
                    obj = json.load(f)
                except json.JSONDecodeError:
                    return False
            if isinstance(obj, dict):
                return "_mode" in obj
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                return "_mode" in obj[0]
    except OSError:
        return False
    return False


def _discover_inputs(
    input_dir: Path,
    failure_matrix_file: Optional[Path],
    excluded_paths: Optional[List[Path]] = None,
) -> Tuple[List[Path], List[Path]]:
    """Discover knowledge and failure input files in ``input_dir``.

    ``excluded_paths`` lets callers (notably ``format_complexes``) exclude
    the output artifact so repeated invocations into the same directory
    remain idempotent. In addition, any file whose first record carries
    the internal ``_mode`` marker is treated as a prior formatter output
    and skipped — this guarantees determinism even when multiple outputs
    have been written into the same directory over time.
    """
    knowledge_files: List[Path] = []
    failure_files: List[Path] = []

    excluded_resolved = {p.resolve() for p in (excluded_paths or [])}

    for path in sorted(input_dir.glob("*.jsonl")) + sorted(input_dir.glob("*.json")):
        if path.resolve() in excluded_resolved:
            continue
        if _looks_like_formatter_output(path):
            continue
        name = path.name.lower()
        if "failure_matrix" in name:
            failure_files.append(path)
        else:
            knowledge_files.append(path)

    if failure_matrix_file:
        failure_files.append(failure_matrix_file)

    unique_knowledge = sorted({p.resolve() for p in knowledge_files})
    unique_failure = sorted({p.resolve() for p in failure_files})
    return unique_knowledge, unique_failure


def _fetch_lima_mixin(target_count: int) -> List[Dict[str, Any]]:
    if target_count <= 0:
        return []
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("[SFT] Alpaca mix-in skipped: 'datasets' library not installed. Run: pip install datasets")
        return []
    try:
        ds = load_dataset("yahma/alpaca-cleaned", split="train", trust_remote_code=False)
        records: List[Dict[str, Any]] = []
        for row in ds:
            if len(records) >= target_count:
                break
            user_msg = str(row.get("instruction", "")).strip()
            extra_input = str(row.get("input", "")).strip()
            if extra_input:
                user_msg = f"{user_msg}\n{extra_input}"
            assistant_msg = str(row.get("output", "")).strip()
            if not user_msg or not assistant_msg:
                continue
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
            rec = _apply_loss_masking_schema(messages)
            rec["_mode"] = "alpaca"
            records.append(rec)
        print(f"[SFT] Fetched {len(records)} alpaca mix-in records (target={target_count}).")
        return records
    except Exception as e:
        print(f"[SFT] Alpaca mix-in failed: {e}")
        return []


def format_complexes(
    input_dir: Path,
    output_file: Path,
    split: float = 1.0,
    identity: Optional[str] = None,
    failure_matrix_file: Optional[Path] = None,
    max_veteran_ratio: Optional[float] = None,
    balance_seed: int = 42,
) -> int:
    del split  # Reserved for compatibility.
    _SIE_LENS_COUNTS.clear()  # reset per-run to prevent cross-invocation state bleed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # Exclude the output artifact itself from input discovery so repeated
    # invocations into the same directory remain idempotent (state-reset
    # per invocation: no module-level accumulators, no mutable defaults).
    knowledge_files, failure_files = _discover_inputs(
        input_dir,
        failure_matrix_file,
        excluded_paths=[output_file],
    )

    records: List[Dict[str, Any]] = []

    # Fields that can carry the assistant-turn content for a knowledge record.
    # Allows bridge_theorist.py to emit `reasoning` or `chemical_analysis` instead
    # of `output` without the record being silently dropped here.
    _OUTPUT_FIELD_ALIASES = ("output", "reasoning", "chemical_analysis")

    for file_path in knowledge_files:
        for payload in _iter_json_records(Path(file_path)):
            print(f"DEBUG PAYLOAD KEYS: {list(payload.keys())}")
            has_output_field = any(k in payload for k in _OUTPUT_FIELD_ALIASES)
            print(f"DEBUG EVAL: instruction={'instruction' in payload}, output_field={has_output_field}")
            if "instruction" in payload and has_output_field:
                rec = _knowledge_chatml_record(_scrub_internal_metadata(payload), identity)
                if rec:
                    records.append(rec)
            else:
                print("DEBUG: Dropped at primary key check")
            if "messages" in payload:
                rec = _knowledge_from_existing_messages(_scrub_internal_metadata(payload), identity)
                if rec:
                    records.append(rec)

    veteran_skipped_orchestration = 0
    for file_path in failure_files:
        for payload in _iter_json_records(Path(file_path)):
            # Only consume FAILURE_MATRIX-style rows.
            if "trajectory" in payload and "advocate_audit" in payload:
                # V2.1: Exclude orchestration-only failures (depth_limit,
                # below_instability_band, cascade_from_*, etc.) that carry
                # no genuine reasoning signal for the target LLM.
                if _is_orchestration_only_failure(payload):
                    veteran_skipped_orchestration += 1
                    continue
                rec = _failure_chatml_record(payload)
                if rec:
                    records.append(rec)

    if veteran_skipped_orchestration:
        print(
            f"[SFT] Filtered {veteran_skipped_orchestration} orchestration-only "
            f"veteran traces (depth_limit, cascade, etc.)"
        )

    if max_veteran_ratio is not None:
        veteran_before = sum(1 for rec in records if rec.get("_mode") == "veteran")
        records = downsample_mode_records(
            records,
            target_mode="veteran",
            max_target_ratio=max_veteran_ratio,
            seed=balance_seed,
        )
        veteran_after = sum(1 for rec in records if rec.get("_mode") == "veteran")
        if veteran_after != veteran_before:
            print(
                "[SFT] Downsampled veteran rows "
                f"{veteran_after}/{veteran_before} "
                f"(max_ratio={max_veteran_ratio}, seed={balance_seed})"
            )

    # Inject LIMA mix-in at a 10:2 ratio (20% of domain record count).
    lima_count = int(len(records) * 0.2)
    if lima_count > 0:
        lima_records = _fetch_lima_mixin(lima_count)
        records.extend(lima_records)
        print(f"[SFT] Injected {len(lima_records)} LIMA mix-in records into sort pool.")

    records_sorted = sorted(records, key=_stable_json_blob)
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in records_sorted:
            f.write(_stable_json_blob(rec) + "\n")

    print(f"[SFT] Wrote {len(records_sorted)} strict ChatML rows to {output_file}")
    if _SIE_LENS_COUNTS:
        total_sie = sum(_SIE_LENS_COUNTS.values())
        print("[SFT] SIE lens distribution:")
        for _lens, _cnt in sorted(_SIE_LENS_COUNTS.items(), key=lambda x: -x[1]):
            print(f"  {_lens:<14} {_cnt:>5}  ({_cnt / total_sie * 100:.1f}%)")
    return len(records_sorted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aletheia strict ChatML formatter")
    parser.add_argument("input_dir")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--apply-loss-masking", action="store_true")
    parser.add_argument("--split", type=float, default=1.0)
    parser.add_argument("--identity", type=str, default=None)
    parser.add_argument("--failure-matrix", type=str, default=None)
    parser.add_argument(
        "--max-veteran-ratio",
        type=float,
        default=None,
        help="Opt-in cap for veteran rows relative to all non-veteran rows, e.g. 1.0 or 0.5.",
    )
    parser.add_argument(
        "--balance-seed",
        type=int,
        default=42,
        help="Deterministic sampling seed used with --max-veteran-ratio.",
    )
    args = parser.parse_args()

    # Loss masking is always applied in strict ChatML output; keep flag for CLI compatibility.
    del args.apply_loss_masking
    fm = Path(args.failure_matrix) if args.failure_matrix else None
    format_complexes(
        Path(args.input_dir),
        Path(args.output),
        args.split,
        args.identity,
        fm,
        max_veteran_ratio=args.max_veteran_ratio,
        balance_seed=args.balance_seed,
    )
