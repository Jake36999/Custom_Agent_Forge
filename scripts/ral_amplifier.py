"""
RAL — Reasoning Amplification Layer
=====================================
Transforms shallow OCR extracts into causal 80-150 word chemistry explanations
via the Claude API, then applies hard quality gates before writing output.

Pipeline position:
    bridge_theorist.py → extracted_theorist.jsonl
    ral_amplifier.py   → reads extracted_theorist.jsonl → writes amplified_theorist.jsonl
    sft_formatter.py   → reads output/datasets/theorist_amplified/ → combined_theorist_v2.jsonl

Usage:
    # Dry-run: preview 5 expansions, no file written
    python scripts/ral_amplifier.py --dry-run

    # Full run
    python scripts/ral_amplifier.py

    # Test with 50 nodes
    python scripts/ral_amplifier.py --limit 50

    # Resume interrupted run (checkpoint is preserved automatically)
    python scripts/ral_amplifier.py
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

# Branch schema and evaluator (Phase 2).
# Imported lazily inside the v2 path so v1 runs never touch Pydantic.
# Full import at module level kept here for IDE navigation.
try:
    from src.pipeline.branching_contracts import (
        BranchTrace,
        evaluate_branch_topology,
        MIN_BRANCH_COUNT,
        BRANCH_MECHANISM_SIMILARITY_MAX,
    )
    _BRANCH_CONTRACTS_AVAILABLE = True
except ImportError:
    _BRANCH_CONTRACTS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_PATH      = "output/datasets/theorist/extracted_theorist.jsonl"
OUTPUT_DIR      = "output/datasets/theorist_amplified"
OUTPUT_PATH     = f"{OUTPUT_DIR}/amplified_theorist.jsonl"
CHECKPOINT_PATH = f"{OUTPUT_DIR}/.ral_checkpoint_v2.json"

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
DEFAULT_MODEL  = "gpt-4o-mini"   # fast / low-cost for bulk work
MAX_TOKENS     = 700
RATE_SLEEP_SEC = 0.05   # seconds between API calls (stays well under rate limits)

# ---------------------------------------------------------------------------
# Mode detection keyword sets
# ---------------------------------------------------------------------------
_THEORIST_KW = {
    "mechanism", "pathway", "kinetics", "thermodynamic", "equilibrium",
    "activation", "entropy", "enthalpy", "orbital", "electron", "bond",
    "reaction", "synthesis", "catalysis", "isomer", "polymer", "radical",
    "spectroscopy", "diffraction", "adsorption", "chemisorption",
}
_ADVOCATE_KW = {
    "comparison", "advantage", "property", "characteristic", "application",
    "alternative", "method", "approach", "technique", "grade", "specification",
    "standard", "performance", "stability", "yield", "purity", "selectivity",
}
_VETERAN_KW = {
    "failure", "toxicity", "hazard", "risk", "adverse", "side effect",
    "inhibit", "degrade", "decompose", "unstable", "contamination",
    "poisoning", "overdose", "defect", "limitation", "problem",
}

# ---------------------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------------------
HARD_MIN_WORDS = 60

CAUSAL_RE = re.compile(
    r"\b(because|therefore|consequently|thus|hence|results?\s+in|leads?\s+to|"
    r"causes?|due\s+to|thereby|as\s+a\s+result|which\s+means|proceeds?|"
    r"this\s+means|enabling|allowing|preventing|driving)\b",
    re.I,
)

# Fix 1: second gate — must reference actual chemistry mechanism, not just causal words
MECHANISM_RE = re.compile(
    r"\b(activation\s+energy|electron|bond|interaction|collision|transfer|orbital|"
    r"kinetic|thermodynamic|reaction\s+rate|surface|molecule|atom|ion|charge|"
    r"temperature|pressure|concentration|catalyst|solvent|phase|crystal|lattice|"
    r"diffusion|adsorption|desorption|equilibrium|enthalpy|entropy)\b",
    re.I,
)

# Concepts whose names start with these words are fragments, not concepts.
BAD_STARTS_RE = re.compile(
    r"^\s*(this|that|these|those|than|and|or|of|from|to|a|an|the|"
    r"such|similar|with|without|in|on|at|by|for|as|"
    r"some|many|all|other|their|its|any|most|another|"
    r"they|which|who|what|after|before|"
    r"highly|large|small|high|low|very|quite|"
    r"millions?|thousands?|hundreds?)\b"
    r"|^edu\s+\d+\b",
    re.I,
)
# Reject names that are purely quantitative measurements (e.g. "103 1 kJ mol 24 65 kcal mol")
UNIT_ONLY_RE = re.compile(
    r"^\d[\d\s\.\-]*\b(kJ|kcal|mol|moles?|mL|kg|kPa|MPa|bar|atm)\b",
    re.I,
)


# ---------------------------------------------------------------------------
# Mode detection  (Fix 2: lower thresholds to prevent theorist collapse)
# ---------------------------------------------------------------------------
def detect_mode(name: str, content: str) -> str:
    tokens = set(re.findall(r"[a-z]+", (name + " " + content).lower()))
    v = len(tokens & _VETERAN_KW)
    a = len(tokens & _ADVOCATE_KW)
    t = len(tokens & _THEORIST_KW)
    if v >= 2:
        return "veteran"
    if a >= 1:
        return "advocate"
    if t >= 1:
        return "theorist"
    return random.choice(["theorist", "advocate"])


# ---------------------------------------------------------------------------
# Prompt builders  (Fix 4: randomised phrasing variants to prevent templating)
# ---------------------------------------------------------------------------
_MECHANISM_VARIANTS = [
    "Explain the mechanism — WHY it behaves this way. Use 'because', 'due to', or 'therefore'.",
    "Describe the underlying mechanism. Explain why this occurs at the molecular or atomic level.",
    "Outline the causal process. Use 'because' or 'due to' to link the cause to the effect.",
]
_TRADEOFF_VARIANTS = [
    "State the key trade-off using 'therefore' or 'consequently'.",
    "Identify the critical trade-off. Use 'consequently' or 'as a result' to connect cause and effect.",
    "Explain the trade-off between the competing factors using 'therefore' or 'however'.",
]
_FAILURE_VARIANTS = [
    "Explain the failure mechanism using 'because' or 'due to'.",
    "Describe why the failure occurs at a mechanistic level. Use 'because' or 'due to'.",
    "Outline the causal chain that leads to failure using 'because' or 'results in'.",
]


def build_system_prompt(mode: str) -> str:  # mode retained for _ral_mode tagging; prompt is unified
    return (
        "You are a structured chemistry reasoning engine generating training data.\n\n"
        "You MUST return ONLY valid JSON. No text before or after. No markdown fences.\n\n"
        "The JSON must contain exactly two keys:\n"
        '  "reasoning_trace": a list of 3 to 8 concise reasoning steps\n'
        '  "final_answer": a single string summarizing the mechanism\n\n'
        "STRICT RULES FOR reasoning_trace:\n"
        "- Each step must be 6 to 40 words\n"
        "- Each step must describe a causal or mechanistic relationship\n"
        "- Do NOT use quotation marks inside steps\n"
        "- Do NOT use angle brackets < or > inside steps\n"
        "- Do NOT use newline characters inside steps\n\n"
        "STRICT RULES FOR final_answer:\n"
        "- Must be 50 to 120 words\n"
        "- Must contain at least one of: because, therefore, leads to, results in\n"
        "- Must summarize the mechanism in readable prose (not repeat the steps)\n\n"
        "If you cannot generate a valid response, return exactly:\n"
        '{"reasoning_trace": [], "final_answer": ""}'
    )


def build_user_msg(name: str, content: str) -> str:
    return (
        f"Concept: {name}\n\n"
        f"Fragment: {content}\n\n"
        "Requirements:\n"
        "- Explain the causal mechanism step by step\n"
        "- Focus on cause and effect at the molecular or atomic level\n"
        "- Avoid generic or descriptive-only explanations\n"
        "- If the concept is abstract or broad, anchor your reasoning to a specific "
        "chemical mechanism, reaction type, or physical process. Do not remain at a "
        "purely descriptive level."
    )


# ---------------------------------------------------------------------------
# Schema v2 prompt builder and parser
# ---------------------------------------------------------------------------

def build_system_prompt_v2() -> str:
    """System prompt requesting BranchTrace JSON (schema_version 2.0)."""
    return (
        "You are a structured chemistry reasoning engine generating branch-aware training data.\n\n"
        "You MUST return ONLY valid JSON. No text before or after. No markdown fences.\n\n"
        "Return an object matching this exact schema:\n"
        '{\n'
        '  "schema_version": "2.0",\n'
        '  "baseline_state": "<20+ words: state of the system before any branch fires>",\n'
        '  "branches": [\n'
        '    {\n'
        '      "branch_id": "<short_label>",\n'
        '      "condition": "If <parameter or regime> ...",\n'
        '      "mechanism": "<molecular or atomic level explanation, 20+ words>",\n'
        '      "first_order_effect": "<immediate consequence>",\n'
        '      "second_order_effect": "<downstream or cascading consequence>",\n'
        '      "failure_boundary": "<where this branch logic fails or breaks down>",\n'
        '      "assumption_risk": "low|medium|high",\n'
        '      "evidence_refs": []\n'
        '    }\n'
        '  ],\n'
        '  "synthesis": "<20+ words: integrating conclusion across all branches>",\n'
        '  "final_answer": "<50-120 word prose summary containing because/therefore/leads to>"\n'
        '}\n\n'
        "RULES:\n"
        "- Include at least 2 branches with GENUINELY DIFFERENT mechanisms — not rephrased variants\n"
        "- Prefer 3 branches when natural: parameter increases / decreases / stable equilibrium regime\n"
        "- Each branch condition must begin with 'If' or 'When'\n"
        "- failure_boundary is REQUIRED for every branch\n"
        "- final_answer must contain at least one of: because, therefore, leads to, results in\n"
        "- Do NOT use angle brackets < > inside field values\n\n"
        "If you cannot generate a valid response, return exactly:\n"
        '{"error": "cannot_generate"}'
    )


def parse_ral_output_v2(response: str):
    """Parse and validate an LLM response as BranchTrace (schema v2.0).

    Returns (BranchTrace, None) on success or (None, error_string) on failure.
    Requires branching_contracts to be importable.
    """
    if not _BRANCH_CONTRACTS_AVAILABLE:
        return None, "branching_contracts_unavailable"

    from pydantic import ValidationError  # already a project dependency via contracts.py

    clean = response.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean.strip())

    try:
        data = json.loads(clean)
    except Exception:
        return None, "json_parse_fail"

    if not isinstance(data, dict):
        return None, "json_parse_fail"

    if data.get("error") == "cannot_generate":
        return None, "model_declined"

    try:
        bt = BranchTrace.model_validate(data)
    except ValidationError as exc:
        errors = exc.errors()
        msg = str(errors[0].get("msg", "")) if errors else ""
        loc = str(errors[0].get("loc", "")) if errors else ""
        if "min_length" in msg and "branches" in loc:
            return None, "insufficient_branches"
        if "failure_boundary" in msg or "failure_boundary" in loc:
            return None, "missing_failure_boundary"
        return None, "schema_validation_fail"

    return bt, None


# ---------------------------------------------------------------------------
# Dual-channel parser & validation (replaces score_expansion)
# ---------------------------------------------------------------------------
_CAUSAL_SIGNAL_RE = re.compile(
    r"\b(because|therefore|leads?\s+to|results?\s+in|consequently|due\s+to)\b",
    re.I,
)


def parse_ral_output(response: str) -> tuple:
    """Parse and validate the LLM's JSON dual-channel response.

    Returns (parsed_dict, None) on success or (None, error_string) on failure.
    """
    clean = response.strip()
    # Strip markdown fences if the model wrapped the JSON
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean.strip())

    try:
        data = json.loads(clean)
    except Exception:
        return None, "json_parse_fail"

    if not isinstance(data, dict):
        return None, "json_parse_fail"

    if not all(k in data for k in ("reasoning_trace", "final_answer")):
        return None, "missing_keys"

    reasoning = data["reasoning_trace"]
    answer    = data["final_answer"]

    if not isinstance(reasoning, list) or not isinstance(answer, str):
        return None, "invalid_types"

    if len(reasoning) < 3 or len(reasoning) > 8:
        return None, "invalid_reasoning_length"

    for step in reasoning:
        if not isinstance(step, str):
            return None, "invalid_step_type"
        if "<" in step or ">" in step:
            return None, "unsafe_characters"
        if len(step.split()) < 6:
            return None, "low_density_step"

    if len(answer.split()) < 40:
        return None, "short_answer"

    if not _CAUSAL_SIGNAL_RE.search(answer):
        return None, "weak_causal_signal"

    # Entropy guard: catches circular/degenerate traces ("happens because it occurs")
    all_trace_words = " ".join(reasoning).split()
    if all_trace_words:
        unique_ratio = len(set(all_trace_words)) / len(all_trace_words)
        if unique_ratio < 0.5:
            return None, "low_reasoning_entropy"

    # Step diversity guard: catches near-duplicate steps
    if len(set(reasoning)) < len(reasoning) * 0.7:
        return None, "duplicate_steps"

    return {"reasoning_trace": reasoning, "final_answer": answer}, None


# ---------------------------------------------------------------------------
# Reasoning Structure Evaluator
# ---------------------------------------------------------------------------
_BRANCH_WORDS     = ["however", "alternatively", "in contrast", "on the other hand", "conversely", "whereas", "by contrast"]
_FAILURE_WORDS    = [
    "fails", "breaks", "unstable", "degrades", "collapses",
    "side reaction", "byproduct formation", "over-oxidation", "under-reaction",
    "decomposes", "precipitates", "corrodes",
]
_CONDITIONAL_WORDS = ["if", "unless", "only when", "provided that", "when"]
_TRANSFORM_WORDS  = [
    "causes", "leads to", "results in", "increases", "decreases",
    "enhances", "reduces", "inhibits", "accelerates", "slows",
    "stabilizes", "destabilizes", "converts", "transforms",
]


def evaluate_reasoning_structure(reasoning_trace: list) -> dict:
    text = " ".join(reasoning_trace).lower()
    words = text.split()
    word_count = max(len(words), 1)

    has_branching   = any(w in text for w in _BRANCH_WORDS)
    has_failure     = any(w in text for w in _FAILURE_WORDS)
    has_conditional = any(w in text for w in _CONDITIONAL_WORDS)
    has_transform   = any(w in text for w in _TRANSFORM_WORDS)
    diversity       = len(set(words)) / word_count

    if has_failure or has_branching:
        structure = "divergent"
    elif has_conditional:
        structure = "conditional"
    else:
        structure = "linear"

    return {
        "structure_type":  structure,
        "has_branching":   has_branching,
        "has_failure":     has_failure,
        "has_conditional": has_conditional,
        "has_transform":   has_transform,
        "diversity_score": round(diversity, 4),
        "word_count":      word_count,
    }


# ---------------------------------------------------------------------------
# OCR artifact normalisation (applied to name + content before API call)
# ---------------------------------------------------------------------------
_OCR_REPLACEMENTS = [
    ("pig/mL",  "µg/mL"),
    ("pig/ml",  "µg/mL"),
    ("\u2014-", "-"),   # em-dash followed by hyphen
    ("\u2014",  "-"),   # bare em-dash
    ("\u2013",  "-"),   # en-dash
]


def clean_ocr_artifacts(text: str) -> str:
    for bad, good in _OCR_REPLACEMENTS:
        text = text.replace(bad, good)
    return text


# ---------------------------------------------------------------------------
# Pre-filter (skip obviously bad concepts before spending API budget)
# ---------------------------------------------------------------------------
def is_valid_concept(name: str, content: str) -> bool:
    name = name.strip()
    if not name or BAD_STARTS_RE.match(name):
        return False
    if not re.search(r"[a-zA-Z]{3,}", name):
        return False
    if UNIT_ONLY_RE.match(name):
        return False
    # OCR fragment: leading digits + no alpha token longer than 3 chars
    # catches "1 2 C1 O and HOC" while preserving "1 1 1 Trichloroethane"
    if re.match(r"^\d", name):
        alpha_tokens = re.findall(r"[a-zA-Z]+", name)
        if alpha_tokens and max(len(t) for t in alpha_tokens) <= 3:
            return False
    if len(content.split()) < 10:
        return False
    return True


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def load_checkpoint() -> dict:
    p = Path(CHECKPOINT_PATH)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"done": [], "expanded": 0, "rejected": 0, "skipped": 0}


def save_checkpoint(state: dict):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(state, f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="RAL — Reasoning Amplification Layer")
    parser.add_argument("--input",         default=INPUT_PATH,   help="Source JSONL from bridge_theorist")
    parser.add_argument("--output",        default=OUTPUT_PATH,  help="Amplified JSONL output path")
    parser.add_argument("--model",         default=DEFAULT_MODEL, help="OpenAI model to use")
    parser.add_argument("--limit",         type=int, default=0,  help="Cap at N nodes (0 = all)")
    parser.add_argument("--dry-run",       action="store_true",  help="Preview 5 expansions, no file written")
    parser.add_argument("--save-rejected",        action="store_true", default=True,
                        help="Stream rejected nodes to rejected_theorist.jsonl (default: on)")
    parser.add_argument("--sample-ratio",         type=float, default=1.0,
                        help="Fraction of valid nodes to process (0.0–1.0). Deterministic md5 sampling.")
    parser.add_argument("--debug-rejections",     action="store_true", default=False,
                        help="Print a structured line per rejection to stdout")
    parser.add_argument("--debug-sample-output",  action="store_true", default=False,
                        help="Print first 10 accepted outputs to stdout for quality inspection")
    parser.add_argument("--schema-version", choices=["v1", "v2"], default="v1",
                        help="Output schema: v1=flat reasoning_trace (default), v2=BranchTrace objects. "
                             "v2 requires src/pipeline/branching_contracts.py and Pydantic.")
    parser.add_argument("--checkpoint", default=None,
                        help="Override checkpoint file path (default: <output_dir>/.ral_checkpoint_v2.json).")
    args = parser.parse_args()

    global CHECKPOINT_PATH
    if args.checkpoint:
        CHECKPOINT_PATH = args.checkpoint

    if args.schema_version == "v2" and not _BRANCH_CONTRACTS_AVAILABLE:
        print("[RAL] ERROR: --schema-version v2 requires src/pipeline/branching_contracts.py. "
              "Ensure the project root is on PYTHONPATH.")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("[RAL] ERROR: openai library not installed.")
        print("      Run: pip install openai")
        sys.exit(1)

    client = OpenAI()

    # ---- Load input ----
    with open(args.input, encoding="utf-8") as f:
        nodes = [json.loads(l) for l in f if l.strip()]

    # ---- Pre-filter ----
    valid = [
        n for n in nodes
        if is_valid_concept(
            str(n.get("name", "")).strip(),
            str(n.get("output") or n.get("code_snippet", "")).strip(),
        )
    ]

    # Deterministic ordering — stable contrastive scheduling requires a fixed sort
    valid = sorted(valid, key=lambda x: x.get("name", ""))

    # Deterministic sampling by name hash (applied after sort, before --limit)
    if not args.dry_run and args.sample_ratio < 1.0:
        threshold = int(args.sample_ratio * 100)
        valid = [
            n for n in valid
            if int(hashlib.md5(str(n.get("name", "")).encode()).hexdigest(), 16) % 100 < threshold
        ]

    if args.dry_run:
        valid = valid[:5]
    elif args.limit:
        valid = valid[:args.limit]

    print(f"[RAL] {len(nodes)} nodes loaded | {len(valid)} after pre-filter" +
          (f" + {args.sample_ratio*100:.0f}% sampling" if args.sample_ratio < 1.0 else ""))
    if not args.dry_run and args.sample_ratio < 1.0:
        print(f"[RAL] Sample mode: {args.sample_ratio*100:.0f}% of corpus → {len(valid)} nodes selected")
    if not args.dry_run:
        est_calls = len(valid)
        est_cost  = est_calls * (125 * 0.15 + 250 * 0.60) / 1_000_000
        est_min   = est_calls * 1.5 / 60
        print(f"[RAL] Estimated: {est_calls} API calls, ~${est_cost:.2f}, ~{est_min:.0f} min")

    # ---- Setup output ----
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    state    = load_checkpoint() if not args.dry_run else {"done": [], "expanded": 0, "rejected": 0, "skipped": 0}
    done_set = set(state["done"])

    file_mode  = "a" if (not args.dry_run and Path(args.output).exists() and done_set) else "w"
    out_handle = open(args.output, file_mode, encoding="utf-8") if not args.dry_run else None

    # ---- Rejection analytics state (session-level, not checkpointed) ----
    stats = {"expanded": 0, "contrastive": 0, "standard": 0, "rejections": {}, "top_rejected": []}

    rejected_fp = None
    if not args.dry_run and args.save_rejected:
        rejected_fp = open(f"{OUTPUT_DIR}/rejected_theorist.jsonl", "a", encoding="utf-8")

    def _normalize_reason(r: str) -> str:
        # Preserve granular gate names; only collapse the variable-format too_short(Nw)
        return "short_output" if r.startswith("too_short") else r

    def log_rejection(node: dict, reason: str, token_count: int) -> None:
        reason = _normalize_reason(reason)
        if rejected_fp:
            rejected_fp.write(json.dumps({
                "name":        node.get("name"),
                "instruction": node.get("instruction"),
                "reason":      reason,
                "token_count": token_count,
            }, ensure_ascii=False) + "\n")
        bucket = stats["rejections"].setdefault(reason, {"count": 0, "tokens": 0})
        bucket["count"]  += 1
        bucket["tokens"] += token_count
        state["rejected"] = state.get("rejected", 0) + 1
        if args.debug_rejections:
            print(f"[REJECTED] {node.get('name', '')[:60]} | reason={reason} | tokens={token_count}")
        if len(stats["top_rejected"]) < 25:
            stats["top_rejected"].append({
                "name":   node.get("name"),
                "reason": reason,
                "tokens": token_count,
            })

    try:
        for i, node in enumerate(valid):
            name    = clean_ocr_artifacts(str(node.get("name", "")).strip())
            content = clean_ocr_artifacts(str(node.get("output") or node.get("code_snippet", "")).strip())
            uid     = hashlib.md5(name.encode()).hexdigest()[:12]

            if uid in done_set:
                state["skipped"] += 1
                continue

            mode           = detect_mode(name, content)
            is_contrastive = (i % 7 == 0)

            user_msg = build_user_msg(name, content)
            use_v2   = (args.schema_version == "v2")

            if is_contrastive and not use_v2:
                user_msg += (
                    "\n\n[Contrastive Requirement]\n"
                    "Include failure modes or parameter variations in the reasoning_trace. "
                    "Explain what changes when conditions vary."
                )
            elif is_contrastive and use_v2:
                user_msg += (
                    "\n\n[Branching Requirement]\n"
                    "You MUST produce at least 3 branches: parameter increases / decreases / "
                    "stable equilibrium. Each branch must have a distinct mechanism."
                )

            sys_msg = build_system_prompt_v2() if use_v2 else build_system_prompt(mode)
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    max_tokens=MAX_TOKENS,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user",   "content": user_msg},
                    ],
                )
                expansion = resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"[RAL] API error on '{name[:50]}': {e}")
                time.sleep(3)
                continue

            token_count = len(expansion.split())

            # ---- Parse (v2 or v1) ----
            if use_v2:
                parsed, parse_error = parse_ral_output_v2(expansion)
            else:
                parsed, parse_error = parse_ral_output(expansion)

            # Single retry on JSON failure — appending a stricter instruction recovers ~20-30%
            if parse_error in ("json_parse_fail", "missing_keys", "schema_validation_fail"):
                try:
                    resp2 = client.chat.completions.create(
                        model=args.model,
                        max_tokens=MAX_TOKENS,
                        messages=[
                            {"role": "system", "content": sys_msg},
                            {"role": "user",   "content": user_msg + "\n\nReturn ONLY valid JSON. No other text."},
                        ],
                    )
                    expansion2 = resp2.choices[0].message.content.strip()
                    if use_v2:
                        parsed2, error2 = parse_ral_output_v2(expansion2)
                    else:
                        parsed2, error2 = parse_ral_output(expansion2)
                    if not error2:
                        expansion    = expansion2
                        token_count  = len(expansion.split())
                        parsed       = parsed2
                        parse_error  = None
                        stats["retries_recovered"] = stats.get("retries_recovered", 0) + 1
                except Exception:
                    pass

            # ---- Evaluate + auto-regen ----
            eval_result  = None
            topology     = None

            if not parse_error and not args.dry_run:
                if use_v2:
                    # Structural topology evaluation (Phase 2) — replaces lexical check for v2.
                    topology = evaluate_branch_topology(parsed)
                    if not topology["pass"]:
                        stats["regen_attempted"] = stats.get("regen_attempted", 0) + 1
                        try:
                            regen_msg = (
                                user_msg
                                + f"\n\n[REGENERATION: {topology['reason']}]\n"
                                "Your previous branches were rejected. Requirements:\n"
                                "- At least 2 branches with GENUINELY DIFFERENT mechanisms\n"
                                "- Every branch must have a non-empty failure_boundary\n"
                                "- No two branches may share >70% word overlap in their mechanism fields\n"
                                "Return ONLY valid JSON matching the schema."
                            )
                            resp_r = client.chat.completions.create(
                                model=args.model,
                                max_tokens=MAX_TOKENS,
                                messages=[
                                    {"role": "system", "content": sys_msg},
                                    {"role": "user",   "content": regen_msg},
                                ],
                            )
                            parsed_r, error_r = parse_ral_output_v2(resp_r.choices[0].message.content.strip())
                            if not error_r:
                                new_topo = evaluate_branch_topology(parsed_r)
                                if new_topo["pass"]:
                                    parsed   = parsed_r
                                    topology = new_topo
                                    stats["regen_success"] = stats.get("regen_success", 0) + 1
                                else:
                                    stats["regen_failed"] = stats.get("regen_failed", 0) + 1
                            else:
                                stats["regen_failed"] = stats.get("regen_failed", 0) + 1
                        except Exception:
                            stats["regen_failed"] = stats.get("regen_failed", 0) + 1

                        # Re-check parse_error after regen attempt
                        if not topology["pass"]:
                            parse_error = topology["reason"]

                else:
                    # v1: lexical structure evaluation
                    eval_result = evaluate_reasoning_structure(parsed["reasoning_trace"])

                    is_weak_contrastive = (
                        is_contrastive
                        and eval_result["structure_type"] == "linear"
                        and not eval_result["has_failure"]
                        and not eval_result["has_branching"]
                        and eval_result["diversity_score"] < 0.55
                    )
                    is_weak_standard = (
                        not is_contrastive
                        and eval_result["diversity_score"] < 0.40
                        and eval_result["word_count"] < 60
                        and eval_result["structure_type"] == "linear"
                    )

                    if is_weak_contrastive or is_weak_standard:
                        stats["regen_attempted"] = stats.get("regen_attempted", 0) + 1
                        try:
                            regen_msg = (
                                user_msg
                                + "\n\n[REGENERATION REQUIREMENT]\n"
                                "Your previous answer lacked reasoning depth. You MUST include:\n"
                                "- at least one failure mode OR\n"
                                "- an alternative mechanism OR\n"
                                "- a conditional pathway (if X changes, Y changes)\n"
                                "Avoid purely linear cause-effect chains. Return ONLY valid JSON."
                            )
                            resp_r = client.chat.completions.create(
                                model=args.model,
                                max_tokens=MAX_TOKENS,
                                messages=[
                                    {"role": "system", "content": sys_msg},
                                    {"role": "user",   "content": regen_msg},
                                ],
                            )
                            parsed_r, error_r = parse_ral_output(resp_r.choices[0].message.content.strip())
                            if not error_r:
                                new_eval = evaluate_reasoning_structure(parsed_r["reasoning_trace"])
                                if new_eval["diversity_score"] > eval_result["diversity_score"]:
                                    parsed      = parsed_r
                                    eval_result = new_eval
                                    stats["regen_success"] = stats.get("regen_success", 0) + 1
                                else:
                                    stats["regen_failed"] = stats.get("regen_failed", 0) + 1
                            else:
                                stats["regen_failed"] = stats.get("regen_failed", 0) + 1
                        except Exception:
                            stats["regen_failed"] = stats.get("regen_failed", 0) + 1

            elif not parse_error and args.dry_run and not use_v2:
                eval_result = evaluate_reasoning_structure(parsed["reasoning_trace"])

            # ---- Dry-run output ----
            if args.dry_run:
                label = "CONTRASTIVE" if is_contrastive else mode.upper()
                schema_tag = "[v2]" if use_v2 else "[v1]"
                print(f"--- {schema_tag} [{label}] {name[:60]} ---")
                print(f"ORIGINAL ({len(content.split())}w): {content[:150]}")
                if parse_error:
                    print(f"GATE FAIL: {parse_error}")
                    print(f"RAW: {expansion[:300]}")
                elif use_v2:
                    print(f"BRANCHES ({len(parsed.branches)}):")
                    for b in parsed.branches:
                        print(f"  [{b.branch_id}] {b.mechanism[:120]}")
                    print(f"ANSWER: {parsed.final_answer[:200]}")
                    if topology:
                        print(f"TOPOLOGY: {topology}")
                else:
                    print(f"TRACE ({len(parsed['reasoning_trace'])} steps) [{eval_result['structure_type']} / div={eval_result['diversity_score']}]:")
                    for s in parsed["reasoning_trace"]:
                        print(f"  - {s}")
                    print(f"ANSWER: {parsed['final_answer'][:200]}")
                print()

            # ---- Write or reject ----
            elif parse_error:
                log_rejection(node, parse_error, token_count)
            else:
                if use_v2:
                    record = {
                        **node,
                        "schema_version":            "2.0",
                        "branch_trace":              parsed.model_dump(),
                        "final_answer":              parsed.final_answer,
                        "output":                    parsed.final_answer,
                        "_ral_mode":                 mode,
                        "reasoning_type":            "contrastive" if is_contrastive else "standard",
                        "branch_count":              topology["branch_count"] if topology else len(parsed.branches),
                        "failure_boundary_coverage": topology["failure_boundary_coverage"] if topology else 0.0,
                        "max_mechanism_similarity":  topology["max_mechanism_similarity"] if topology else 0.0,
                        "condition_coverage":        topology["condition_coverage"] if topology else 0.0,
                    }
                else:
                    record = {
                        **node,
                        "reasoning_trace": parsed["reasoning_trace"],
                        "final_answer":    parsed["final_answer"],
                        "output":          parsed["final_answer"],
                        "_ral_mode":       mode,
                        "reasoning_type":  "contrastive" if is_contrastive else "standard",
                        "structure_type":  eval_result["structure_type"] if eval_result else "unknown",
                        "diversity_score": eval_result["diversity_score"] if eval_result else 0.0,
                    }
                out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                state["expanded"] += 1
                stats["expanded"] += 1
                if is_contrastive:
                    stats["contrastive"] += 1
                else:
                    stats["standard"] += 1
                if args.debug_sample_output and stats["expanded"] <= 10:
                    print(f"\n[ACCEPTED SAMPLE]")
                    print(f"  Name   : {name[:80]}")
                    print(f"  Schema : {'v2' if use_v2 else 'v1'}")
                    print(f"  Type   : {'contrastive' if is_contrastive else 'standard'}")
                    if use_v2:
                        print(f"  Branches: {len(parsed.branches)}")
                    else:
                        print(f"  Steps  : {len(parsed['reasoning_trace'])}")
                    print(f"  Answer : {parsed.final_answer[:200] if use_v2 else parsed['final_answer'][:200]}")

            done_set.add(uid)
            state["done"] = list(done_set)

            if (i + 1) % 100 == 0 and not args.dry_run:
                out_handle.flush()
                save_checkpoint(state)
                total_rej   = sum(r["count"] for r in stats["rejections"].values())
                total_so_far = stats["expanded"] + total_rej
                rr = total_rej / max(total_so_far, 1) * 100
                print(f"[RAL] {i+1}/{len(valid)} | expanded={stats['expanded']} rejected={total_rej} ({rr:.1f}%) skipped={state['skipped']}")

            time.sleep(RATE_SLEEP_SEC)

    finally:
        if out_handle:
            out_handle.close()
        if rejected_fp:
            rejected_fp.close()
        if not args.dry_run:
            save_checkpoint(state)

    # ---- Final aggregation report ----
    if args.dry_run:
        print("[RAL] Dry run complete — no files written.")
    else:
        total_rej  = sum(r["count"] for r in stats["rejections"].values())
        total_atts = stats["expanded"] + total_rej
        print(f"\n{'='*50}")
        print("[RAL PIPELINE COMPLETE] AGGREGATION REPORT")
        print(f"{'='*50}")
        print(f"Total Processed : {total_atts}")
        print(f"Expanded        : {stats['expanded']}")
        exp = max(stats["expanded"], 1)
        print(f"  Contrastive   : {stats['contrastive']}  ({stats['contrastive'] / exp * 100:.1f}%)")
        print(f"  Standard      : {stats['standard']}  ({stats['standard'] / exp * 100:.1f}%)")
        recovered = stats.get("retries_recovered", 0)
        if recovered:
            print(f"  Retry saves   : {recovered}  (JSON retry recoveries)")
        attempted = stats.get("regen_attempted", 0)
        if attempted:
            print(f"  Regen attempted: {attempted}")
            print(f"  Regen success  : {stats.get('regen_success', 0)}")
            print(f"  Regen failed   : {stats.get('regen_failed', 0)}")
        if stats["rejections"]:
            print(f"\nFAILURE HEATMAP:")
            for reas, data in sorted(stats["rejections"].items(), key=lambda x: -x[1]["count"]):
                pct     = data["count"] / max(total_atts, 1) * 100
                avg_tok = data["tokens"] / max(data["count"], 1)
                print(f"  {reas.upper():<32} {pct:5.1f}%  ({data['count']} nodes, avg {avg_tok:.1f} tokens)")
        summary_path = f"{OUTPUT_DIR}/rejected_summary.json"
        with open(summary_path, "w", encoding="utf-8") as _sf:
            json.dump({
                "total_rejected":    total_rej,
                "by_reason":         {r: d["count"] for r, d in stats["rejections"].items()},
                "top_rejected_nodes": stats["top_rejected"],
            }, _sf, indent=2, ensure_ascii=False)

        run_summary_path = f"{OUTPUT_DIR}/run_summary_v2.json"
        with open(run_summary_path, "w", encoding="utf-8") as _rf:
            json.dump({
                "total_nodes":        len(nodes),
                "after_prefilter":    len(valid),
                "successful":         stats["expanded"],
                "rejected":           total_rej,
                "contrastive":        stats["contrastive"],
                "contrastive_ratio":  round(stats["contrastive"] / max(stats["expanded"], 1), 4),
                "retry_recovered":    stats.get("retries_recovered", 0),
                "json_failure_rate":  round(stats["rejections"].get("json_parse_fail", {}).get("count", 0) / max(total_atts, 1), 4),
                "failure_breakdown":  {r: d["count"] for r, d in stats["rejections"].items()},
                "regen_attempted":    stats.get("regen_attempted", 0),
                "regen_success":      stats.get("regen_success", 0),
                "regen_failed":       stats.get("regen_failed", 0),
            }, _rf, indent=2, ensure_ascii=False)
        print(f"[RAL] Run summary    : {run_summary_path}")
        print(f"\n[RAL] Output        : {args.output}")
        print(f"[RAL] Rejected log  : {OUTPUT_DIR}/rejected_theorist.jsonl")
        print(f"[RAL] Rejection summary: {summary_path}")
        print(f"\nNext step:")
        print(f"  python -m src.pipeline.sft_formatter {OUTPUT_DIR} \\")
        print(f"    -o output/datasets/combined_theorist_v2.jsonl --apply-loss-masking")


if __name__ == "__main__":
    main()
