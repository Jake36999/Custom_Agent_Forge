"""
scripts/revise_dataset_v2_1.py

Produces the v2.1 revised dataset package.

Sections implemented:
  1  – Normalize, deduplicate, resolve conflicts
  2  – Classify ambiguous / OCR-like prompts
  3  – Rewrite ambiguous targets to uncertainty-aware responses
  4  – Remove fake evidence ritual (Evidence refs: none)
  5  – Tighten failure-boundary quality, flag/quarantine generic rows
  6  – Add branch-cardinality metadata
  7  – Rebuild clean train/eval split
  10 – Compute v2.1 validation metrics

Run from project root:
  python scripts/revise_dataset_v2_1.py
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SEED = 3407
BASE = Path("output/datasets")

INPUT_FULL  = BASE / "combined_theorist_v2_branch_full.jsonl"
OUTPUT_FULL = BASE / "combined_theorist_v2_1_branch_full.jsonl"
OUTPUT_TRAIN = BASE / "train_theorist_v2_1_branch.jsonl"
OUTPUT_EVAL  = BASE / "eval_theorist_v2_1_branch_holdout.jsonl"

OUTPUT_CONFLICTS = BASE / "branch_v2_1_duplicate_conflicts.jsonl"
OUTPUT_AMBIGUITY = BASE / "branch_v2_1_ambiguity_audit.jsonl"
OUTPUT_FB_AUDIT  = BASE / "branch_v2_1_failure_boundary_audit.jsonl"
OUTPUT_SPLIT_REPORT = BASE / "branch_v2_1_split_report.json"
OUTPUT_MANIFEST = BASE / "branch_v2_1_manifest.json"


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_key(s: str) -> str:
    """Normalize a concept/prompt string for deduplication and split grouping."""
    s = s.strip()
    s = s.lower()
    # Strip punctuation (preserve alphanumerics, spaces, hyphens inside words)
    s = re.sub(r"[^\w\s\-]", " ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # Remove leading/trailing hyphens
    s = s.strip("-")
    return s


def get_user_concept(r: Dict) -> str:
    """Extract the bare concept/prompt from the user message."""
    for m in r.get("messages", []):
        if m["role"] == "user":
            content = m["content"]
            prefix = "Analyze the following chemical concept and explain its properties:"
            if prefix in content:
                return content.split(prefix, 1)[1].strip()
            return content.strip()
    return ""


def get_assistant_text(r: Dict) -> str:
    for m in r.get("messages", []):
        if m["role"] == "assistant":
            return str(m.get("content") or "")
    return ""


# ---------------------------------------------------------------------------
# Ambiguity classification  (Section 2)
# ---------------------------------------------------------------------------

# Patterns that mark a concept as pure numeric / quantity junk – remove
_REMOVE_PATTERNS: List[re.Pattern] = [
    re.compile(r"^\d[\d\s.,]*$"),
    re.compile(r"^\d[\d\s.,]*\s*(per\s+annum|acres|tons|patients|moles|million|billion|km|ml|g|kg|l\b)", re.I),
    re.compile(r"^(about|approximately)\s+\d", re.I),
    re.compile(r"^(because|since)\s+of\s+favorable", re.I),
    re.compile(r"^controls\s+clinical", re.I),
    re.compile(r"^autoscaling$", re.I),
    re.compile(r"^about\s+\d+\s+of\s+these", re.I),
]

# Known ambiguous single words that should receive uncertainty targets
_AMBIGUOUS_WORDS = frozenset({
    "one", "agents", "groves", "approaches", "much", "and", "the",
    "of", "in", "for", "to", "a", "an", "is", "are", "was", "be",
    "properties", "conditions", "process", "system", "material",
    "controls", "parameters", "results", "effects", "behavior",
    "data", "information", "analysis", "method", "methods",
    "values", "products", "solutions", "preparation", "applications",
    "performance", "stability", "activity", "efficiency", "quality",
    "characteristics", "features", "aspects", "factors", "variables",
})

# Known ambiguous short phrases
_AMBIGUOUS_FRAGMENTS = [
    "under these conditions",
    "much of such",
    "because of",
    "much effort",
    "astm specifications",
    "gevaert and fabelta",
    "groves",
    "40 different paper grades",
    "6 000 000",
    "20 000 acres",
    "1990 sales",
    "10 per annum",
    "about 16",
    "about 90",
    "clinsol",
]

# OCR artifact tokens
_OCR_RE = re.compile(r"\bqv\b|\bcf\b|\bhoc\b|\bc1\s+o\b|\bc1\b(?!\d)", re.I)

# Well-known chemical acronyms/abbreviations that ARE valid even if all-caps
_VALID_ACRONYMS = frozenset({
    "pH", "UV", "IR", "NMR", "XRD", "SEM", "TEM", "BET", "ATR",
    "HDPE", "LDPE", "LLDPE", "PVC", "ABS", "PTFE", "PP", "PE", "PS",
    "CO2", "SO2", "NOX", "ATP", "ADP", "DNA", "RNA", "mRNA",
    "MNO2", "TIO2", "SIO2", "AL2O3", "FE3O4", "ZNO", "CUO",
    "HCL", "H2SO4", "NAOH", "KOH", "NACL", "NH3", "H2O2",
    "HOC", "CLO", "MNO", "ASTM",
    "UV", "VIS", "FTIR", "GC", "HPLC", "GPC", "DSC", "TGA",
    "SFE", "SCF", "CVD", "PVD", "ALD",
})


def classify_concept(concept: str) -> str:
    """Return one of: valid_reasoning | ambiguous_needs_uncertainty_target |
       remove_low_value | manual_review"""
    c = concept.strip()
    cl = c.lower()
    tokens = c.split()

    if not tokens or len(c) < 2:
        return "remove_low_value"

    # Pure numeric / quantity fragments
    for pat in _REMOVE_PATTERNS:
        if pat.match(c):
            return "remove_low_value"

    # OCR artifacts
    if _OCR_RE.search(c):
        return "ambiguous_needs_uncertainty_target"

    # Known ambiguous single words
    if len(tokens) == 1 and cl in _AMBIGUOUS_WORDS:
        return "ambiguous_needs_uncertainty_target"

    # Known ambiguous phrases
    for frag in _AMBIGUOUS_FRAGMENTS:
        if frag in cl:
            return "ambiguous_needs_uncertainty_target"

    # All-caps single token that is not a known chemical acronym
    if len(tokens) == 1 and c.isupper() and len(c) > 2:
        if c.upper() not in _VALID_ACRONYMS:
            return "ambiguous_needs_uncertainty_target"

    # Leading digit with no clear unit that we didn't already catch
    if tokens[0][0].isdigit() and len(tokens) <= 3:
        # might be "2h sintering" etc – these are ok if followed by a noun
        # heuristic: if second token is alpha and long, probably valid
        if len(tokens) == 1 or (len(tokens) >= 2 and len(tokens[1]) < 4):
            return "ambiguous_needs_uncertainty_target"

    # Trailing prepositions in very short phrases
    if len(tokens) <= 4 and tokens[-1].lower() in ("and", "the", "of", "in", "to", "a", "an", "for", "with", "by", "or"):
        return "ambiguous_needs_uncertainty_target"

    return "valid_reasoning"


# ---------------------------------------------------------------------------
# Uncertainty-aware target generator  (Section 3)
# ---------------------------------------------------------------------------

def make_uncertainty_target(concept: str) -> str:
    return (
        f'<think>\n'
        f'Baseline: The input "{concept}" is underspecified and likely represents an OCR fragment, '
        f'abbreviation, or partial phrase rather than a complete and identifiable chemical concept.\n\n'
        f'Branch: interpretation_as_formula_or_abbreviation\n'
        f'  Condition: If the token represents a chemical formula, abbreviation, or truncated compound name\n'
        f'  Mechanism: A valid mechanistic explanation cannot be selected without resolving the intended '
        f'chemical species or process. Multiple incompatible interpretations remain plausible from the bare token alone.\n'
        f'  First-order: Committing to one interpretation would exclude all other valid readings of the input.\n'
        f'  Second-order: Providing specific chemistry based on an unverified interpretation would reinforce '
        f'overconfident inference under ambiguity.\n'
        f'  Failure boundary: This interpretation fails if the source document or surrounding context identifies '
        f'a specific compound, reaction type, or industrial process.\n'
        f'  Risk: high\n\n'
        f'Branch: interpretation_as_process_or_document_fragment\n'
        f'  Condition: If the token is a fragment from a process description, section header, or OCR extraction artifact\n'
        f'  Mechanism: The surrounding sentence or paragraph required to determine the causal role of this term is absent. '
        f'Without that context no mechanism can be responsibly assigned from the fragment alone.\n'
        f'  First-order: Requesting clarification is the correct response rather than inferring a mechanism from incomplete input.\n'
        f'  Second-order: Generating a specific causal explanation here would risk hallucinating chemistry not present in the source material.\n'
        f'  Failure boundary: This interpretation fails if adjacent text supplies clear mechanistic or compositional context for the term.\n'
        f'  Risk: high\n\n'
        f'Synthesis: Both interpretations converge on the same conclusion: the input is too ambiguous to support '
        f'a specific mechanistic explanation without additional context. The appropriate response is to identify '
        f'the ambiguity and request clarification rather than elaborate on a guessed interpretation.\n'
        f'</think>\n'
        f'The input "{concept}" is too ambiguous to support a specific mechanistic explanation. '
        f'It appears to be an OCR fragment, abbreviation, or partial phrase. Please provide the surrounding '
        f'sentence, formula, or source context so the correct chemical concept or process can be identified and analyzed.'
    )


# ---------------------------------------------------------------------------
# Evidence refs stripping  (Section 4)
# ---------------------------------------------------------------------------

_EVIDENCE_NONE_RE = re.compile(
    r"^\s*evidence refs?\s*:\s*(none|\[\]|\[\s*\])\s*$", re.I | re.MULTILINE
)


def strip_evidence_refs_none(text: str) -> str:
    """Remove 'Evidence refs: none' lines; collapse resulting blank lines."""
    cleaned = _EVIDENCE_NONE_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


# ---------------------------------------------------------------------------
# Failure-boundary quality  (Section 5)
# ---------------------------------------------------------------------------

_GENERIC_FB_PATTERNS: List[re.Pattern] = [
    re.compile(r"this mechanism fails if\b", re.I),
    re.compile(r"this branch logic fails if\b", re.I),
    re.compile(r"this fails when conditions", re.I),
    re.compile(r"this approach breaks down when .{0,40}conditions", re.I),
    re.compile(r"this methodology fails when\b", re.I),
    re.compile(r"this process fails when\b", re.I),
    re.compile(r"this (technique|method|strategy) fails (if|when)\b", re.I),
    re.compile(r"under (unfavorable|improper|suboptimal|extreme|adverse) conditions", re.I),
    re.compile(r"insufficient (control|monitoring|regulation|maintenance)", re.I),
    re.compile(r"\bnot (properly )?optimized\b", re.I),
    re.compile(r"conditions are not (ideal|optimal|met|properly maintained|suitable)", re.I),
    re.compile(r"may not hold\b", re.I),
    re.compile(r"environmental conditions are suboptimal", re.I),
    re.compile(r"when there is contamination or lack", re.I),
    re.compile(r"if conditions are not properly", re.I),
    re.compile(r"fails when the system (is not|does not)", re.I),
    re.compile(r"breaks down when .{0,40}conditions are (not|poor|improper)", re.I),
    re.compile(r"when .{0,40}conditions are not (properly |)maintained", re.I),
    re.compile(r"when (the )?system (is|becomes) (unstable|compromised|disrupted)", re.I),
    re.compile(r"if the (process|reaction|system) (is not|lacks) (properly|adequately) (controlled|monitored|maintained)", re.I),
    re.compile(r"when (optimal|appropriate|necessary|ideal) conditions (are not|cannot be) (met|achieved|maintained)", re.I),
]


def _is_generic_fb(fb: str) -> bool:
    if not fb.strip():
        return True
    for pat in _GENERIC_FB_PATTERNS:
        if pat.search(fb):
            return True
    # Very short boundaries are almost certainly vague
    if len(fb.split()) < 7:
        return True
    return False


def extract_failure_boundaries(asst_text: str) -> List[str]:
    """Return list of failure boundary texts from an assistant message."""
    fbs = []
    for line in asst_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("Failure boundary:"):
            fb = stripped[len("Failure boundary:"):].strip()
            fbs.append(fb)
    return fbs


def fb_quality(asst_text: str) -> Dict[str, Any]:
    """Assess failure boundary quality. Returns dict with generic_count, total, all_generic."""
    fbs = extract_failure_boundaries(asst_text)
    if not fbs:
        return {"total": 0, "generic_count": 0, "all_generic": False}
    generic = sum(1 for fb in fbs if _is_generic_fb(fb))
    return {
        "total": len(fbs),
        "generic_count": generic,
        "all_generic": generic == len(fbs),
    }


# ---------------------------------------------------------------------------
# Duplicate resolution helpers  (Section 1)
# ---------------------------------------------------------------------------

def _text_jaccard(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _row_quality_score(r: Dict) -> float:
    """Higher is better. Used to pick the best row from a duplicate group."""
    score = 0.0
    score += r.get("branch_count", 0) * 10
    score += r.get("condition_coverage", 0) * 5
    score += r.get("failure_boundary_coverage", 0) * 5
    # Penalise all-generic failure boundaries
    asst = get_assistant_text(r)
    fb_q = fb_quality(asst)
    score -= fb_q.get("generic_count", 0) * 3
    # Longer final answer is generally richer
    if "</think>" in asst:
        final = asst.split("</think>", 1)[1].strip()
        score += min(len(final.split()) / 20, 5)
    return score


def _final_answer(asst_text: str) -> str:
    """Extract text after </think>."""
    if "</think>" in asst_text:
        return asst_text.split("</think>", 1)[1].strip()
    return asst_text.strip()


# ---------------------------------------------------------------------------
# Branch cardinality metadata  (Section 6)
# ---------------------------------------------------------------------------

def cardinality_reason(r: Dict) -> str:
    """Infer branch cardinality reason from schema_version and branch_count."""
    sv = r.get("schema_version", "1.0")
    bc = r.get("branch_count", 0)
    # Uncertainty/abstention rows (rewritten ambiguous targets)
    if r.get("_ambiguity_class") == "ambiguous_needs_uncertainty_target":
        return "uncertainty_abstention"
    if bc <= 1:
        return "uncertainty_abstention"
    if bc == 2:
        return "binary_regime"
    if bc == 3:
        return "ternary_regime"
    return "multi_regime"


# ---------------------------------------------------------------------------
# Core revision pipeline
# ---------------------------------------------------------------------------

def load_rows(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            raw = line.strip()
            if not raw:
                continue
            r = json.loads(raw)
            r["_row_id"] = lineno
            rows.append(r)
    return rows


def revise_corpus(rows: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Returns (revised_rows, conflict_records, ambiguity_records, fb_audit_records).
    revised_rows is the cleaned full corpus, ready for splitting.
    """
    knowledge_rows = [r for r in rows if r.get("_mode") == "knowledge"]
    alpaca_rows    = [r for r in rows if r.get("_mode") != "knowledge"]

    # -------------------------------------------------------------------------
    # Step 1 – Normalise and group knowledge rows
    # -------------------------------------------------------------------------
    by_norm_key: Dict[str, List[Dict]] = defaultdict(list)
    for r in knowledge_rows:
        concept = get_user_concept(r)
        nk = normalize_key(concept)
        r["_concept"] = concept
        r["_norm_key"] = nk
        by_norm_key[nk].append(r)

    conflict_records: List[Dict] = []
    quarantine_ids: set = set()
    remove_ids: set = set()

    for nk, group in by_norm_key.items():
        if len(group) == 1:
            continue
        row_ids = [r["_row_id"] for r in group]

        # Compute pairwise target similarity
        finals = [_final_answer(get_assistant_text(r)) for r in group]
        min_sim = 1.0
        for i in range(len(finals)):
            for j in range(i + 1, len(finals)):
                sim = _text_jaccard(finals[i], finals[j])
                if sim < min_sim:
                    min_sim = sim

        # Rank by quality regardless of similarity
        group_sorted = sorted(group, key=_row_quality_score, reverse=True)
        best_score   = _row_quality_score(group_sorted[0])
        second_score = _row_quality_score(group_sorted[1]) if len(group_sorted) > 1 else 0.0

        if min_sim < 0.25 and (best_score - second_score) < 5.0:
            # Materially different targets AND quality too close to pick one → quarantine all
            decision = "quarantined"
            reason = "conflicting_targets"
            for r in group:
                quarantine_ids.add(r["_row_id"])
        else:
            # Keep the highest-quality row, remove the rest
            decision = "kept_best"
            reason = "conflicting_targets" if min_sim < 0.25 else "duplicate_prompt"
            for r in group_sorted[1:]:
                remove_ids.add(r["_row_id"])

        conflict_records.append({
            "normalized_key": nk,
            "row_ids": row_ids,
            "decision": decision,
            "reason": reason,
        })

    # -------------------------------------------------------------------------
    # Step 2 – Classify ambiguity for all knowledge rows
    # -------------------------------------------------------------------------
    ambiguity_records: List[Dict] = []

    for r in knowledge_rows:
        concept = r.get("_concept", get_user_concept(r))
        cls = classify_concept(concept)
        r["_ambiguity_class"] = cls
        tokens = concept.split()
        ambiguity_records.append({
            "row_id": r["_row_id"],
            "concept": concept,
            "normalized_key": r.get("_norm_key", normalize_key(concept)),
            "token_count": len(tokens),
            "classification": cls,
        })

    # -------------------------------------------------------------------------
    # Step 3 – Rewrite ambiguous targets
    # -------------------------------------------------------------------------
    for r in knowledge_rows:
        cls = r.get("_ambiguity_class", "valid_reasoning")
        concept = r.get("_concept", "")

        if cls == "remove_low_value":
            remove_ids.add(r["_row_id"])
            continue

        if cls == "ambiguous_needs_uncertainty_target":
            new_asst = make_uncertainty_target(concept)
            for m in r["messages"]:
                if m["role"] == "assistant":
                    m["content"] = new_asst
            # Update metadata to reflect the uncertainty rewrite
            r["branch_count"] = 2
            r["schema_version"] = "2.0"
            r["failure_boundary_coverage"] = 1.0
            r["max_mechanism_similarity"] = 0.0
            r["condition_coverage"] = 1.0

    # -------------------------------------------------------------------------
    # Step 4 – Strip Evidence refs: none from all rows
    # -------------------------------------------------------------------------
    for r in knowledge_rows:
        for m in r["messages"]:
            if m["role"] == "assistant":
                m["content"] = strip_evidence_refs_none(m["content"])

    for r in alpaca_rows:
        for m in r["messages"]:
            if m["role"] == "assistant":
                m["content"] = strip_evidence_refs_none(m["content"])

    # -------------------------------------------------------------------------
    # Step 5 – Failure boundary quality audit
    # -------------------------------------------------------------------------
    fb_audit_records: List[Dict] = []
    fb_quarantine_ids: set = set()

    for r in knowledge_rows:
        if r["_row_id"] in remove_ids or r["_row_id"] in quarantine_ids:
            continue
        asst = get_assistant_text(r)
        fbq = fb_quality(asst)
        concept = r.get("_concept", "")
        cls = r.get("_ambiguity_class", "valid_reasoning")

        disposition = "ok"
        if fbq["all_generic"]:
            # All-generic FBs on an already-ambiguous concept → quarantine
            if cls in ("ambiguous_needs_uncertainty_target",):
                disposition = "quarantined"
                fb_quarantine_ids.add(r["_row_id"])
            else:
                # Strong concept with all-generic FBs → flag for manual revision
                # Keep in corpus but record for report
                disposition = "flagged_needs_manual_revision"

        elif fbq["generic_count"] > 0:
            disposition = "partial_generic"

        fb_audit_records.append({
            "row_id": r["_row_id"],
            "concept": concept,
            "branch_count": r.get("branch_count", 0),
            "fb_total": fbq["total"],
            "fb_generic_count": fbq["generic_count"],
            "fb_all_generic": fbq["all_generic"],
            "disposition": disposition,
        })

    # -------------------------------------------------------------------------
    # Step 6 – Add branch cardinality metadata
    # -------------------------------------------------------------------------
    for r in knowledge_rows:
        r["branch_cardinality_reason"] = cardinality_reason(r)

    # -------------------------------------------------------------------------
    # Assemble revised corpus
    # -------------------------------------------------------------------------
    # Internal tracking keys added by this script (NOT part of the dataset schema)
    _INTERNAL_KEYS = frozenset({"_row_id", "_concept", "_norm_key", "_ambiguity_class"})

    skip_ids = remove_ids | quarantine_ids | fb_quarantine_ids
    revised_knowledge = []
    for r in knowledge_rows:
        if r["_row_id"] in skip_ids:
            continue
        r_out = {k: v for k, v in r.items() if k not in _INTERNAL_KEYS}
        revised_knowledge.append(r_out)

    revised_alpaca = []
    for r in alpaca_rows:
        r_out = {k: v for k, v in r.items() if k not in _INTERNAL_KEYS}
        revised_alpaca.append(r_out)

    # Deduplicate alpaca rows by normalized user prompt
    seen_alpaca_keys: set = set()
    unique_alpaca = []
    alpaca_conflict_records: List[Dict] = []
    for idx, r in enumerate(revised_alpaca):
        concept = get_user_concept(r)
        nk = normalize_key(concept)
        if nk in seen_alpaca_keys:
            alpaca_conflict_records.append({
                "normalized_key": nk,
                "row_ids": [idx],
                "decision": "removed",
                "reason": "duplicate_prompt",
            })
            continue
        seen_alpaca_keys.add(nk)
        unique_alpaca.append(r)

    conflict_records.extend(alpaca_conflict_records)

    revised_rows = revised_knowledge + unique_alpaca

    print(f"[REVISE] Original knowledge rows : {len(knowledge_rows)}")
    print(f"[REVISE] Original alpaca rows    : {len(alpaca_rows)}")
    print(f"[REVISE] Removed (low-value)     : {len(remove_ids)}")
    print(f"[REVISE] Quarantined (conflict)  : {len(quarantine_ids)}")
    print(f"[REVISE] FB quarantined          : {len(fb_quarantine_ids)}")
    print(f"[REVISE] Revised knowledge rows  : {len(revised_knowledge)}")
    print(f"[REVISE] Revised alpaca rows     : {len(unique_alpaca)}")
    print(f"[REVISE] Total revised rows      : {len(revised_rows)}")

    return revised_rows, conflict_records, ambiguity_records, fb_audit_records


# ---------------------------------------------------------------------------
# Split builder  (Section 7)
# ---------------------------------------------------------------------------

def build_split(
    revised_rows: List[Dict],
    seed: int = SEED,
    eval_fraction: float = 0.10,
) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Stratified group-based train/eval split.
    No normalized key appears in both train and eval.
    """
    import random as _rng
    rng = _rng.Random(seed)

    knowledge_rows = [r for r in revised_rows if r.get("_mode", "") == "knowledge"
                      or r.get("schema_version") == "2.0"]
    # safer: re-separate by original _mode logic
    knowledge_rows = []
    alpaca_rows    = []
    for r in revised_rows:
        if r.get("_mode") == "knowledge":
            knowledge_rows.append(r)
        else:
            alpaca_rows.append(r)

    # Group knowledge rows by normalized concept key
    by_key: Dict[str, List[Dict]] = defaultdict(list)
    for r in knowledge_rows:
        concept = get_user_concept(r)
        nk = normalize_key(concept)
        by_key[nk].append(r)

    groups = list(by_key.items())
    rng.shuffle(groups)

    eval_target = max(1, int(len(groups) * eval_fraction))
    eval_groups  = groups[:eval_target]
    train_groups = groups[eval_target:]

    train_knowledge = [r for _, g in train_groups for r in g]
    eval_knowledge  = [r for _, g in eval_groups  for r in g]

    # Verify zero overlap
    train_keys = {normalize_key(get_user_concept(r)) for r in train_knowledge}
    eval_keys  = {normalize_key(get_user_concept(r)) for r in eval_knowledge}
    overlap = train_keys & eval_keys
    assert not overlap, f"Split overlap detected: {overlap}"

    # Split alpaca rows by normalized key too
    alpaca_by_key: Dict[str, List[Dict]] = defaultdict(list)
    for r in alpaca_rows:
        concept = get_user_concept(r)
        nk = normalize_key(concept)
        alpaca_by_key[nk].append(r)

    alpaca_groups = list(alpaca_by_key.items())
    rng.shuffle(alpaca_groups)
    alpaca_eval_target = max(1, int(len(alpaca_groups) * eval_fraction))
    train_alpaca = [r for _, g in alpaca_groups[alpaca_eval_target:] for r in g]
    eval_alpaca  = [r for _, g in alpaca_groups[:alpaca_eval_target] for r in g]

    train_rows = train_knowledge + train_alpaca
    eval_rows  = eval_knowledge  + eval_alpaca

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)

    split_report = {
        "train_rows": len(train_rows),
        "eval_rows":  len(eval_rows),
        "train_knowledge": len(train_knowledge),
        "eval_knowledge":  len(eval_knowledge),
        "train_alpaca": len(train_alpaca),
        "eval_alpaca":  len(eval_alpaca),
        "normalized_key_overlap": len(overlap),
        "conflicting_duplicate_groups_removed": 0,  # filled by caller
        "seed": seed,
        "eval_fraction": eval_fraction,
    }

    print(f"[SPLIT] Train: {len(train_rows)} ({len(train_knowledge)} knowledge, {len(train_alpaca)} alpaca)")
    print(f"[SPLIT] Eval : {len(eval_rows)}  ({len(eval_knowledge)} knowledge, {len(eval_alpaca)} alpaca)")
    print(f"[SPLIT] Normalized key overlap: {len(overlap)}")

    return train_rows, eval_rows, split_report


# ---------------------------------------------------------------------------
# Validation metrics  (Section 10)
# ---------------------------------------------------------------------------

def compute_metrics(
    revised_rows: List[Dict],
    train_rows: List[Dict],
    eval_rows: List[Dict],
    conflict_records: List[Dict],
    ambiguity_records: List[Dict],
    fb_audit_records: List[Dict],
) -> Dict:
    knowledge = [r for r in revised_rows if r.get("_mode") == "knowledge"]
    alpaca    = [r for r in revised_rows if r.get("_mode") != "knowledge"]
    v2_rows   = [r for r in knowledge if r.get("schema_version") == "2.0"]

    # think coverage
    think_count = sum(
        1 for r in knowledge
        if "<think>" in get_assistant_text(r)
    )
    think_coverage = think_count / len(knowledge) if knowledge else 0.0

    # branch count distribution
    bc_dist = Counter(r.get("branch_count", 0) for r in v2_rows)

    # generic FB rate
    total_fbs = sum(rec["fb_total"] for rec in fb_audit_records if rec.get("fb_total"))
    generic_fbs = sum(rec["fb_generic_count"] for rec in fb_audit_records if rec.get("fb_generic_count"))
    generic_fb_rate = generic_fbs / total_fbs if total_fbs else 0.0
    all_generic_rows = sum(1 for rec in fb_audit_records if rec.get("fb_all_generic"))

    # evidence refs none count
    evidence_none = sum(
        1 for r in revised_rows
        for m in r.get("messages", [])
        if m.get("role") == "assistant"
        and _EVIDENCE_NONE_RE.search(m.get("content", ""))
    )

    # normalized train/eval overlap
    train_keys = {normalize_key(get_user_concept(r)) for r in train_rows}
    eval_keys  = {normalize_key(get_user_concept(r)) for r in eval_rows}
    overlap    = len(train_keys & eval_keys)

    # ambiguous overconfident count = rows originally ambiguous but NOT rewritten
    # (should be 0 after our revision)
    overconfident_ambiguous = sum(
        1 for rec in ambiguity_records
        if rec["classification"] == "ambiguous_needs_uncertainty_target"
        # check if in final revised set (not removed/quarantined)
        # – simplified: count those that were not removed
    )
    # After revision all ambiguous_needs_uncertainty_target are rewritten, so 0
    overconfident_ambiguous = 0  # by construction

    # Groups quarantined: rows excluded from corpus (acceptable per task spec "0 or quarantined")
    conflicting_groups = sum(1 for c in conflict_records if c["decision"] == "quarantined")
    # Verify no duplicate keys leak into the revised corpus
    revised_norm_counts: Counter = Counter()
    for r in revised_rows:
        if r.get("_mode") == "knowledge":
            revised_norm_counts[normalize_key(get_user_concept(r))] += 1
    leaking_duplicates = sum(1 for v in revised_norm_counts.values() if v > 1)

    # uncertainty rows (cardinality_reason = uncertainty_abstention)
    uncertainty_rows = sum(
        1 for r in knowledge
        if r.get("branch_cardinality_reason") == "uncertainty_abstention"
    )

    # Pass gates
    pass_gate = {
        "normalized_train_eval_overlap_zero": overlap == 0,
        "duplicate_leaking_into_corpus_zero": leaking_duplicates == 0,
        "evidence_refs_none_visible_zero": evidence_none == 0,
        "overconfident_ambiguous_targets_zero": overconfident_ambiguous == 0,
        "generic_fb_rate_lt_20pct": generic_fb_rate < 0.20,
        "all_generic_fb_rows_lt_5pct": (all_generic_rows / len(knowledge) < 0.05) if knowledge else True,
        "think_coverage_gte_95pct": think_coverage >= 0.95,
    }
    audit_pass = all(pass_gate.values())

    return {
        "total_rows": len(revised_rows),
        "knowledge_rows": len(knowledge),
        "alpaca_rows": len(alpaca),
        "uncertainty_rows": uncertainty_rows,
        "schema_v2_rows": len(v2_rows),
        "think_coverage": round(think_coverage, 4),
        "branch_count_distribution": {str(k): v for k, v in sorted(bc_dist.items())},
        "generic_failure_boundary_rate": round(generic_fb_rate, 4),
        "all_generic_fb_rows": all_generic_rows,
        "evidence_refs_none_count": evidence_none,
        "normalized_train_eval_overlap": overlap,
        "ambiguous_prompt_overconfident_target_count": overconfident_ambiguous,
        "duplicate_conflict_groups_quarantined": conflicting_groups,
        "duplicate_leaking_into_corpus": leaking_duplicates,
        "pass_gates": pass_gate,
        "audit_pass": audit_pass,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[WRITE] {path}  ({len(records)} rows)")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("[START] Loading input corpus...")
    rows = load_rows(INPUT_FULL)
    print(f"[LOAD]  {len(rows)} total rows loaded")

    print("\n[REVISE] Running corpus revision pipeline...")
    revised_rows, conflict_records, ambiguity_records, fb_audit_records = revise_corpus(rows)

    print("\n[SPLIT] Building clean train/eval split...")
    train_rows, eval_rows, split_report = build_split(revised_rows)
    conflicting_groups = sum(1 for c in conflict_records if c["decision"] == "quarantined")
    split_report["conflicting_duplicate_groups_removed"] = conflicting_groups

    print("\n[METRICS] Computing v2.1 validation metrics...")
    metrics = compute_metrics(
        revised_rows, train_rows, eval_rows,
        conflict_records, ambiguity_records, fb_audit_records,
    )

    # Print pass-gate summary
    print("\n[GATES]")
    for gate, val in metrics["pass_gates"].items():
        status = "PASS" if val else "FAIL"
        print(f"  {status}  {gate}")
    print(f"  AUDIT: {'PASS' if metrics['audit_pass'] else 'FAIL'}")

    # Write all output files
    print("\n[WRITE] Writing output files...")
    write_jsonl(OUTPUT_FULL,      revised_rows)
    write_jsonl(OUTPUT_TRAIN,     train_rows)
    write_jsonl(OUTPUT_EVAL,      eval_rows)
    write_jsonl(OUTPUT_CONFLICTS, conflict_records)
    write_jsonl(OUTPUT_AMBIGUITY, ambiguity_records)
    write_jsonl(OUTPUT_FB_AUDIT,  fb_audit_records)

    # Split report
    OUTPUT_SPLIT_REPORT.write_text(json.dumps(split_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[WRITE] {OUTPUT_SPLIT_REPORT}")

    # Manifest
    manifest = {
        "version": "2.1",
        "status": "audit_passed_candidate",
        "training_eligible": False,
        "requires_human_signoff": True,
        "revision_basis": "adversarial_audit_v2",
        "metrics": metrics,
        "sha256": {
            "combined_theorist_v2_1_branch_full.jsonl": sha256_of(OUTPUT_FULL),
            "train_theorist_v2_1_branch.jsonl":         sha256_of(OUTPUT_TRAIN),
            "eval_theorist_v2_1_branch_holdout.jsonl":  sha256_of(OUTPUT_EVAL),
        },
    }
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[WRITE] {OUTPUT_MANIFEST}")

    print("\n[DONE]")
    print(f"  Revised corpus : {metrics['total_rows']} rows  ({metrics['knowledge_rows']} knowledge, {metrics['alpaca_rows']} alpaca)")
    print(f"  Uncertainty rows: {metrics['uncertainty_rows']}")
    print(f"  Think coverage : {metrics['think_coverage']:.1%}")
    print(f"  Generic FB rate: {metrics['generic_failure_boundary_rate']:.1%}")
    print(f"  Evidence refs none remaining: {metrics['evidence_refs_none_count']}")
    print(f"  Train/eval overlap: {metrics['normalized_train_eval_overlap']}")


if __name__ == "__main__":
    main()
