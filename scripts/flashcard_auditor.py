"""Audit strict ChatML flashcards and emit a qualitative sample report and branch quality summary."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SAMPLE_SEED = 42
EXPECTED_ROLES = ("system", "user", "assistant")
_ALPACA_ROLES = ("user", "assistant")

# Modes that must always use the 3-turn system+user+assistant format.
_STRICT_THREE_TURN_MODES = frozenset({"knowledge", "veteran"})

# Pass-gate thresholds for v2 branch-aware datasets.
_THINK_COVERAGE_MIN: float = 0.95
_BRANCH_COUNT_P10_MIN: int = 2
_FB_COVERAGE_MEAN_MIN: float = 0.80
_FAKE_CONTRAST_RATE_MAX: float = 0.10
_FAKE_CONTRAST_SIMILARITY_THRESHOLD: float = 0.70


Record = Dict[str, Any]
IndexedRecord = Tuple[int, Record]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path) -> Iterable[IndexedRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                yield line_no, {"__audit_error__": "empty_line"}
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                yield line_no, {"__audit_error__": f"invalid_json: {exc}"}
                continue
            if not isinstance(payload, dict):
                yield line_no, {"__audit_error__": "row_not_object"}
                continue
            yield line_no, payload


# ---------------------------------------------------------------------------
# Structural validation (unchanged from original)
# ---------------------------------------------------------------------------

def validate_record(line_no: int, record: Record) -> List[str]:
    violations: List[str] = []
    audit_error = record.get("__audit_error__")
    if audit_error:
        return [f"line {line_no}: {audit_error}"]

    messages = record.get("messages")
    if not isinstance(messages, list):
        return [f"line {line_no}: missing_or_invalid_messages_array"]

    roles = tuple(message.get("role") if isinstance(message, dict) else None for message in messages)
    mode = str(record.get("_mode", "")).strip().lower()

    # Build the set of permitted role sequences for this record.
    allowed_roles: set = {EXPECTED_ROLES}

    if mode == "alpaca":
        # Formatter intentionally emits 2-turn (user, assistant) alpaca mix-ins.
        allowed_roles.add(_ALPACA_ROLES)
    elif mode not in _STRICT_THREE_TURN_MODES and mode == "":
        # Unknown/missing mode: allow 2-turn only if the row carries no theorist
        # or knowledge metadata that would indicate it should be 3-turn.
        theorist_keys = {"schema_version", "branch_count", "branch_trace", "reasoning_trace"}
        if not theorist_keys.intersection(record):
            allowed_roles.add(_ALPACA_ROLES)

    if roles not in allowed_roles:
        violations.append(
            f"line {line_no}: invalid_role_sequence {roles!r}; "
            f"expected {EXPECTED_ROLES!r}"
            + (f" or {_ALPACA_ROLES!r}" if _ALPACA_ROLES in allowed_roles else "")
        )

    for index, message in enumerate(messages):
        label = f"line {line_no} message[{index}]"
        if not isinstance(message, dict):
            violations.append(f"{label}: message_not_object")
            continue
        content = message.get("content")
        if not isinstance(content, str):
            violations.append(f"{label}: content_not_string")
        elif content.strip() == "":
            violations.append(f"{label}: empty_content")

    return violations


# ---------------------------------------------------------------------------
# Branch metrics
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _get_think_content(assistant_text: str) -> str:
    """Extract text inside the first <think>...</think> block, or empty string."""
    m = _THINK_RE.search(assistant_text)
    return m.group(1) if m else ""


def compute_branch_metrics(record: Record) -> Dict[str, Any]:
    """Extract structural branch metrics from a formatted SFT ChatML row.

    Operates on the metadata fields preserved by sft_formatter (schema_version,
    branch_count, failure_boundary_coverage, max_mechanism_similarity,
    condition_coverage) rather than re-parsing raw branch JSON.  This makes the
    computation deterministic and avoids carrying full branch payloads in the SFT file.

    Returns a flat dict with all metrics.  Callers must only count rows whose
    _mode == "knowledge" in branch-quality denominators.
    """
    mode           = record.get("_mode", "")
    schema_version = record.get("schema_version", "1.0")
    is_knowledge   = (mode == "knowledge")

    # Locate assistant turn.
    assistant_text = ""
    for msg in record.get("messages") or []:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            assistant_text = str(msg.get("content") or "")
            break

    think_content   = _get_think_content(assistant_text)
    has_think_block = bool(think_content)
    think_word_count = len(think_content.split()) if think_content else 0

    # Branch metrics are only meaningful for v2 knowledge rows.
    if is_knowledge and schema_version == "2.0":
        branch_count              = int(record.get("branch_count") or 0)
        failure_boundary_coverage = float(record.get("failure_boundary_coverage") or 0.0)
        max_mechanism_similarity  = float(record.get("max_mechanism_similarity") or 0.0)
        condition_coverage        = float(record.get("condition_coverage") or 0.0)
        fake_contrast             = max_mechanism_similarity > _FAKE_CONTRAST_SIMILARITY_THRESHOLD
    else:
        branch_count              = 0
        failure_boundary_coverage = 0.0
        max_mechanism_similarity  = 0.0
        condition_coverage        = 0.0
        fake_contrast             = False

    return {
        "is_knowledge":              is_knowledge,
        "schema_version":            schema_version,
        "has_think_block":           has_think_block,
        "think_word_count":          think_word_count,
        "branch_count":              branch_count,
        "failure_boundary_coverage": failure_boundary_coverage,
        "max_mechanism_similarity":  max_mechanism_similarity,
        "condition_coverage":        condition_coverage,
        "fake_contrast":             fake_contrast,
    }


# ---------------------------------------------------------------------------
# Branch quality summary
# ---------------------------------------------------------------------------

def generate_branch_quality_summary(
    all_metrics: List[Dict[str, Any]],
    total_rows: int,
) -> Dict[str, Any]:
    """Aggregate per-row metrics into a dataset-level branch quality summary.

    Branch-quality denominators exclude alpaca/veteran rows — only knowledge rows count.
    Pass gate applies only when schema_v2_rows > 0; a pure v1 dataset sets pass=False
    but does not crash, allowing auditing of baseline datasets.
    """
    knowledge_metrics = [m for m in all_metrics if m["is_knowledge"]]
    v2_metrics        = [m for m in knowledge_metrics if m["schema_version"] == "2.0"]
    v1_metrics        = [m for m in knowledge_metrics if m["schema_version"] != "2.0"]
    knowledge_rows    = len(knowledge_metrics)

    # <think> coverage — over knowledge rows only.
    think_coverage = (
        sum(1 for m in knowledge_metrics if m["has_think_block"]) / knowledge_rows
        if knowledge_rows > 0 else 0.0
    )

    schema_v2_rows = len(v2_metrics)
    schema_v1_rows = len(v1_metrics)

    if v2_metrics:
        branch_counts = sorted(m["branch_count"] for m in v2_metrics)
        n = len(branch_counts)
        branch_count_p10 = branch_counts[max(0, int(n * 0.10) - 1)]
        branch_count_p50 = branch_counts[max(0, int(n * 0.50) - 1)]
        fb_coverage_mean     = statistics.mean(m["failure_boundary_coverage"] for m in v2_metrics)
        condition_cov_mean   = statistics.mean(m["condition_coverage"] for m in v2_metrics)
        fake_contrast_count  = sum(1 for m in v2_metrics if m["fake_contrast"])
        fake_contrast_rate   = fake_contrast_count / schema_v2_rows
    else:
        branch_count_p10     = 0
        branch_count_p50     = 0
        fb_coverage_mean     = 0.0
        condition_cov_mean   = 0.0
        fake_contrast_count  = 0
        fake_contrast_rate   = 0.0

    # Pass gate.
    reasons: List[str] = []
    if schema_v2_rows == 0:
        reasons.append("no_v2_rows")
    if think_coverage < _THINK_COVERAGE_MIN:
        reasons.append(f"think_coverage={think_coverage:.3f} < {_THINK_COVERAGE_MIN}")
    if branch_count_p10 < _BRANCH_COUNT_P10_MIN:
        reasons.append(f"branch_count_p10={branch_count_p10} < {_BRANCH_COUNT_P10_MIN}")
    if fb_coverage_mean < _FB_COVERAGE_MEAN_MIN:
        reasons.append(f"failure_boundary_coverage_mean={fb_coverage_mean:.3f} < {_FB_COVERAGE_MEAN_MIN}")
    if fake_contrast_rate >= _FAKE_CONTRAST_RATE_MAX:
        reasons.append(f"fake_contrast_rate={fake_contrast_rate:.3f} >= {_FAKE_CONTRAST_RATE_MAX}")

    return {
        "total_rows":                    total_rows,
        "knowledge_rows":                knowledge_rows,
        "schema_v1_rows":                schema_v1_rows,
        "schema_v2_rows":                schema_v2_rows,
        "think_coverage":                round(think_coverage, 4),
        "branch_count_p10":              branch_count_p10,
        "branch_count_p50":              branch_count_p50,
        "failure_boundary_coverage_mean": round(fb_coverage_mean, 4),
        "condition_coverage_mean":       round(condition_cov_mean, 4),
        "fake_contrast_flagged":         fake_contrast_count,
        "fake_contrast_rate":            round(fake_contrast_rate, 4),
        "pass":                          len(reasons) == 0,
        "reasons":                       reasons,
    }


# ---------------------------------------------------------------------------
# Report rendering (unchanged structure)
# ---------------------------------------------------------------------------

def _fence_for(text: str) -> str:
    longest = 0
    current = 0
    for char in text:
        if char == "`":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return "`" * max(3, longest + 1)


def _block(title: str, text: str) -> str:
    fence = _fence_for(text)
    return f"### {title}\n\n{fence}text\n{text}\n{fence}\n"


def _render_sample(sample_no: int, label: str, line_no: int, record: Record) -> str:
    messages = record["messages"]
    by_role = {message["role"]: message.get("content", "") for message in messages}
    parts = [
        f"## Sample {sample_no}: {label} Trace\n",
        f"- Source line: `{line_no}`\n",
        _block("System Prompt", by_role["system"]),
        _block("User Prompt", by_role["user"]),
        _block("Assistant Response", by_role["assistant"]),
    ]
    return "\n".join(parts).rstrip() + "\n"


def write_report(
    samples: List[Tuple[str, int, Record]],
    counts: Counter,
    violations: List[str],
    dataset_path: Path,
    report_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# Flashcard Sample Report",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Sample seed: `{SAMPLE_SEED}`",
        f"- Rows: `{sum(counts.values())}`",
        f"- Knowledge rows: `{counts.get('knowledge', 0)}`",
        f"- Veteran rows: `{counts.get('veteran', 0)}`",
        f"- Structural violations: `{len(violations)}`",
        "",
    ]
    body: List[str] = []
    for index, (label, line_no, record) in enumerate(samples, start=1):
        body.append(_render_sample(index, label, line_no, record))
    report_path.write_text("\n".join(header + body), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit Aletheia strict ChatML dataset and emit branch quality summary."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("output/datasets/theorist/qlora_theorist_dataset.jsonl"),
        help="Path to the ChatML JSONL dataset to audit.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("output/flashcard_sample_report.md"),
        help="Path for the qualitative Markdown sample report.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Path for branch_quality_summary.json. "
             "Defaults to <dataset_dir>/branch_quality_summary.json.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    dataset_path = args.dataset
    report_path  = args.report_out
    summary_path = args.summary_out or dataset_path.parent / "branch_quality_summary.json"

    if not dataset_path.exists():
        raise SystemExit(f"[FLASHCARD AUDIT] Dataset not found: {dataset_path}")

    violations: List[str] = []
    counts: Counter = Counter()
    records_by_mode: Dict[str, List[IndexedRecord]] = defaultdict(list)
    all_branch_metrics: List[Dict[str, Any]] = []

    for line_no, record in _iter_jsonl(dataset_path):
        row_violations = validate_record(line_no, record)
        violations.extend(row_violations)
        if row_violations:
            continue

        mode = str(record.get("_mode", "<missing>"))
        counts[mode] += 1
        records_by_mode[mode].append((line_no, record))

        # Collect branch metrics for knowledge rows; skip veteran/alpaca denominators.
        if mode in ("knowledge", "alpaca", "veteran"):
            m = compute_branch_metrics(record)
            all_branch_metrics.append(m)

    total_rows = sum(counts.values())

    # Branch quality summary (excludes alpaca/veteran from denominators inside the function).
    summary = generate_branch_quality_summary(all_branch_metrics, total_rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Qualitative sample report.
    rng = random.Random(SAMPLE_SEED)
    samples: List[Tuple[str, int, Record]] = []
    for mode_key, label in (("knowledge", "Knowledge"), ("veteran", "Veteran")):
        population  = records_by_mode.get(mode_key, [])
        sample_size = min(3, len(population))
        for line_no, record in rng.sample(population, sample_size):
            samples.append((label, line_no, record))

    write_report(samples, counts, violations, dataset_path, report_path)

    status = "PASS" if summary["pass"] else "FAIL"
    print("[FLASHCARD AUDIT] Structural validation complete.")
    print(f"[FLASHCARD AUDIT] Dataset     : {dataset_path}")
    print(f"[FLASHCARD AUDIT] Total rows  : {total_rows}")
    print(f"[FLASHCARD AUDIT] Mode counts : {dict(sorted(counts.items()))}")
    print(f"[FLASHCARD AUDIT] Violations  : {len(violations)}")
    print(f"[FLASHCARD AUDIT] Branch gate : {status}")
    if summary["reasons"]:
        for r in summary["reasons"]:
            print(f"  - {r}")
    print(f"[FLASHCARD AUDIT] Summary     : {summary_path}")
    print(f"[FLASHCARD AUDIT] Report      : {report_path}")
    # Branch-gate failure is also a failing exit so CI cannot proceed on a
    # structurally weak v2 corpus even when no per-row structural violations exist.
    return 1 if violations or not summary["pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
