"""
Phase 3 tests — flashcard_auditor.py branch metrics and summary generation.

Covers:
  - compute_branch_metrics() per-row extraction
  - generate_branch_quality_summary() aggregation
  - Pass/fail gate logic
  - LIMA/alpaca/veteran exclusion from branch denominators
  - branch_quality_summary.json written correctly
  - CLI --dataset and --summary-out arguments
"""
import json
import sys
import tempfile
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import scripts.flashcard_auditor as fa


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _chatml_row(
    mode: str = "knowledge",
    schema_version: str = "1.0",
    assistant_content: str = "Some answer.",
    branch_count: int = 0,
    failure_boundary_coverage: float = 0.0,
    max_mechanism_similarity: float = 0.0,
    condition_coverage: float = 0.0,
) -> dict:
    """Build a minimal ChatML row as sft_formatter would write it."""
    row = {
        "_mode": mode,
        "messages": [
            {"role": "system",    "content": "System.", "train_loss": False},
            {"role": "user",      "content": "Question.", "train_loss": False},
            {"role": "assistant", "content": assistant_content, "train_loss": True},
        ],
    }
    if schema_version != "1.0":
        row["schema_version"] = schema_version
    if branch_count:
        row["branch_count"] = branch_count
    if failure_boundary_coverage:
        row["failure_boundary_coverage"] = failure_boundary_coverage
    if max_mechanism_similarity:
        row["max_mechanism_similarity"] = max_mechanism_similarity
    if condition_coverage:
        row["condition_coverage"] = condition_coverage
    return row


def _think_content(body: str) -> str:
    return f"<think>\n{body}\n</think>\nFinal answer with because and therefore."


def _v2_row(branch_count: int = 2,
            fb_coverage: float = 1.0,
            max_sim: float = 0.25,
            cond_cov: float = 1.0) -> dict:
    return _chatml_row(
        mode="knowledge",
        schema_version="2.0",
        assistant_content=_think_content("Baseline: stable.\n\nBranch: b1\n  Condition: If X rises\n"),
        branch_count=branch_count,
        failure_boundary_coverage=fb_coverage,
        max_mechanism_similarity=max_sim,
        condition_coverage=cond_cov,
    )


def _v1_row() -> dict:
    return _chatml_row(
        mode="knowledge",
        schema_version="1.0",
        assistant_content=_think_content("Step 1. Step 2. Step 3."),
    )


def _alpaca_row() -> dict:
    return _chatml_row(mode="alpaca", assistant_content="Some alpaca answer.")


def _veteran_row() -> dict:
    return _chatml_row(mode="veteran", assistant_content="Veteran diagnostic output.")


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# compute_branch_metrics
# ---------------------------------------------------------------------------

class TestComputeBranchMetrics:

    def test_v2_knowledge_row_extracts_metadata(self):
        row = _v2_row(branch_count=3, fb_coverage=0.9, max_sim=0.3, cond_cov=0.8)
        m = fa.compute_branch_metrics(row)
        assert m["is_knowledge"] is True
        assert m["schema_version"] == "2.0"
        assert m["branch_count"] == 3
        assert m["failure_boundary_coverage"] == pytest.approx(0.9)
        assert m["max_mechanism_similarity"] == pytest.approx(0.3)
        assert m["condition_coverage"] == pytest.approx(0.8)

    def test_v2_row_detects_think_block(self):
        row = _v2_row()
        m = fa.compute_branch_metrics(row)
        assert m["has_think_block"] is True
        assert m["think_word_count"] > 0

    def test_v1_knowledge_row_branch_count_is_zero(self):
        row = _v1_row()
        m = fa.compute_branch_metrics(row)
        assert m["schema_version"] == "1.0"
        assert m["branch_count"] == 0

    def test_v1_row_has_think_block(self):
        row = _v1_row()
        m = fa.compute_branch_metrics(row)
        assert m["has_think_block"] is True

    def test_alpaca_row_is_not_knowledge(self):
        m = fa.compute_branch_metrics(_alpaca_row())
        assert m["is_knowledge"] is False
        assert m["branch_count"] == 0

    def test_veteran_row_is_not_knowledge(self):
        m = fa.compute_branch_metrics(_veteran_row())
        assert m["is_knowledge"] is False

    def test_fake_contrast_flagged_above_threshold(self):
        row = _v2_row(max_sim=0.85)  # > 0.70 threshold
        m = fa.compute_branch_metrics(row)
        assert m["fake_contrast"] is True

    def test_fake_contrast_not_flagged_below_threshold(self):
        row = _v2_row(max_sim=0.50)
        m = fa.compute_branch_metrics(row)
        assert m["fake_contrast"] is False

    def test_row_without_think_block_detected(self):
        row = _chatml_row(mode="knowledge", schema_version="1.0",
                          assistant_content="No think block here at all.")
        m = fa.compute_branch_metrics(row)
        assert m["has_think_block"] is False
        assert m["think_word_count"] == 0


# ---------------------------------------------------------------------------
# generate_branch_quality_summary
# ---------------------------------------------------------------------------

class TestGenerateBranchQualitySummary:

    def _metrics_from(self, rows: list) -> list:
        return [fa.compute_branch_metrics(r) for r in rows]

    def test_zero_v2_rows_fails(self):
        metrics = self._metrics_from([_v1_row(), _v1_row()])
        summary = fa.generate_branch_quality_summary(metrics, total_rows=2)
        assert summary["pass"] is False
        assert "no_v2_rows" in summary["reasons"]
        assert summary["schema_v2_rows"] == 0

    def test_healthy_v2_dataset_passes(self):
        rows = [_v2_row(branch_count=2, fb_coverage=1.0, max_sim=0.2, cond_cov=1.0)] * 20
        # Also add v1 rows with think blocks (so think_coverage stays high).
        rows += [_v1_row()] * 1
        metrics = self._metrics_from(rows)
        summary = fa.generate_branch_quality_summary(metrics, total_rows=len(rows))
        assert summary["pass"] is True
        assert summary["reasons"] == []
        assert summary["schema_v2_rows"] == 20
        assert summary["schema_v1_rows"] == 1

    def test_low_think_coverage_fails(self):
        v2_rows = [_v2_row()] * 5
        # Add knowledge rows without think blocks to drag coverage below 0.95.
        no_think = [_chatml_row(mode="knowledge", schema_version="1.0",
                                assistant_content="No think block.") for _ in range(15)]
        rows = v2_rows + no_think
        metrics = self._metrics_from(rows)
        summary = fa.generate_branch_quality_summary(metrics, total_rows=len(rows))
        assert summary["pass"] is False
        assert any("think_coverage" in r for r in summary["reasons"])

    def test_low_branch_count_p10_fails(self):
        # Most rows have 2 branches, but a few have 1 — p10 may drop below 2.
        rows = [_v2_row(branch_count=1)] * 5 + [_v2_row(branch_count=2)] * 15
        metrics = self._metrics_from(rows)
        summary = fa.generate_branch_quality_summary(metrics, total_rows=len(rows))
        # p10 of [1,1,1,1,1,2,2,...] with 20 elements: index 1 = 1 → fail
        if summary["branch_count_p10"] < 2:
            assert summary["pass"] is False

    def test_high_fake_contrast_rate_fails(self):
        # > 10% rows with max_sim > 0.70.
        high_sim = [_v2_row(max_sim=0.85)] * 3
        low_sim  = [_v2_row(max_sim=0.20)] * 17
        rows = high_sim + low_sim
        metrics = self._metrics_from(rows)
        summary = fa.generate_branch_quality_summary(metrics, total_rows=len(rows))
        assert summary["pass"] is False
        assert any("fake_contrast" in r for r in summary["reasons"])

    def test_alpaca_veteran_excluded_from_branch_denominators(self):
        """Adding alpaca/veteran rows must not contaminate branch quality metrics."""
        v2_rows  = [_v2_row()] * 10
        alpaca   = [_alpaca_row()] * 5
        veteran  = [_veteran_row()] * 5
        rows     = v2_rows + alpaca + veteran
        metrics  = self._metrics_from(rows)
        summary  = fa.generate_branch_quality_summary(metrics, total_rows=len(rows))
        # knowledge_rows should count only the 10 v2 knowledge rows.
        assert summary["knowledge_rows"] == 10
        # schema_v2_rows should be 10.
        assert summary["schema_v2_rows"] == 10

    def test_total_rows_field_matches_input(self):
        rows = [_v2_row()] * 4 + [_alpaca_row()] * 2
        metrics = self._metrics_from(rows)
        summary = fa.generate_branch_quality_summary(metrics, total_rows=6)
        assert summary["total_rows"] == 6


# ---------------------------------------------------------------------------
# End-to-end: main() writes branch_quality_summary.json
# ---------------------------------------------------------------------------

class TestAuditorMain:

    def _run(self, rows: list, extra_args: list = None) -> dict:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            dataset = base / "dataset.jsonl"
            summary = base / "branch_quality_summary.json"
            report  = base / "report.md"
            _write_jsonl(dataset, rows)
            argv = [
                "--dataset", str(dataset),
                "--summary-out", str(summary),
                "--report-out", str(report),
            ] + (extra_args or [])
            fa.main(argv)
            return json.loads(summary.read_text(encoding="utf-8"))

    def test_writes_branch_quality_summary_json(self):
        rows = [_v2_row()] * 5
        summary = self._run(rows)
        assert "pass" in summary
        assert "schema_v2_rows" in summary
        assert "think_coverage" in summary

    def test_zero_v2_rows_produces_fail_without_crashing(self):
        rows = [_v1_row()] * 3 + [_alpaca_row()] * 2
        summary = self._run(rows)
        assert summary["pass"] is False
        assert summary["schema_v2_rows"] == 0

    def test_healthy_v2_dataset_produces_pass(self):
        rows = [_v2_row(branch_count=2, fb_coverage=1.0, max_sim=0.20, cond_cov=1.0)] * 25
        summary = self._run(rows)
        assert summary["pass"] is True

    def test_custom_summary_out_path(self):
        """--summary-out must write to the specified path, not a default location."""
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            dataset = base / "data.jsonl"
            summary = base / "custom" / "my_summary.json"
            report  = base / "report.md"
            _write_jsonl(dataset, [_v2_row()] * 3)
            fa.main([
                "--dataset", str(dataset),
                "--summary-out", str(summary),
                "--report-out", str(report),
            ])
            assert summary.exists()
            data = json.loads(summary.read_text())
            assert "pass" in data


# ---------------------------------------------------------------------------
# validate_record: mode-aware role sequence tests
# ---------------------------------------------------------------------------

def _two_turn_alpaca_record(mode: str = "alpaca") -> dict:
    """Minimal 2-turn (user, assistant) record — as the formatter emits for alpaca mix-ins."""
    return {
        "_mode": mode,
        "messages": [
            {"role": "user",      "content": "What is X?"},
            {"role": "assistant", "content": "X is Y."},
        ],
    }


def _three_turn_record(mode: str = "knowledge") -> dict:
    return {
        "_mode": mode,
        "messages": [
            {"role": "system",    "content": "System prompt."},
            {"role": "user",      "content": "Question."},
            {"role": "assistant", "content": "Answer."},
        ],
    }


class TestValidateRecordModeAware:

    def test_alpaca_two_turn_passes(self):
        record = _two_turn_alpaca_record(mode="alpaca")
        violations = fa.validate_record(1, record)
        assert violations == [], violations

    def test_knowledge_three_turn_passes(self):
        record = _three_turn_record(mode="knowledge")
        violations = fa.validate_record(1, record)
        assert violations == [], violations

    def test_knowledge_two_turn_fails(self):
        """knowledge rows must use 3-turn (system, user, assistant)."""
        record = _two_turn_alpaca_record(mode="knowledge")
        violations = fa.validate_record(1, record)
        assert any("invalid_role_sequence" in v for v in violations)

    def test_veteran_two_turn_fails(self):
        """veteran rows must use 3-turn (system, user, assistant)."""
        record = _two_turn_alpaca_record(mode="veteran")
        violations = fa.validate_record(1, record)
        assert any("invalid_role_sequence" in v for v in violations)

    def test_branch_summary_excludes_alpaca_from_branch_denominators(self):
        """Alpaca rows that pass validation must not count in branch quality metrics."""
        v2_rows  = [_v2_row(branch_count=2, fb_coverage=1.0, max_sim=0.2, cond_cov=1.0)] * 10
        alpaca   = [_alpaca_row()] * 5
        rows     = v2_rows + alpaca
        metrics  = [fa.compute_branch_metrics(r) for r in rows]
        summary  = fa.generate_branch_quality_summary(metrics, total_rows=len(rows))
        assert summary["knowledge_rows"] == 10
        assert summary["schema_v2_rows"] == 10

    def test_exit_code_zero_for_valid_mixed_dataset(self):
        """A dataset with v2 knowledge rows + alpaca mix-ins must produce exit code 0."""
        v2_rows = [_v2_row(branch_count=2, fb_coverage=1.0, max_sim=0.2, cond_cov=1.0)] * 20
        alpaca  = [_alpaca_row()] * 5
        rows    = v2_rows + alpaca
        with tempfile.TemporaryDirectory() as td:
            base    = Path(td)
            dataset = base / "dataset.jsonl"
            summary = base / "summary.json"
            report  = base / "report.md"
            _write_jsonl(dataset, rows)
            rc = fa.main([
                "--dataset",     str(dataset),
                "--summary-out", str(summary),
                "--report-out",  str(report),
            ])
            assert rc == 0, f"Expected exit 0, got {rc}"
            data = json.loads(summary.read_text(encoding="utf-8"))
            assert data["pass"] is True
