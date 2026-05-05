"""
SFT Telemetry Gap Regression Tests (GAP-S2 through GAP-S5)
===========================================================

These tests lock the four telemetry/data-provenance guarantees added in
the Priority-2 hardening wave:

    GAP-S2 — get_hash emits SENTINEL-HASH (success) and SENTINEL-HASH-ERR
             (failure), and the success path produces a full SHA256 hex
             digest (64 characters).

    GAP-S3 — format_dataset halts the commit with SENTINEL-DIVERSITY-CRIT
             and returns early when rolling_entropy drops below 3.5 and
             there are more than 10 valid payloads.

    GAP-S4 — Zero-density waveform nodes (sie_node with all-zero waveform
             components) emit SENTINEL-WARN-WAVEFORM *and* receive an
             active Containment Rule demotion: _sample_weight halved
             relative to an identical non-zero-waveform node, plus an
             explicit _waveform_demoted=True provenance tag.

    GAP-S5 — format_rejected_traces emits SENTINEL-DPO-EMPTY not only
             for empty input lists (list-level) but also on every
             per-trace skip caused by a missing 'rejected' or
             'chosen'/'corrected' field. A summary line is also emitted
             when any skips occurred.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import jsonlines
import pytest
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipeline import dataset_formatter as df


# ---------------------------------------------------------------------------
# Shared fixture: a minimal AletheiaSkill-shaped dict that passes the
# survival gate (state=ACCEPTED) and the Pydantic invariant lock.
# ---------------------------------------------------------------------------

def _make_accepted_skill(node_id: str, sie_node: dict = None) -> dict:
    skill = {
        "node_id": node_id,
        "name": node_id,
        "file": "test.py",
        "code_snippet": "def f(): pass",
        "imports": [],
        "operator_type": "function",
        "source_type": "ast_code",
        "skill_type": "execution",
        "semantics": {"code_snippet": "def f(): pass", "name": "f"},
        "teaching_layer": {
            "skill_identity": {"name": node_id},
            "method_metadata": {"name": node_id, "language": "python"},
            "reasoning_vectors": {
                "intent": "compute",
                "strategy": "direct",
                "constraints": ["none"],
                "execution_pattern": ["functional"],
                "failure_modes": ["none"],
            },
            "implementation_template": {"code": "pass"},
        },
        "epistemic": {
            "state": "ACCEPTED",
            "c_node": 0.9,
            "retry_budget": 6,
            "depth": 0,
            "branch_id": "root",
        },
    }
    if sie_node is not None:
        skill["sie_node"] = sie_node
    return skill


def _write_yaml(tmp_path: Path, name: str, skill_list):
    # format_dataset expects YAML or JSON at the top level; a list of skill
    # dicts is accepted and iterated directly.
    path = tmp_path / f"{name}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(skill_list, f)
    return path


# ===========================================================================
# GAP-S2 — get_hash fingerprint telemetry
# ===========================================================================

class TestGapS2FingerprintTelemetry:
    def test_get_hash_success_emits_sentinel_hash(self, capsys):
        result = df.get_hash([{"a": 1}, {"b": 2}], process_id="GAP-S2-OK")
        captured = capsys.readouterr()
        assert "[SENTINEL-HASH]" in captured.out
        assert "GAP-S2-OK" in captured.out

    def test_get_hash_success_produces_full_sha256_hex_digest(self):
        """Guard the fingerprint's length/shape. SHA256 hex = 64 characters."""
        result = df.get_hash([{"a": 1}], process_id="GAP-S2-LEN")
        assert isinstance(result, str)
        assert result != "FINGERPRINT_UNAVAILABLE"
        assert len(result) == 64, (
            f"SHA256 hex digest must be 64 characters, got {len(result)}"
        )
        # Must be valid lowercase hex
        int(result, 16)
        assert result == result.lower()

    def test_get_hash_failure_emits_sentinel_hash_err(self, capsys):
        """Force json.dumps to raise and verify the error sentinel fires."""
        with patch.object(df.json, "dumps", side_effect=TypeError("unserialisable")):
            result = df.get_hash([{"a": 1}], process_id="GAP-S2-ERR")
        captured = capsys.readouterr()
        assert "[SENTINEL-HASH-ERR]" in captured.out
        assert "GAP-S2-ERR" in captured.out
        assert result == "FINGERPRINT_UNAVAILABLE"


# ===========================================================================
# GAP-S3 — entropy abort verification
# ===========================================================================

class TestGapS3EntropyAbort:
    def test_entropy_collapse_halts_export_with_sentinel(self, tmp_path, capsys):
        """
        Build 12 valid ACCEPTED skills (>10, satisfies the valid_payloads
        gate), then patch the rolling entropy calculator to return 1.0
        (well below the 3.5 collapse threshold). The formatter must:
            1. Emit SENTINEL-DIVERSITY-CRIT
            2. Return early without writing the primary output file
        """
        skills = [_make_accepted_skill(f"entropy_node_{i}") for i in range(12)]
        _write_yaml(tmp_path, "entropy_collapse_batch", skills)

        out = tmp_path / "output" / "dataset.jsonl"
        with patch.object(
            df.FIFORollingBuffer,
            "calculate_rolling_entropy",
            return_value=1.0,
        ):
            df.format_dataset(tmp_path, out)

        captured = capsys.readouterr()
        assert "[SENTINEL-DIVERSITY-CRIT]" in captured.out, (
            "entropy collapse must emit SENTINEL-DIVERSITY-CRIT"
        )
        assert "HALTING EXPORT" in captured.out
        # Early return: primary output must NOT be written to disk
        assert not out.exists(), (
            "format_dataset must return early on diversity collapse "
            "and never commit the primary output file"
        )

    def test_normal_entropy_does_not_halt_export(self, tmp_path, capsys):
        """Baseline: above-threshold entropy must write the output file."""
        skills = [_make_accepted_skill(f"entropy_ok_{i}") for i in range(12)]
        _write_yaml(tmp_path, "entropy_ok_batch", skills)

        out = tmp_path / "output" / "dataset.jsonl"
        with patch.object(
            df.FIFORollingBuffer,
            "calculate_rolling_entropy",
            return_value=5.0,  # comfortably above 3.5
        ):
            df.format_dataset(tmp_path, out)

        captured = capsys.readouterr()
        assert "[SENTINEL-DIVERSITY-CRIT]" not in captured.out
        assert out.exists(), "above-threshold entropy must commit output"


# ===========================================================================
# GAP-S4 — waveform fields removed (no longer part of the ontology)
# ===========================================================================

# TestGapS4WaveformDemotion was removed: waveform fields
# (waveform_amplitude, waveform_frequency, waveform_spatial_freq,
# waveform_phase) were deleted from SemanticReasoningNode as part of
# ontological drift eradication.  Pydantic extra="forbid" now correctly
# rejects payloads that carry these legacy fields.


# ===========================================================================
# GAP-S5 — DPO per-trace skip observability
# ===========================================================================

class TestGapS5DpoSkipObservability:
    def test_empty_trace_list_emits_list_level_sentinel(self, capsys):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "dpo.jsonl"
            count = df.format_rejected_traces([], out)
        assert count == 0
        captured = capsys.readouterr()
        assert "[SENTINEL-DPO-EMPTY]" in captured.out

    def test_missing_rejected_field_emits_per_trace_sentinel(self, capsys):
        traces = [
            {"node_id": "missing_rejected_01", "corrected": {"fix": "ok"}},
        ]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "dpo.jsonl"
            count = df.format_rejected_traces(traces, out)
        assert count == 0
        captured = capsys.readouterr()
        assert "[SENTINEL-DPO-EMPTY]" in captured.out
        assert "missing_rejected_01" in captured.out
        assert "'rejected'" in captured.out

    def test_missing_chosen_field_emits_per_trace_sentinel(self, capsys):
        traces = [
            {"node_id": "missing_chosen_01", "rejected": {"bug": "xyz"}},
        ]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "dpo.jsonl"
            count = df.format_rejected_traces(traces, out)
        assert count == 0
        captured = capsys.readouterr()
        assert "[SENTINEL-DPO-EMPTY]" in captured.out
        assert "missing_chosen_01" in captured.out
        assert "chosen" in captured.out.lower()

    def test_summary_line_reports_skip_counts(self, capsys):
        traces = [
            {"node_id": "t_a", "corrected": {"fix": "ok"}},          # missing rejected
            {"node_id": "t_b", "rejected": {"bug": "xyz"}},          # missing chosen
            {                                                        # valid pair
                "node_id": "t_c",
                "rejected": {"bug": "boom"},
                "corrected": {"fix": "patched"},
            },
        ]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "dpo.jsonl"
            count = df.format_rejected_traces(traces, out)
        assert count == 1
        captured = capsys.readouterr()
        # Per-trace emissions
        assert captured.out.count("[SENTINEL-DPO-EMPTY]") >= 3  # 2 per-trace + 1 summary
        # Summary fields
        assert "Summary" in captured.out
        assert "missing rejected" in captured.out
        assert "missing chosen" in captured.out

    def test_mixed_valid_and_invalid_pairs_only_writes_valid(self, capsys):
        traces = [
            {"node_id": "bad", "rejected": {"x": 1}},  # missing chosen
            {                                           # valid
                "node_id": "good",
                "rejected": {"x": 1},
                "corrected": {"x": 2},
            },
        ]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "dpo.jsonl"
            count = df.format_rejected_traces(traces, out)
            assert count == 1
            lines = out.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert "good" in record["prompt"]
            assert "bad" not in record["prompt"]
