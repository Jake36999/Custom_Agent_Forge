import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipeline import sft_formatter as sf



def _write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")



def test_phase_e_chatml_includes_knowledge_and_failure_rows() -> None:
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        knowledge = base / "knowledge_cards.jsonl"
        failure = base / "phase_failure_matrix.jsonl"
        out_file = base / "chatml.jsonl"

        _write_jsonl(
            knowledge,
            [
                {
                    "instruction": "Explain safe branching.",
                    "input": '{"topic":"dag"}',
                    "output": '{"result":"Use guarded transitions"}',
                    "node_id": "should_be_removed",
                    "execution_eligible": False,
                }
            ],
        )

        _write_jsonl(
            failure,
            [
                {
                    "failure_id": "f1",
                    "node_id": "bad-node",
                    "failure_type": "epistemic_gate",
                    "rejection_reason": "state=REJECTED",
                    "sie_snapshot": {"content_density": 0.2, "s_sie": 0.1, "composite_quality_score": 0.03},
                    "trajectory": {
                        "trajectory_id": "t1",
                        "steps": [{"error": "bad", "diagnosis": "drift", "fix": "repair", "attempt_code": ""}],
                    },
                    "advocate_audit": {"root_cause": "upstream_sie_failure", "recommendation": "revalidate"},
                    "execution_eligible": False,
                    "_reroll_count": 2,
                }
            ],
        )

        count = sf.format_complexes(base, out_file, identity="Aletheia Identity", failure_matrix_file=failure)
        assert count == 2

        lines = out_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        records = [json.loads(line) for line in lines]

        for rec in records:
            assert [m["role"] for m in rec["messages"]] == ["system", "user", "assistant"]
            assert rec["messages"][0]["train_loss"] is False
            assert rec["messages"][1]["train_loss"] is False
            assert rec["messages"][2]["train_loss"] is True
            blob = json.dumps(rec, sort_keys=True)
            assert "node_id" not in blob
            assert "execution_eligible" not in blob
            assert "_reroll_count" not in blob

        modes = sorted(r.get("_mode") for r in records)
        assert modes == ["knowledge", "veteran"]



def test_phase_e_chatml_is_byte_stable() -> None:
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        knowledge = base / "knowledge_cards.jsonl"
        out_a = base / "a.jsonl"
        out_b = base / "b.jsonl"

        _write_jsonl(
            knowledge,
            [
                {
                    "instruction": "I2",
                    "input": '{"k":2}',
                    "output": '{"o":2}',
                },
                {
                    "instruction": "I1",
                    "input": '{"k":1}',
                    "output": '{"o":1}',
                },
            ],
        )

        sf.format_complexes(base, out_a, identity="StableIdentity")
        sf.format_complexes(base, out_b, identity="StableIdentity")

        assert out_a.read_text(encoding="utf-8") == out_b.read_text(encoding="utf-8")



def test_phase_e_optional_veteran_downsample_is_deterministic() -> None:
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        knowledge = base / "knowledge_cards.jsonl"
        failure = base / "phase_failure_matrix.jsonl"
        out_a = base / "downsample_a.jsonl"
        out_b = base / "downsample_b.jsonl"

        _write_jsonl(
            knowledge,
            [
                {"instruction": "K1", "output": "A1"},
                {"instruction": "K2", "output": "A2"},
            ],
        )
        _write_jsonl(
            failure,
            [
                {
                    "failure_id": f"f{i}",
                    "failure_type": "epistemic_gate",
                    "rejection_reason": f"state=REJECTED:{i}",
                    "sie_snapshot": {"content_density": 0.2, "s_sie": 0.1},
                    "trajectory": {"trajectory_id": f"t{i}", "steps": [{"error": "bad"}]},
                    "advocate_audit": {"root_cause": "upstream_sie_failure"},
                }
                for i in range(5)
            ],
        )

        count_a = sf.format_complexes(
            base,
            out_a,
            identity="StableIdentity",
            failure_matrix_file=failure,
            max_veteran_ratio=0.5,
            balance_seed=7,
        )
        count_b = sf.format_complexes(
            base,
            out_b,
            identity="StableIdentity",
            failure_matrix_file=failure,
            max_veteran_ratio=0.5,
            balance_seed=7,
        )

        assert count_a == count_b == 3
        assert out_a.read_text(encoding="utf-8") == out_b.read_text(encoding="utf-8")

        records = [json.loads(line) for line in out_a.read_text(encoding="utf-8").splitlines()]
        modes = Counter(rec["_mode"] for rec in records)
        assert modes == {"knowledge": 2, "veteran": 1}


# ---------------------------------------------------------------------------
# Phase 1 — Parrot guard tests
# ---------------------------------------------------------------------------

def _make_theorist_payload(*, reasoning_trace=None, final_answer=None,
                           code_snippet=None, output=None, schema_version=None) -> dict:
    """Minimal valid theorist record with configurable fields."""
    p = {
        "mode": "theorist",
        "skill_type": "theoretical_reasoning",
        "source_type": "ocr_document",
        "name": "Activation Energy",
        "instruction": "Explain activation energy.",
    }
    if reasoning_trace is not None:
        p["reasoning_trace"] = reasoning_trace
    if final_answer is not None:
        p["final_answer"] = final_answer
    if code_snippet is not None:
        p["code_snippet"] = code_snippet
    if output is not None:
        p["output"] = output
    if schema_version is not None:
        p["schema_version"] = schema_version
    return p


def test_theorist_without_reasoning_trace_is_dropped() -> None:
    """Theorist record with no reasoning_trace must be dropped — never formatted."""
    payload = _make_theorist_payload(final_answer="Some final answer with enough words here.")
    result = sf._knowledge_chatml_record(sf._scrub_internal_metadata(payload), identity=None)
    assert result == {}, f"Expected empty dict, got {result}"


def test_theorist_with_empty_reasoning_trace_is_dropped() -> None:
    """reasoning_trace=[] (empty list) must not produce a ChatML row."""
    payload = _make_theorist_payload(
        reasoning_trace=[],
        final_answer="Some final answer with enough words here.",
    )
    result = sf._knowledge_chatml_record(sf._scrub_internal_metadata(payload), identity=None)
    assert result == {}


def test_theorist_code_snippet_never_becomes_assistant_output() -> None:
    """code_snippet raw OCR text must never appear as assistant content for theorist records."""
    raw_ocr = "This raw OCR source text should never end up as training output for the model."
    payload = _make_theorist_payload(
        code_snippet=raw_ocr,
        output=raw_ocr,
        # No reasoning_trace — simulates a pre-RAL or RAL-failed record.
    )
    result = sf._knowledge_chatml_record(sf._scrub_internal_metadata(payload), identity=None)
    assert result == {}, "Parrot guard failed: raw OCR reached the formatter output"
    # Double-check: even if it somehow returned a record, OCR must not be in assistant turn.
    if result:
        assistant_content = next(
            (m["content"] for m in result.get("messages", []) if m["role"] == "assistant"), ""
        )
        assert raw_ocr not in assistant_content, "OCR text leaked into assistant turn"


def test_theorist_v2_schema_version_returns_empty_pending_phase3() -> None:
    """schema_version=2.0 records are recognized but not rendered until Phase 3."""
    payload = _make_theorist_payload(
        schema_version="2.0",
        final_answer="Fifty word final answer with because and therefore and leads to mechanistic prose.",
    )
    result = sf._knowledge_chatml_record(sf._scrub_internal_metadata(payload), identity=None)
    assert result == {}, "v2 records must not be formatted before Phase 3 renderer is in place"


def test_theorist_valid_v1_dual_channel_formats_correctly() -> None:
    """Regression: valid v1 reasoning_trace + final_answer must still produce correct ChatML.

    The payload includes 'output' matching final_answer because real RAL amplified records
    always write output=final_answer, and format_complexes() gates on _OUTPUT_FIELD_ALIASES
    before calling _knowledge_chatml_record().
    """
    steps = [
        "Higher temperature increases collision frequency between reactant molecules.",
        "More collisions means more molecules exceed the activation energy threshold.",
        "Therefore the reaction rate increases because more successful collisions occur per second.",
    ]
    final = (
        "Activation energy acts as a kinetic barrier because only molecules with sufficient "
        "thermal energy can achieve the transition state, therefore increasing temperature "
        "leads to exponentially higher reaction rates via the Arrhenius relationship."
    )
    # Real RAL records: output is set to final_answer, code_snippet carries the raw OCR source.
    raw_ocr = "Raw OCR source text that must never appear in the training assistant turn."
    payload = _make_theorist_payload(
        reasoning_trace=steps,
        final_answer=final,
        output=final,           # simulates what ral_amplifier.py writes: "output": parsed["final_answer"]
        code_snippet=raw_ocr,   # raw OCR persists in the record but must not leak into output
    )

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        knowledge = base / "theorist.jsonl"
        out_file  = base / "out.jsonl"
        _write_jsonl(knowledge, [payload])
        count = sf.format_complexes(base, out_file)

        assert count == 1
        rec = json.loads(out_file.read_text(encoding="utf-8").strip())
        messages = rec["messages"]
        assert [m["role"] for m in messages] == ["system", "user", "assistant"]
        assistant_text = messages[2]["content"]
        assert "<think>" in assistant_text
        assert "</think>" in assistant_text
        assert final in assistant_text
        # Raw OCR must not appear in the assistant turn.
        assert raw_ocr not in assistant_text


# ---------------------------------------------------------------------------
# Phase 3 — v2 branch renderer tests
# ---------------------------------------------------------------------------

def _valid_branch_trace_dict() -> dict:
    """Minimal valid BranchTrace as a plain dict (simulates ral_amplifier output)."""
    return {
        "schema_version": "2.0",
        "baseline_state": (
            "At standard operating conditions the catalyst surface is partially covered "
            "by adsorbed reactant and maintains steady-state turnover."
        ),
        "branches": [
            {
                "branch_id": "upward_shift",
                "condition": "If temperature increases significantly",
                "mechanism": (
                    "Higher thermal energy increases collision frequency and the fraction of "
                    "molecules exceeding the activation energy threshold."
                ),
                "first_order_effect": "Reaction rate increases.",
                "second_order_effect": "Product yield rises faster than heat dissipation allows.",
                "failure_boundary": "Above 600 K catalyst sintering collapses selectivity.",
                "assumption_risk": "medium",
                "evidence_refs": ["doc_001"],
            },
            {
                "branch_id": "downward_shift",
                "condition": "If temperature decreases below operating range",
                "mechanism": (
                    "Lower thermal energy reduces collision frequency so fewer molecules "
                    "reach the transition state, slowing the net reaction rate."
                ),
                "first_order_effect": "Reaction rate decreases.",
                "second_order_effect": "Selectivity may improve as competing side reactions are suppressed.",
                "failure_boundary": "Below 200 K reactant viscosity limits diffusion.",
                "assumption_risk": "low",
                "evidence_refs": [],
            },
        ],
        "synthesis": (
            "Temperature is the primary control variable; opposing rate and selectivity effects "
            "define an optimal operating window between 200 K and 600 K."
        ),
        "final_answer": (
            "Activation energy governs the reaction rate because only molecules with sufficient "
            "thermal energy reach the transition state; therefore increasing temperature leads to "
            "exponentially higher rates via the Arrhenius equation, while decreasing temperature "
            "slows the reaction and improves selectivity by suppressing competing pathways."
        ),
    }


def _valid_v2_payload(branch_trace_override=None, final_answer=None, **metadata) -> dict:
    """Simulate a ral_amplifier v2 amplified record fed into sft_formatter."""
    bt = branch_trace_override or _valid_branch_trace_dict()
    _default_fa = (
        "Activation energy governs the reaction rate because only molecules with sufficient "
        "thermal energy reach the transition state; therefore increasing temperature leads to "
        "exponentially higher rates via the Arrhenius equation."
    )
    fa = final_answer or bt.get("final_answer") or _default_fa
    p = {
        "mode": "theorist",
        "skill_type": "theoretical_reasoning",
        "source_type": "ocr_document",
        "name": "Activation Energy",
        "instruction": "Explain activation energy.",
        "schema_version": "2.0",
        "branch_trace": bt,
        "final_answer": fa,
        "output": fa,
        "branch_count": len(bt.get("branches", [])),
        "failure_boundary_coverage": 1.0,
        "max_mechanism_similarity": 0.25,
        "condition_coverage": 1.0,
        "reasoning_type": "standard",
    }
    p.update(metadata)
    return p


class TestRenderBranchTrace:

    def _render(self, bt_dict=None, final_answer=None):
        bt_dict = bt_dict or _valid_branch_trace_dict()
        fa = final_answer or bt_dict["final_answer"]
        return sf._render_branch_trace(bt_dict, fa)

    def test_output_is_deterministic(self):
        bt = _valid_branch_trace_dict()
        assert self._render(bt) == self._render(bt)

    def test_contains_exactly_one_think_open_and_close(self):
        out = self._render()
        assert out.count("<think>") == 1
        assert out.count("</think>") == 1

    def test_think_opens_before_closes(self):
        out = self._render()
        assert out.index("<think>") < out.index("</think>")

    def test_no_angle_brackets_inside_think_body(self):
        out = self._render()
        think_body = out.split("</think>")[0].replace("<think>", "")
        # The only allowed angle-bracket sequences are the outer tags themselves.
        assert "<" not in think_body
        assert ">" not in think_body

    def test_field_values_sanitized(self):
        """Angle brackets inside branch fields must be stripped from the rendered output."""
        bt = _valid_branch_trace_dict()
        bt["branches"][0]["mechanism"] = "Electrons flow <through> the <barrier> bond."
        out = self._render(bt)
        assert "<through>" not in out
        assert "<barrier>" not in out
        assert "Electrons flow" in out

    def test_contains_required_section_labels(self):
        out = self._render()
        for label in ("Baseline:", "Branch:", "Condition:", "Mechanism:",
                      "Failure boundary:", "Synthesis:"):
            assert label in out, f"Missing label: {label!r}"

    def test_all_branch_ids_present(self):
        out = self._render()
        assert "upward_shift" in out
        assert "downward_shift" in out

    def test_final_answer_appears_after_closing_tag(self):
        out = self._render()
        close_pos = out.index("</think>")
        fa_fragment = "Arrhenius equation"
        assert out.index(fa_fragment) > close_pos

    def test_evidence_refs_rendered_or_none(self):
        out = self._render()
        # Branch 1 has evidence_refs=["doc_001"] — should appear.
        assert "doc_001" in out
        # Branch 2 has evidence_refs=[] — should show "none".
        assert "none" in out


class TestV2KnowledgeChatmlRecord:

    def test_valid_v2_payload_produces_chatml(self):
        payload = _valid_v2_payload()
        result = sf._knowledge_chatml_record(sf._scrub_internal_metadata(payload), identity=None)
        assert result != {}
        messages = result["messages"]
        assert [m["role"] for m in messages] == ["system", "user", "assistant"]

    def test_assistant_content_contains_required_labels(self):
        payload = _valid_v2_payload()
        result = sf._knowledge_chatml_record(sf._scrub_internal_metadata(payload), identity=None)
        assistant = next(m["content"] for m in result["messages"] if m["role"] == "assistant")
        for label in ("Baseline:", "Branch:", "Condition:", "Mechanism:", "Failure boundary:", "Synthesis:"):
            assert label in assistant

    def test_compact_metadata_preserved(self):
        payload = _valid_v2_payload()
        result = sf._knowledge_chatml_record(sf._scrub_internal_metadata(payload), identity=None)
        assert result["schema_version"] == "2.0"
        assert result["branch_count"] == 2
        assert result["failure_boundary_coverage"] == 1.0
        assert result["max_mechanism_similarity"] == pytest.approx(0.25)
        assert result["condition_coverage"] == 1.0

    def test_branch_trace_not_in_final_record(self):
        payload = _valid_v2_payload()
        result = sf._knowledge_chatml_record(sf._scrub_internal_metadata(payload), identity=None)
        result_str = json.dumps(result)
        # branch_trace should not appear as a key in the formatted row.
        assert '"branch_trace"' not in result_str

    def test_malformed_branch_trace_returns_empty(self):
        payload = _valid_v2_payload(branch_trace_override={"schema_version": "2.0", "branches": []})
        result = sf._knowledge_chatml_record(sf._scrub_internal_metadata(payload), identity=None)
        assert result == {}

    def test_missing_branch_trace_key_returns_empty(self):
        payload = _valid_v2_payload()
        del payload["branch_trace"]
        result = sf._knowledge_chatml_record(sf._scrub_internal_metadata(payload), identity=None)
        assert result == {}

    def test_v2_payload_via_format_complexes(self):
        """End-to-end: v2 payload → format_complexes → valid ChatML JSONL row."""
        payload = _valid_v2_payload()
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            _write_jsonl(base / "amplified.jsonl", [payload])
            out = base / "out.jsonl"
            count = sf.format_complexes(base, out)
            assert count == 1
            rec = json.loads(out.read_text(encoding="utf-8").strip())
            assert rec["schema_version"] == "2.0"
            assert rec["branch_count"] == 2
            assistant = next(m["content"] for m in rec["messages"] if m["role"] == "assistant")
            assert "<think>" in assistant
            assert "Baseline:" in assistant

    def test_v1_records_still_render_after_v2_added(self):
        """Regression: v1 records must not be broken by the Phase 3 changes."""
        steps = [
            "Higher temperature increases molecular collision frequency.",
            "More collisions means more molecules exceed the activation energy.",
            "Therefore rate increases because more successful collisions occur.",
        ]
        final = (
            "Activation energy acts as a kinetic barrier because only molecules with "
            "sufficient thermal energy reach the transition state, therefore increasing "
            "temperature leads to higher rates via the Arrhenius relationship."
        )
        v1_payload = _make_theorist_payload(
            reasoning_trace=steps, final_answer=final, output=final,
        )
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            _write_jsonl(base / "v1.jsonl", [v1_payload])
            out = base / "out.jsonl"
            count = sf.format_complexes(base, out)
            assert count == 1
            rec = json.loads(out.read_text(encoding="utf-8").strip())
            assistant = next(m["content"] for m in rec["messages"] if m["role"] == "assistant")
            assert "<think>" in assistant
            # v1 records must NOT have schema_version field.
            assert "schema_version" not in rec

    def test_theorist_no_fields_still_returns_empty(self):
        """After Phase 3, the basic parrot guard must still hold."""
        payload = _make_theorist_payload()
        result = sf._knowledge_chatml_record(sf._scrub_internal_metadata(payload), identity=None)
        assert result == {}
