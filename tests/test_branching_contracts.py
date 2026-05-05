"""
Phase 2 tests — branching_contracts.py

Covers:
  - BranchTrace schema validation (pass and fail cases)
  - evaluate_branch_topology() structural metrics
  - Fake-contrast detection via Jaccard similarity
  - Configurable threshold behaviour
"""
import sys
from pathlib import Path
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pydantic import ValidationError
import src.pipeline.branching_contracts as bc
from src.pipeline.branching_contracts import (
    Branch,
    BranchTrace,
    evaluate_branch_topology,
    _jaccard,
    MIN_BRANCH_COUNT,
    BRANCH_MECHANISM_SIMILARITY_MAX,
    REQUIRE_FAILURE_BOUNDARY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _branch(branch_id="b1", condition="If temperature increases",
            mechanism="Higher temperature provides more thermal energy to reactant molecules, "
                      "increasing collision frequency beyond the activation energy threshold.",
            first_order_effect="Reaction rate increases.",
            second_order_effect="Product yield rises faster than heat dissipation allows.",
            failure_boundary="Above 600 K, catalyst sintering occurs and selectivity collapses.",
            assumption_risk="medium") -> dict:
    return dict(
        branch_id=branch_id,
        condition=condition,
        mechanism=mechanism,
        first_order_effect=first_order_effect,
        second_order_effect=second_order_effect,
        failure_boundary=failure_boundary,
        assumption_risk=assumption_risk,
    )


def _branch2() -> dict:
    return _branch(
        branch_id="b2",
        condition="If temperature decreases",
        mechanism="Lower thermal energy reduces collision frequency, so fewer molecules "
                  "reach the activation energy threshold, slowing the reaction rate.",
        first_order_effect="Reaction rate decreases.",
        second_order_effect="Selectivity may improve because competing side reactions are also suppressed.",
        failure_boundary="Below 200 K, reactant viscosity increases and diffusion limits dominate.",
    )


def _valid_trace(**kwargs) -> dict:
    base = dict(
        schema_version="2.0",
        baseline_state="At standard operating temperature the catalyst surface is partially "
                       "covered by adsorbed reactant and maintains steady-state turnover.",
        branches=[_branch(), _branch2()],
        synthesis="Both branches confirm that temperature is the primary control variable, "
                  "with opposing effects on rate and selectivity defining an optimal operating window.",
        final_answer="Activation energy governs the reaction rate because only molecules with "
                     "sufficient thermal energy reach the transition state; therefore increasing "
                     "temperature leads to exponentially higher rates via the Arrhenius equation, "
                     "while decreasing temperature slows the reaction and improves selectivity by "
                     "suppressing competing pathways.",
    )
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# BranchTrace schema validation
# ---------------------------------------------------------------------------

class TestBranchTraceValidation:

    def test_valid_trace_passes(self):
        bt = BranchTrace.model_validate(_valid_trace())
        assert bt.schema_version == "2.0"
        assert len(bt.branches) == 2

    def test_wrong_schema_version_fails(self):
        with pytest.raises(ValidationError):
            BranchTrace.model_validate(_valid_trace(schema_version="1.0"))

    def test_single_branch_fails_min_length(self):
        data = _valid_trace(branches=[_branch()])
        with pytest.raises(ValidationError) as exc_info:
            BranchTrace.model_validate(data)
        assert "branches" in str(exc_info.value).lower() or "min_length" in str(exc_info.value).lower()

    def test_empty_branches_fails(self):
        with pytest.raises(ValidationError):
            BranchTrace.model_validate(_valid_trace(branches=[]))

    def test_short_baseline_state_fails(self):
        with pytest.raises(ValidationError):
            BranchTrace.model_validate(_valid_trace(baseline_state="Too short."))

    def test_short_final_answer_fails(self):
        with pytest.raises(ValidationError):
            BranchTrace.model_validate(_valid_trace(final_answer="Short."))

    def test_extra_fields_forbidden(self):
        data = _valid_trace()
        data["unexpected_field"] = "surprise"
        with pytest.raises(ValidationError):
            BranchTrace.model_validate(data)


class TestBranchValidation:

    def test_branch_requires_failure_boundary_when_globally_required(self):
        if not REQUIRE_FAILURE_BOUNDARY:
            pytest.skip("REQUIRE_FAILURE_BOUNDARY is False in this environment")
        b = _branch(failure_boundary="")
        with pytest.raises(ValidationError) as exc_info:
            Branch.model_validate(b)
        assert "failure_boundary" in str(exc_info.value).lower()

    def test_branch_short_condition_fails(self):
        with pytest.raises(ValidationError):
            Branch.model_validate(_branch(condition="Short"))

    def test_branch_short_mechanism_fails(self):
        with pytest.raises(ValidationError):
            Branch.model_validate(_branch(mechanism="Too short."))

    def test_branch_invalid_assumption_risk_fails(self):
        with pytest.raises(ValidationError):
            Branch.model_validate(_branch(assumption_risk="extreme"))

    def test_branch_evidence_refs_defaults_to_empty_list(self):
        b = Branch.model_validate(_branch())
        assert b.evidence_refs == []

    def test_branch_evidence_refs_accepted(self):
        b = Branch.model_validate(_branch())
        data = _branch()
        data["evidence_refs"] = ["doc_001", "concept_A"]
        b2 = Branch.model_validate(data)
        assert b2.evidence_refs == ["doc_001", "concept_A"]


# ---------------------------------------------------------------------------
# evaluate_branch_topology
# ---------------------------------------------------------------------------

class TestEvaluateBranchTopology:

    def _bt(self, **kwargs) -> BranchTrace:
        return BranchTrace.model_validate(_valid_trace(**kwargs))

    def test_valid_diverse_branches_pass(self):
        bt = self._bt()
        result = evaluate_branch_topology(bt)
        assert result["pass"] is True
        assert result["reason"] is None
        assert result["branch_count"] == 2

    def test_fake_contrast_detected(self):
        """Two branches sharing >70% mechanism word overlap must fail."""
        identical_mechanism = (
            "Higher temperature provides more thermal energy to reactant molecules, "
            "increasing collision frequency beyond the activation energy threshold."
        )
        b1 = _branch(branch_id="b1", mechanism=identical_mechanism)
        b2 = _branch(branch_id="b2",
                     condition="When temperature decreases",
                     mechanism=identical_mechanism)  # identical → Jaccard = 1.0
        bt = BranchTrace.model_validate(_valid_trace(branches=[b1, b2]))
        result = evaluate_branch_topology(bt)
        assert result["pass"] is False
        assert result["reason"] == "weak_mechanism_diversity"
        assert result["max_mechanism_similarity"] > BRANCH_MECHANISM_SIMILARITY_MAX

    def test_missing_failure_boundary_fails(self):
        if not REQUIRE_FAILURE_BOUNDARY:
            pytest.skip("REQUIRE_FAILURE_BOUNDARY is False in this environment")
        b1 = _branch(branch_id="b1")
        b2_data = _branch2()
        b2_data["failure_boundary"] = ""
        # Bypass Pydantic validator by patching after construction
        bt = BranchTrace.model_validate(_valid_trace())
        # Directly mutate branch failure_boundary on the model
        bt.branches[1].failure_boundary = ""
        result = evaluate_branch_topology(bt)
        assert result["pass"] is False
        assert result["reason"] == "missing_failure_boundary"

    def test_condition_coverage_below_threshold_fails(self):
        """A branch with a terse condition (>= 10 chars but < 20) reduces condition_coverage.

        Pydantic enforces min_length=10, so the condition passes schema validation.
        evaluate_branch_topology() applies a stricter >= 20 char threshold for "substantive"
        conditions, so coverage falls below 1.0 and may trigger incomplete_branch_conditions.
        """
        b1 = _branch(branch_id="b1")
        b2_data = _branch2()
        b2_data["condition"] = "When X rises"  # 13 chars — passes Pydantic, fails evaluator threshold
        bt = BranchTrace.model_validate(_valid_trace(branches=[b1, b2_data]))
        result = evaluate_branch_topology(bt)
        assert result["condition_coverage"] < 1.0

    def test_three_diverse_branches_pass(self):
        b3 = _branch(
            branch_id="b3",
            condition="When the system is at equilibrium",
            mechanism="At steady state the forward and reverse reaction rates are equal, so no net "
                      "change in concentration occurs and the system is thermodynamically stable.",
            first_order_effect="No net change in reactant or product concentrations.",
            second_order_effect="Small perturbations are absorbed via Le Chatelier equilibrium shifts.",
            failure_boundary="Catalyst poisoning at this regime shifts equilibrium irreversibly.",
        )
        bt = BranchTrace.model_validate(_valid_trace(branches=[_branch(), _branch2(), b3]))
        result = evaluate_branch_topology(bt)
        assert result["pass"] is True
        assert result["branch_count"] == 3


# ---------------------------------------------------------------------------
# _jaccard helper
# ---------------------------------------------------------------------------

class TestJaccard:

    def test_identical_strings_score_one(self):
        assert _jaccard("activation energy catalyst", "activation energy catalyst") == 1.0

    def test_disjoint_strings_score_zero(self):
        assert _jaccard("alpha beta gamma", "delta epsilon zeta") == 0.0

    def test_partial_overlap(self):
        sim = _jaccard("activation energy catalyst reaction", "activation energy bond thermal")
        assert 0.0 < sim < 1.0

    def test_empty_strings_score_one(self):
        assert _jaccard("", "") == 1.0

    def test_one_empty_string_scores_zero(self):
        assert _jaccard("word", "") == 0.0


# ---------------------------------------------------------------------------
# Phase 3 formatter/auditor integration tests
# ---------------------------------------------------------------------------

from src.pipeline.sft_formatter import _knowledge_chatml_record, _render_branch_trace
from scripts.flashcard_auditor import compute_branch_metrics, generate_branch_quality_summary


def _v2_payload(**overrides) -> dict:
    trace = _valid_trace()
    payload = {
        "mode": "theorist",
        "skill_type": "theoretical_reasoning",
        "source_type": "ocr_document",
        "name": "activation energy",
        "schema_version": "2.0",
        "branch_trace": trace,
        "final_answer": trace["final_answer"],
        "code_snippet": "RAW OCR SOURCE SHOULD NOT BECOME ASSISTANT OUTPUT",
        "_source_doc": "source.pdf",
        "_node_id": "node-001",
    }
    payload.update(overrides)
    return payload


def _assistant_text(record: dict) -> str:
    return next(m["content"] for m in record["messages"] if m["role"] == "assistant")


class TestPhase3BranchRendering:

    def test_render_branch_trace_is_deterministic_and_sanitizes_inner_angles(self):
        trace = _valid_trace()
        trace["branches"][0]["mechanism"] = "<unsafe> Molecular pathway\nwith newline > artifact"
        first  = _render_branch_trace(trace, trace["final_answer"])
        second = _render_branch_trace(trace, trace["final_answer"])
        assert first == second
        assert first.count("<think>") == 1
        assert first.count("</think>") == 1
        think_body = first.split("</think>", 1)[0].replace("<think>", "")
        assert "<" not in think_body
        assert ">" not in think_body

    def test_v2_knowledge_record_renders_chatml_and_preserves_compact_metadata_only(self):
        record = _knowledge_chatml_record(_v2_payload(), identity=None)
        assert record["_mode"] == "knowledge"
        assert record["schema_version"] == "2.0"
        assert record["branch_count"] == 2
        assert record["failure_boundary_coverage"] >= 0.8
        assert record["max_mechanism_similarity"] >= 0.0
        assert record["condition_coverage"] >= 0.8
        assert "branch_trace" not in record
        assert "code_snippet" not in record
        assistant = _assistant_text(record)
        assert assistant.startswith("<think>")
        assert "Branch: b1" in assistant
        assert "Failure boundary:" in assistant
        assert "RAW OCR SOURCE SHOULD NOT BECOME ASSISTANT OUTPUT" not in assistant

    def test_malformed_v2_payload_is_rejected(self):
        bad_trace = _valid_trace(branches=[])
        assert _knowledge_chatml_record(_v2_payload(branch_trace=bad_trace), identity=None) == {}

    def test_v1_dual_channel_theorist_path_still_renders(self):
        payload = {
            "mode": "theorist",
            "skill_type": "theoretical_reasoning",
            "source_type": "ocr_document",
            "name": "activation energy",
            "reasoning_trace": ["<unsafe> causal step with angle brackets"],
            "final_answer": "Activation energy controls rate by setting the barrier reactants must overcome before products can form.",
        }
        record = _knowledge_chatml_record(payload, identity=None)
        assistant = _assistant_text(record)
        assert "<think>" in assistant
        assert "</think>" in assistant
        assert "<unsafe>" not in assistant


class TestPhase3BranchAuditor:

    def test_auditor_branch_summary_passes_for_valid_v2_row(self):
        record = _knowledge_chatml_record(_v2_payload(), identity=None)
        metrics = [compute_branch_metrics(record)]
        summary = generate_branch_quality_summary(metrics, total_rows=1)
        assert summary["schema_v2_rows"] == 1
        assert summary["think_coverage"] == 1.0
        assert summary["branch_count_p10"] >= 2
        assert summary["failure_boundary_coverage_mean"] >= 0.8
        assert summary["fake_contrast_rate"] < 0.10
        assert summary["pass"] is True

    def test_zero_v2_dataset_returns_pass_false_without_crashing(self):
        payload = {
            "mode": "theorist",
            "skill_type": "theoretical_reasoning",
            "source_type": "ocr_document",
            "name": "activation energy",
            "reasoning_trace": ["Linear but valid v1 reasoning trace."],
            "final_answer": "Activation energy controls rate by setting the barrier reactants must overcome before products can form.",
        }
        record = _knowledge_chatml_record(payload, identity=None)
        summary = generate_branch_quality_summary([compute_branch_metrics(record)], total_rows=1)
        assert summary["schema_v2_rows"] == 0
        assert summary["pass"] is False
        assert "no_v2_rows" in summary["reasons"]
