import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipeline import sft_formatter as sf
from src.pipeline.dag_scoring_pass import sanitize_scored_node


def _valid_branch_trace_dict() -> dict:
    return {
        "schema_version": "2.0",
        "baseline_state": "The catalyst begins in a stable resting state before any perturbation.",
        "branches": [
            {
                "branch_id": "thermal_shift",
                "condition": "If the local temperature rises above the activation window",
                "mechanism": "Thermal energy changes the catalyst geometry and alters binding affinity.",
                "first_order_effect": "The reaction rate increases",
                "second_order_effect": "Selectivity begins to drift",
                "failure_boundary": "This branch fails when the catalyst denatures.",
                "assumption_risk": "medium",
                "evidence_refs": ["source:a"],
            },
            {
                "branch_id": "solvent_shift",
                "condition": "When solvent polarity increases across the active site",
                "mechanism": "Polar stabilization shifts the transition state and changes ion pairing.",
                "first_order_effect": "Intermediate stability changes",
                "second_order_effect": "The product distribution shifts",
                "failure_boundary": "This branch fails in nonpolar solvent systems.",
                "assumption_risk": "low",
                "evidence_refs": ["source:b"],
            },
        ],
        "synthesis": "Both branches alter catalytic selectivity through different physical mechanisms.",
        "final_answer": (
            "The catalyst can shift behavior through temperature or solvent polarity. "
            "Each branch changes the active-site conditions through a distinct mechanism, "
            "so the final selectivity depends on which perturbation dominates."
        ),
    }


def _valid_v2_payload(**overrides) -> dict:
    payload = {
        "mode": "theorist",
        "skill_type": "theoretical_reasoning",
        "name": "Catalyst selectivity",
        "schema_version": "2.0",
        "branch_trace": _valid_branch_trace_dict(),
        "final_answer": (
            "The catalyst can shift behavior through temperature or solvent polarity. "
            "Each branch changes the active-site conditions through a distinct mechanism, "
            "so the final selectivity depends on which perturbation dominates."
        ),
    }
    payload.update(overrides)
    return payload


def test_strip_hidden_governance_fields_removes_hidden_keys():
    from src.core.meta_reasoning_models import strip_hidden_governance_fields

    cleaned = strip_hidden_governance_fields(
        {
            "identity_scope": {"compiled_mode": "scholar"},
            "task_scope": {"task_id": "t"},
            "nested": {
                "control_nodes": [{"kind": "boundary"}],
                "safe": "kept",
            },
            "messages": [{"role": "assistant", "content": "safe"}],
        }
    )

    assert "identity_scope" not in cleaned
    assert "task_scope" not in cleaned
    assert "control_nodes" not in cleaned["nested"]
    assert cleaned["nested"]["safe"] == "kept"
    assert cleaned["messages"][0]["content"] == "safe"


def test_strip_hidden_governance_fields_preserves_compact_scalars():
    from src.core.meta_reasoning_models import strip_hidden_governance_fields

    cleaned = strip_hidden_governance_fields(
        {
            "schema_version": "2.0",
            "canonical_mode": "scholar",
            "branch_count": 2,
            "bridge_risk_score": 0.4,
            "thought_action_separation_score": 0.9,
            "identity_scope": {"compiled_mode": "scholar"},
        }
    )

    assert cleaned == {
        "schema_version": "2.0",
        "canonical_mode": "scholar",
        "branch_count": 2,
        "bridge_risk_score": 0.4,
        "thought_action_separation_score": 0.9,
    }


def test_assert_no_hidden_governance_in_text_rejects_identity_scope():
    from src.core.meta_reasoning_models import assert_no_hidden_governance_in_text

    with pytest.raises(ValueError, match="identity_scope"):
        assert_no_hidden_governance_in_text('{"identity_scope": {"compiled_mode": "scholar"}}')


def test_assert_no_hidden_governance_in_messages_rejects_assistant_json():
    from src.core.meta_reasoning_models import assert_no_hidden_governance_in_messages

    with pytest.raises(ValueError, match="control_nodes"):
        assert_no_hidden_governance_in_messages(
            [
                {"role": "system", "content": "safe"},
                {"role": "user", "content": "safe"},
                {"role": "assistant", "content": '{"control_nodes": []}'},
            ]
        )


def test_sft_formatter_drops_theorist_v2_record_when_final_answer_leaks_governance():
    payload = _valid_v2_payload(
        final_answer='The answer is otherwise valid, but it leaks {"identity_scope": {}}.'
    )

    assert sf._knowledge_chatml_record(sf._scrub_internal_metadata(payload), identity=None) == {}


def test_sft_formatter_formats_valid_v2_branch_trace_exactly_as_before_without_leak():
    payload = _valid_v2_payload()
    result = sf._knowledge_chatml_record(sf._scrub_internal_metadata(payload), identity=None)

    assert result
    assistant = result["messages"][2]["content"]
    assert assistant == sf._render_branch_trace(payload["branch_trace"], payload["final_answer"])


def test_sft_formatter_drops_existing_messages_when_assistant_leaks_governance():
    payload = {
        "messages": [
            {"role": "system", "content": "safe"},
            {"role": "user", "content": "safe"},
            {"role": "assistant", "content": "This leaks lens_coordinates into training."},
        ]
    }

    assert sf._knowledge_from_existing_messages(payload, identity=None) == {}


def test_dag_scoring_pass_sanitization_removes_hidden_governance_keys():
    node = sanitize_scored_node(
        {
            "node_id": "node.a",
            "identity_scope": {"compiled_mode": "scholar"},
            "training_visibility_policy": {"default": "hidden_metadata"},
            "semantics": {
                "source_provenance": {"doc": "a"},
                "_lens_diagnostics": [{"risk": "medium"}],
                "domain_fact": "kept",
            },
        }
    )

    assert "identity_scope" not in node
    assert "training_visibility_policy" not in node
    assert "source_provenance" not in node["semantics"]
    assert "_lens_diagnostics" not in node["semantics"]
    assert node["semantics"]["domain_fact"] == "kept"


def test_dag_scoring_pass_sanitization_preserves_branch_audit_scalars():
    node = sanitize_scored_node(
        {
            "schema_version": "2.0",
            "branch_count": 2,
            "failure_boundary_coverage": 1.0,
            "identity_scope": {"compiled_mode": "scholar"},
            "semantics": {"_bridge_risk": {"score": 0.4}},
        }
    )

    assert node["schema_version"] == "2.0"
    assert node["branch_count"] == 2
    assert node["failure_boundary_coverage"] == 1.0
    assert "identity_scope" not in node


def test_theorist_output_equal_to_code_snippet_is_rejected():
    code = "raw OCR source text should not become assistant output"
    payload = {
        "mode": "theorist",
        "skill_type": "theoretical_reasoning",
        "name": "Raw source",
        "code_snippet": code,
        "reasoning_trace": ["A valid reasoning step from the amplified channel."],
        "final_answer": code,
    }

    assert sf._knowledge_chatml_record(sf._scrub_internal_metadata(payload), identity=None) == {}
