import sys
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipeline.branching_contracts import BranchTrace


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


def test_mode_aliases_normalize_to_canonical_modes():
    from src.core.mode_registry import CanonicalMode, normalize_mode

    assert normalize_mode("software_tech") is CanonicalMode.SOFTWARE_TECH
    assert normalize_mode("scholar") is CanonicalMode.SCHOLAR
    assert normalize_mode("project_reviewer") is CanonicalMode.PROJECT_REVIEWER
    assert normalize_mode("matured") is CanonicalMode.MATURED
    assert normalize_mode("coding_assistant") is CanonicalMode.SOFTWARE_TECH
    assert normalize_mode("theorist") is CanonicalMode.SCHOLAR
    assert normalize_mode("advocate") is CanonicalMode.PROJECT_REVIEWER
    assert normalize_mode("project_manager") is CanonicalMode.PROJECT_REVIEWER
    assert normalize_mode("veteran") is CanonicalMode.MATURED


def test_unknown_mode_rejects():
    from src.core.mode_registry import normalize_mode

    with pytest.raises(ValueError, match="Unknown mode"):
        normalize_mode("storyteller")


def test_canonical_mode_value_and_is_mode_helpers():
    from src.core.mode_registry import canonical_mode_value, is_mode

    assert canonical_mode_value("advocate") == "project_reviewer"
    assert canonical_mode_value("project_manager") == "project_reviewer"
    assert is_mode("theorist", "scholar")
    assert is_mode("coding_assistant", "software_tech")
    assert not is_mode("veteran", "scholar")


def test_strict_control_model_rejects_extra_fields():
    from src.core.meta_reasoning_models import IdentityScope

    with pytest.raises(ValidationError):
        IdentityScope.model_validate(
            {
                "compiled_mode": "scholar",
                "role_contract_id": "role.scholar",
                "role_contract_version": 1,
                "stable_self_hash": "abc123",
                "identity_invariants": ["stay source grounded"],
                "allowed_output_forms": ["answer"],
                "prohibited_conflations": ["identity is not personality"],
                "allowed_action_classes": ["answer"],
                "unexpected": "blocked",
            }
        )


def test_free_floating_lens_is_invalid():
    from src.core.meta_reasoning_models import LensCoordinate

    with pytest.raises(ValidationError):
        LensCoordinate.model_validate(
            {
                "kind": "lens",
                "lens_id": "lens.false_bridge",
                "subtype": "bridge_lens",
                "target_edge_types": ["domain_bridge"],
                "prompt_pattern": "Check whether this bridge has evidence.",
                "activation_condition": "cross_domain == true",
                "intervention_type": "warn",
                "confidence": 0.8,
                "risk": "medium",
            }
        )


def test_invalid_training_visibility_rejects():
    from src.core.meta_reasoning_models import BoundaryNode

    with pytest.raises(ValidationError):
        BoundaryNode.model_validate(
            {
                "kind": "boundary",
                "subtype": "identity_boundary",
                "scope": "node",
                "activation_condition": "always",
                "intervention_type": "observe",
                "confidence": 0.7,
                "risk": "low",
                "training_visibility": "assistant_visible",
            }
        )


def test_control_node_discriminated_union_selects_subtype():
    from src.core.meta_reasoning_models import BoundaryNode, ControlNode, TopologyBridgeNode

    adapter = TypeAdapter(ControlNode)

    boundary = adapter.validate_python(
        {
            "kind": "boundary",
            "subtype": "task_boundary",
            "scope": "task",
            "activation_condition": "task scope present",
            "intervention_type": "warn",
            "confidence": 0.9,
            "risk": "low",
        }
    )
    bridge = adapter.validate_python(
        {
            "kind": "bridge",
            "bridge_id": "bridge.mechanism.1",
            "subtype": "mechanism_bridge",
            "source_cluster_id": "cluster.a",
            "target_cluster_id": "cluster.b",
            "relation_type": "domain_bridge",
            "evidence_basis": "explicit_citation",
            "edge_confidence": 0.82,
            "cross_domain": True,
        }
    )

    assert isinstance(boundary, BoundaryNode)
    assert isinstance(bridge, TopologyBridgeNode)


def test_governed_ral_record_v21_accepts_minimal_envelope():
    from src.core.meta_reasoning_models import GovernedRALRecordV21

    branch_trace = _valid_branch_trace_dict()
    record = GovernedRALRecordV21.model_validate(
        {
            "schema_version": "2.1",
            "branch_trace": branch_trace,
            "training_visibility_policy": {},
            "final_answer": "A compact final answer for the governed outer envelope.",
        }
    )

    assert record.schema_version == "2.1"
    assert record.branch_trace == BranchTrace.model_validate(branch_trace)
    assert record.control_nodes == []
    assert record.domain_tags == []


def test_governed_ral_record_v21_preserves_inner_branch_trace_unchanged():
    from src.core.meta_reasoning_models import GovernedRALRecordV21

    branch_trace = _valid_branch_trace_dict()
    record = GovernedRALRecordV21.model_validate(
        {
            "schema_version": "2.1",
            "branch_trace": branch_trace,
            "training_visibility_policy": {"default": "hidden_metadata"},
            "final_answer": "A compact final answer for the governed outer envelope.",
        }
    )

    assert record.branch_trace.model_dump() == BranchTrace.model_validate(branch_trace).model_dump()
