import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.mode_registry import CanonicalMode
from src.core.models import AletheiaSkill, IdentityState, ReasoningEdge, TopologyCluster


def _minimal_teaching_layer() -> dict:
    return {
        "skill_identity": {"name": "phase2_fixture"},
        "method_metadata": {"name": "phase2_fixture", "language": "python"},
        "implementation_template": {"code": "def phase2_fixture():\n    return 1\n"},
    }


def _minimal_skill_payload() -> dict:
    return {
        "node_id": "node.phase2",
        "name": "phase2_fixture",
        "file": "phase2_fixture.py",
        "code_snippet": "def phase2_fixture():\n    return 1\n",
        "imports": [],
        "operator_type": "function",
        "teaching_layer": _minimal_teaching_layer(),
    }


def _identity_scope_payload() -> dict:
    return {
        "compiled_mode": "scholar",
        "role_contract_id": "role.scholar",
        "role_contract_version": 1,
        "stable_self_hash": "hash-scholar",
        "identity_invariants": ["stay source grounded"],
        "allowed_output_forms": ["answer"],
        "prohibited_conflations": ["identity is not personality"],
        "allowed_action_classes": ["answer"],
    }


def _task_scope_payload() -> dict:
    return {
        "task_id": "task.phase2",
        "task_class": "schema_expansion",
        "in_scope": ["add optional model fields"],
        "out_of_scope": ["runtime routing"],
        "required_evidence_kinds": ["tests"],
        "abstention_conditions": ["missing schema"],
    }


def _thought_action_payload() -> dict:
    return {
        "thought_state_class": "hypothesis",
        "action_state_class": "none",
        "evidence_threshold_met": True,
        "action_authorised": False,
        "action_justification": "Schema metadata is being stored only.",
        "thought_action_separation_score": 0.95,
    }


def _lens_payload() -> dict:
    return {
        "kind": "lens",
        "lens_id": "lens.false_bridge",
        "subtype": "bridge_lens",
        "applies_to": "edge",
        "target_edge_types": ["domain_bridge"],
        "prompt_pattern": "Check whether this bridge has evidence.",
        "activation_condition": "cross_domain == true",
        "intervention_type": "warn",
        "confidence": 0.75,
        "risk": "medium",
    }


def test_existing_minimal_identity_state_still_validates():
    state = IdentityState.model_validate(
        {
            "role": "Aletheia/LOGOS Principal Architectural Auditor",
            "behavior_profile": {},
            "mode_weights": {
                "theorist": 0.25,
                "coding_assistant": 0.25,
                "advocate": 0.25,
                "veteran": 0.25,
            },
        }
    )

    assert state.compiled_mode is None
    assert state.legacy_mode_aliases == []
    assert state.identity_invariants == []


def test_identity_state_accepts_compiled_mode_string_and_enum():
    from_string = IdentityState.model_validate({"compiled_mode": "scholar"})
    from_enum = IdentityState.model_validate({"compiled_mode": CanonicalMode.SCHOLAR})

    assert from_string.compiled_mode is CanonicalMode.SCHOLAR
    assert from_enum.compiled_mode is CanonicalMode.SCHOLAR


def test_existing_minimal_aletheia_skill_still_validates():
    skill = AletheiaSkill.model_validate(_minimal_skill_payload())

    assert skill.node_id == "node.phase2"
    assert skill.control_nodes == []
    assert skill.identity_scope is None
    assert skill.canonical_mode is None


def test_aletheia_skill_accepts_control_plane_fields():
    payload = _minimal_skill_payload()
    payload.update(
        {
            "identity_scope": _identity_scope_payload(),
            "task_scope": _task_scope_payload(),
            "thought_action_boundary": _thought_action_payload(),
            "control_nodes": [
                {
                    "kind": "boundary",
                    "subtype": "task_boundary",
                    "scope": "task",
                    "activation_condition": "task scope present",
                    "intervention_type": "warn",
                    "confidence": 0.8,
                    "risk": "low",
                },
                _lens_payload(),
            ],
            "governance_version": "control-plane-v1",
            "training_visibility_policy": {"default": "hidden_metadata"},
            "canonical_mode": "scholar",
            "legacy_mode_alias": "theorist",
        }
    )

    skill = AletheiaSkill.model_validate(payload)

    assert skill.identity_scope.compiled_mode is CanonicalMode.SCHOLAR
    assert skill.task_scope.task_id == "task.phase2"
    assert skill.thought_action_boundary.action_state_class == "none"
    assert len(skill.control_nodes) == 2
    assert skill.canonical_mode is CanonicalMode.SCHOLAR
    assert skill.legacy_mode_alias == "theorist"


def test_existing_reasoning_edge_payload_still_validates():
    edge = ReasoningEdge.model_validate(
        {
            "source_id": "source",
            "target_id": "target",
            "edge_type": "dependency",
        }
    )

    assert edge.cross_domain is False
    assert edge.edge_confidence == 0.0
    assert edge.lens_coordinates == []


def test_reasoning_edge_accepts_lens_coordinates():
    edge = ReasoningEdge.model_validate(
        {
            "source_id": "source",
            "target_id": "target",
            "relation_type": "domain_bridge",
            "evidence_basis": "explicit_citation",
            "source_domain": "chemistry",
            "target_domain": "materials",
            "cross_domain": True,
            "edge_confidence": 0.72,
            "bridge_confidence": 0.64,
            "risk_level": "medium",
            "quarantine_status": "none",
            "requires_review": True,
            "lens_coordinates": [_lens_payload()],
            "reason": "Evidence-backed analogy under review.",
        }
    )

    assert edge.cross_domain is True
    assert edge.lens_coordinates[0].lens_id == "lens.false_bridge"
    assert edge.reason == "Evidence-backed analogy under review."


def test_existing_topology_cluster_payload_still_validates():
    cluster = TopologyCluster.model_validate({"cluster_id": "cluster.a"})

    assert cluster.cluster_id == "cluster.a"
    assert cluster.cluster_stage is None
    assert cluster.preserved_contrast_terms == []


def test_topology_cluster_accepts_governed_metadata():
    cluster = TopologyCluster.model_validate(
        {
            "cluster_id": "cluster.a",
            "cluster_stage": "bridge_candidate",
            "merge_basis": "mechanism",
            "merge_confidence": 0.81,
            "preserved_contrast_terms": ["thermal", "solvent"],
            "compressed_from": ["cluster.source_a", "cluster.source_b"],
            "quarantine_reason": "manual review pending",
        }
    )

    assert cluster.cluster_stage == "bridge_candidate"
    assert cluster.merge_basis == "mechanism"
    assert cluster.preserved_contrast_terms == ["thermal", "solvent"]
    assert cluster.compressed_from == ["cluster.source_a", "cluster.source_b"]


def test_invalid_nested_control_plane_fields_reject_when_present():
    payload = _minimal_skill_payload()
    payload["identity_scope"] = {
        **_identity_scope_payload(),
        "training_visibility": "assistant_visible",
    }

    with pytest.raises(ValidationError):
        AletheiaSkill.model_validate(payload)
