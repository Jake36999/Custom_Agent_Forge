import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.models import AletheiaSkill, ReasoningEdge, TopologyCluster
from src.pipeline.dag_runtime import (
    MIN_DOMAIN_BRIDGE_CONFIDENCE,
    TransitionTrigger,
    _detect_training_visibility_leak,
    _detect_thought_action_leak,
    _govern_cluster_promotion,
    _govern_edge_policy,
    _validate_control_plane_fields,
    _validate_lens_attachment,
)


def _minimal_skill_payload() -> dict:
    return {
        "node_id": "node.runtime",
        "name": "runtime_fixture",
        "file": "runtime_fixture.py",
        "code_snippet": "def runtime_fixture():\n    return 1\n",
        "imports": [],
        "operator_type": "function",
        "teaching_layer": {
            "skill_identity": {"name": "runtime_fixture"},
            "method_metadata": {"name": "runtime_fixture", "language": "python"},
            "implementation_template": {"code": "def runtime_fixture():\n    return 1\n"},
        },
        "epistemic": {"state": "CREATED", "c_node": 0.5},
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
        "task_id": "task.runtime",
        "task_class": "runtime_governance",
        "in_scope": ["validate controls"],
        "out_of_scope": ["hard lens rejection"],
        "required_evidence_kinds": ["tests"],
        "abstention_conditions": ["missing evidence"],
    }


def _boundary_node_payload() -> dict:
    return {
        "kind": "boundary",
        "subtype": "task_boundary",
        "scope": "task",
        "activation_condition": "task scope present",
        "intervention_type": "warn",
        "confidence": 0.8,
        "risk": "low",
    }


def _lens_payload(applies_to: str = "edge") -> dict:
    return {
        "kind": "lens",
        "lens_id": "lens.false_bridge",
        "subtype": "bridge_lens",
        "applies_to": applies_to,
        "target_edge_types": ["domain_bridge"],
        "prompt_pattern": "Check whether this bridge has evidence.",
        "activation_condition": "cross_domain == true",
        "intervention_type": "warn",
        "confidence": 0.75,
        "risk": "medium",
    }


def test_valid_node_with_control_fields_passes_boundary_validation():
    payload = _minimal_skill_payload()
    payload.update(
        {
            "identity_scope": _identity_scope_payload(),
            "task_scope": _task_scope_payload(),
            "control_nodes": [_boundary_node_payload()],
        }
    )
    node = AletheiaSkill.model_validate(payload)

    assert _validate_control_plane_fields(node) == []


def test_malformed_identity_scope_fails_boundary_validation():
    node = AletheiaSkill.model_validate(_minimal_skill_payload())
    node.identity_scope = {"compiled_mode": "unknown"}

    issues = _validate_control_plane_fields(node)

    assert issues[0]["trigger"] == TransitionTrigger.BOUNDARY_SCOPE_FAILURE
    assert issues[0]["action"] == "reject"


def test_unauthorised_action_produces_thought_action_leak():
    node = AletheiaSkill.model_validate(_minimal_skill_payload())
    node.thought_action_boundary = {
        "thought_state_class": "candidate_plan",
        "action_state_class": "answer",
        "evidence_threshold_met": False,
        "action_authorised": False,
        "action_justification": "Not enough evidence.",
        "thought_action_separation_score": 0.2,
    }

    issue = _detect_thought_action_leak(node)

    assert issue["trigger"] == TransitionTrigger.THOUGHT_ACTION_LEAK
    assert issue["action"] == "reroll"


def test_free_floating_lens_is_quarantined():
    issue = _validate_lens_attachment(_lens_payload(), owner_kind=None)

    assert issue["trigger"] == TransitionTrigger.FREE_FLOATING_LENS
    assert issue["action"] == "quarantine"


def test_lens_attached_to_edge_does_not_reject_by_itself():
    edge = ReasoningEdge.model_validate(
        {
            "source_id": "source",
            "target_id": "target",
            "lens_coordinates": [_lens_payload("edge")],
        }
    )

    issue = _govern_edge_policy(edge)

    assert issue["action"] == "observe"
    assert issue["trigger"] is None


def test_lexical_similarity_edge_is_quarantined():
    edge = ReasoningEdge.model_validate(
        {
            "source_id": "source",
            "target_id": "target",
            "relation_type": "lexical_similarity",
        }
    )

    issue = _govern_edge_policy(edge)

    assert issue["trigger"] == TransitionTrigger.LEXICAL_EDGE_QUARANTINE
    assert issue["action"] == "quarantine"


def test_low_confidence_cross_domain_edge_is_quarantined():
    edge = ReasoningEdge.model_validate(
        {
            "source_id": "source",
            "target_id": "target",
            "cross_domain": True,
            "edge_confidence": MIN_DOMAIN_BRIDGE_CONFIDENCE - 0.01,
        }
    )

    issue = _govern_edge_policy(edge)

    assert issue["trigger"] == TransitionTrigger.LOW_CONFIDENCE_DOMAIN_BRIDGE
    assert issue["action"] == "quarantine"


def test_domain_bridge_without_evidence_basis_is_quarantined():
    edge = ReasoningEdge.model_validate(
        {
            "source_id": "source",
            "target_id": "target",
            "relation_type": "domain_bridge",
            "edge_confidence": 0.8,
        }
    )

    issue = _govern_edge_policy(edge)

    assert issue["trigger"] == TransitionTrigger.LOW_CONFIDENCE_DOMAIN_BRIDGE
    assert issue["action"] == "quarantine"


def test_control_or_lens_node_cannot_create_bridge():
    edge = ReasoningEdge.model_validate(
        {
            "source_id": "control:lens.false_bridge",
            "target_id": "cluster.target",
            "relation_type": "domain_bridge",
            "evidence_basis": "explicit_citation",
            "edge_confidence": 0.9,
        }
    )

    issue = _govern_edge_policy(edge)

    assert issue["trigger"] == TransitionTrigger.LOW_CONFIDENCE_DOMAIN_BRIDGE
    assert issue["action"] == "reject"


def test_cluster_compression_with_lexical_merge_basis_is_blocked():
    cluster = TopologyCluster.model_validate(
        {
            "cluster_id": "cluster.lexical",
            "cluster_stage": "compressed_invariant",
            "merge_basis": "lexical",
            "merge_confidence": 0.95,
            "preserved_contrast_terms": ["term"],
        }
    )

    issue = _govern_cluster_promotion(cluster)

    assert issue["trigger"] == TransitionTrigger.CLUSTER_COMPRESSION_BLOCKED
    assert issue["action"] == "quarantine"


def test_compressed_invariant_without_preserved_contrast_terms_is_blocked():
    cluster = TopologyCluster.model_validate(
        {
            "cluster_id": "cluster.no_contrast",
            "cluster_stage": "compressed_invariant",
            "merge_basis": "mechanism",
            "merge_confidence": 0.95,
        }
    )

    issue = _govern_cluster_promotion(cluster)

    assert issue["trigger"] == TransitionTrigger.CLUSTER_COMPRESSION_BLOCKED
    assert issue["action"] == "quarantine"


def test_hidden_governance_key_in_output_text_produces_visibility_leak():
    node = AletheiaSkill.model_validate(_minimal_skill_payload())
    node.semantics = {"output": 'This leaks {"identity_scope": {}} into output.'}

    issue = _detect_training_visibility_leak(node)

    assert issue["trigger"] == TransitionTrigger.TRAINING_VISIBILITY_LEAK
    assert issue["action"] == "reject"
