"""Strict control-plane schemas for SIE self-inference and lens metadata.

These models are governance metadata, not executable runtime policy. Phase 1
keeps them isolated so later phases can wire them into models, formatting, and
runtime gates deliberately.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Mapping, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.core.mode_registry import CanonicalMode, normalize_mode
from src.pipeline.branching_contracts import BranchTrace


TrainingVisibility = Literal["hidden_metadata", "prompt_visible", "target_visible"]
RiskLevel = Literal["low", "medium", "high"]

HIDDEN_GOVERNANCE_KEYS = {
    "identity_scope",
    "task_scope",
    "thought_action_boundary",
    "control_nodes",
    "lens_coordinates",
    "training_visibility_policy",
    "source_provenance",
    "_lens_diagnostics",
    "_bridge_risk",
    "_identity_scope_verdict",
    "_task_scope_verdict",
    "_thought_action_violation",
    "_cluster_compression_risk",
}

GOVERNANCE_VISIBILITY_VALUES = {
    "hidden_metadata",
    "prompt_visible",
    "target_visible",
}

COMPACT_AUDIT_SCALARS = {
    "schema_version",
    "canonical_mode",
    "legacy_mode_alias",
    "branch_count",
    "failure_boundary_coverage",
    "max_mechanism_similarity",
    "condition_coverage",
    "bridge_risk_score",
    "lens_pressure_score",
    "thought_action_separation_score",
    "boundary_separation_score",
    "cross_domain_risk_score",
    "task_scope_completeness",
    "abstention_integrity_score",
    "cluster_compression_risk",
}


def is_hidden_governance_key(key: str) -> bool:
    """Return True when a key carries hidden governance metadata."""

    return str(key) in HIDDEN_GOVERNANCE_KEYS


def strip_hidden_governance_fields(record: Mapping[str, Any]) -> dict:
    """Recursively remove hidden governance fields while preserving audit scalars."""

    cleaned: Dict[str, Any] = {}
    for key, value in record.items():
        if is_hidden_governance_key(str(key)):
            continue
        if isinstance(value, Mapping):
            cleaned[key] = strip_hidden_governance_fields(value)
        elif isinstance(value, list):
            cleaned[key] = [
                strip_hidden_governance_fields(item)
                if isinstance(item, Mapping)
                else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def assert_no_hidden_governance_in_text(text: str) -> None:
    """Reject assistant-visible text that names or serializes hidden governance fields."""

    text_s = str(text)
    for key in sorted(HIDDEN_GOVERNANCE_KEYS):
        if key in text_s:
            raise ValueError(f"hidden governance metadata leaked into text: {key}")


def assert_no_hidden_governance_in_messages(messages: list[dict]) -> None:
    """Reject ChatML messages that expose hidden governance metadata."""

    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        assert_no_hidden_governance_in_text(str(content))


class StrictControlModel(BaseModel):
    """Strict base for governance objects where silent extras are unsafe."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        populate_by_name=True,
    )


class IdentityScope(StrictControlModel):
    compiled_mode: CanonicalMode
    legacy_mode_alias: Optional[str] = None
    role_contract_id: str
    role_contract_version: int = Field(..., ge=1)
    stable_self_hash: str
    identity_invariants: List[str] = Field(default_factory=list)
    allowed_output_forms: List[str] = Field(default_factory=list)
    prohibited_conflations: List[str] = Field(default_factory=list)
    allowed_action_classes: List[str] = Field(default_factory=list)
    training_visibility: TrainingVisibility = "hidden_metadata"

    @field_validator("compiled_mode", mode="before")
    @classmethod
    def normalize_compiled_mode(cls, value: Any) -> CanonicalMode:
        return normalize_mode(value)


class TaskScope(StrictControlModel):
    task_id: str
    task_class: str
    in_scope: List[str] = Field(default_factory=list)
    out_of_scope: List[str] = Field(default_factory=list)
    required_evidence_kinds: List[str] = Field(default_factory=list)
    abstention_conditions: List[str] = Field(default_factory=list)
    escalation_policy: str = "manual_review"
    training_visibility: TrainingVisibility = "hidden_metadata"


class ThoughtActionBoundary(StrictControlModel):
    thought_state_class: Literal[
        "hypothesis",
        "candidate_plan",
        "uncertainty_assessment",
        "selected_reasoning",
    ]
    action_state_class: Literal["none", "answer", "patch", "reroll", "reject", "escalate"]
    evidence_threshold_met: bool
    action_authorised: bool
    action_justification: str
    thought_action_separation_score: float = Field(..., ge=0.0, le=1.0)
    training_visibility: TrainingVisibility = "hidden_metadata"


class BoundaryNode(StrictControlModel):
    kind: Literal["boundary"]
    subtype: Literal[
        "identity_boundary",
        "role_boundary",
        "task_boundary",
        "thought_action_boundary",
    ]
    scope: Literal["global", "mode", "task", "node", "edge", "cluster"]
    activation_condition: str
    intervention_type: Literal["observe", "warn", "reroute", "quarantine", "reroll", "reject"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk: RiskLevel
    identity_scope: Optional[IdentityScope] = None
    task_scope: Optional[TaskScope] = None
    thought_action_boundary: Optional[ThoughtActionBoundary] = None
    training_visibility: TrainingVisibility = "hidden_metadata"


class LensCoordinate(StrictControlModel):
    kind: Literal["lens"]
    lens_id: str
    subtype: Literal["bias_lens", "uncertainty_lens", "adversarial_lens", "bridge_lens"]
    applies_to: Literal["node", "edge", "cluster"]
    target_edge_types: List[str] = Field(default_factory=list)
    prompt_pattern: str
    activation_condition: str
    intervention_type: Literal["observe", "warn", "reroute", "quarantine", "reroll"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk: RiskLevel
    training_visibility: TrainingVisibility = "hidden_metadata"


class TopologyBridgeNode(StrictControlModel):
    kind: Literal["bridge"]
    bridge_id: str
    subtype: Literal[
        "mechanism_bridge",
        "analogy_bridge",
        "contrast_bridge",
        "failure_mode_bridge",
        "shared_math_bridge",
    ]
    source_cluster_id: str
    target_cluster_id: str
    relation_type: str
    evidence_basis: str
    edge_confidence: float = Field(..., ge=0.0, le=1.0)
    cross_domain: bool
    requires_review: bool = False
    quarantine_status: Literal["none", "quarantined", "approved"] = "none"
    training_visibility: TrainingVisibility = "hidden_metadata"


ControlNode = Annotated[
    Union[BoundaryNode, LensCoordinate, TopologyBridgeNode],
    Field(discriminator="kind"),
]


class GovernedRALRecordV21(StrictControlModel):
    """Governed RAL v2.1 outer envelope with an unchanged BranchTrace v2.0 body."""

    schema_version: Literal["2.1"]
    branch_trace: BranchTrace
    training_visibility_policy: Dict[str, Any]
    final_answer: str
    identity_scope: Optional[IdentityScope] = None
    task_scope: Optional[TaskScope] = None
    thought_action_boundary: Optional[ThoughtActionBoundary] = None
    control_nodes: List[ControlNode] = Field(default_factory=list)
    source_provenance: Optional[Dict[str, Any]] = None
    domain_tags: List[str] = Field(default_factory=list)
    reasoning_tags: List[str] = Field(default_factory=list)
    ambiguity_tags: List[str] = Field(default_factory=list)
