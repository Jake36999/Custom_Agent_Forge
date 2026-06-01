"""
Pydantic v2 models for Aletheia Knowledge Compiler Engine
Defines strict schemas for all core domain entities.
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from uuid import uuid4
from pydantic import Field, field_validator, model_validator, computed_field
from src.core.models_contract import BaseAletheiaModel
from src.core.mode_registry import CanonicalMode, normalize_mode
from src.core.meta_reasoning_models import (
    ControlNode,
    IdentityScope,
    LensCoordinate,
    TaskScope,
    ThoughtActionBoundary,
)


class StructuralAlignment(BaseAletheiaModel):
    """
    First-class representation of the 3D alignment vector.
    Never artificially normalized — poorly aligned code mathematically starves.
    Components:
        coverage:    Deterministic subset match of target invariants
        novelty:     Bounded AST diff metric against upstream dependencies
        consistency: Hybrid structural/embedding similarity to base repo
    """
    coverage: float = Field(0.0, ge=0.0, le=1.0, description="Deterministic subset match of target invariants")
    novelty: float = Field(0.0, ge=0.0, le=1.0, description="Bounded AST diff metric against upstream dependencies")
    consistency: float = Field(0.0, ge=0.0, le=1.0, description="Hybrid structural/embedding similarity to base repo")

    @computed_field
    @property
    def magnitude(self) -> float:
        """Unnormalized vector magnitude ||v_align||."""
        return round((self.coverage ** 2 + self.novelty ** 2 + self.consistency ** 2) ** 0.5, 6)

class SemanticReasoningNode(BaseAletheiaModel):
    """
    The fundamental SIE unit of the cognitive architecture.
    Carries content density, structural alignment vector,
    and Information Flow (composite_quality_score / s_sie).
    """
    node_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique SIE node identifier")
    content_density: float = Field(0.0, ge=0.0, le=1.0, validation_alias="rho", description="AST actionability density [0,1]")
    alignment_vector: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], validation_alias="phase_gradient", description="3-vector: [coverage, novelty, consistency]")
    composite_quality_score: float = Field(0.0, ge=0.0, validation_alias="J_info", description="Information Flow (mode_scaling_factor * content_density * ||alignment_vector||)")
    mode_scaling_factor: float = Field(1.0, ge=0.0, validation_alias="kappa", description="Mode-specific scaling factor for Information Flow")
    s_sie: float = Field(0.0, ge=0.0, description="Computed SIE score (= composite_quality_score)")
    blast_radius_threshold: float = Field(default=0.15, ge=0.0, description="Minimum structural impact for DAG propagation")
    temporal_state: Optional[datetime] = Field(default=None, description="Timestamp of last SIE computation")

    @property
    def structural_alignment(self) -> StructuralAlignment:
        """Projects alignment_vector into a StructuralAlignment."""
        g = self.alignment_vector
        return StructuralAlignment(
            coverage=g[0] if len(g) > 0 else 0.0,
            novelty=g[1] if len(g) > 1 else 0.0,
            consistency=g[2] if len(g) > 2 else 0.0,
        )


class IdentityState(BaseAletheiaModel):
    """Tracks the agent's semantic identity across DAG execution."""
    identity_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identity instance identifier")
    role: str = Field(default="Aletheia/LOGOS Principal Architectural Auditor")
    behavior_profile: Dict[str, float] = Field(default_factory=dict)
    mode_weights: Dict[str, float] = Field(default_factory=lambda: {"theorist": 0.25, "coding_assistant": 0.25, "advocate": 0.25, "veteran": 0.25})
    drift_score: float = Field(0.0, ge=0.0)
    stable_hash_reference: str = Field(default="")
    frozen: bool = Field(default=False)
    capabilities: Dict[str, Any] = Field(default_factory=dict, description="Tracked agent capabilities extracted from processed skills")
    constraints: List[Any] = Field(default_factory=list, description="Active identity-level constraints")
    version: int = Field(default=1, ge=1, description="Identity schema version, incremented on trajectory updates")
    compiled_mode: Optional[CanonicalMode] = None
    legacy_mode_aliases: List[str] = Field(default_factory=list)
    role_contract_id: Optional[str] = None
    role_contract_version: int = Field(default=1, ge=1)
    identity_invariants: List[str] = Field(default_factory=list)
    allowed_output_forms: List[str] = Field(default_factory=list)
    prohibited_conflations: List[str] = Field(default_factory=list)

    @field_validator('compiled_mode', mode='before')
    @classmethod
    def normalize_compiled_mode(cls, v):
        if v is None:
            return None
        return normalize_mode(v)


class Constraint(BaseAletheiaModel):
    """
    Strict constraint object for structural, semantic, operational, and epistemic constraints.
    Replaces raw violation strings with a typed schema.
    """
    type: str = Field(..., description="Constraint category: structural | semantic | operational | epistemic")
    description: str = Field(..., min_length=1, description="Human-readable constraint description")
    valid: bool = Field(default=True, description="Whether the constraint is currently satisfied")
    severity: str = Field(default="warning", description="Severity level: info | warning | error | fatal")
    tags: List[str] = Field(default_factory=list, description="Classification tags for filtering")


class TrajectoryStep(BaseAletheiaModel):
    """Single step in a Veteran-mode reasoning trajectory."""
    attempt_code: str = Field(default="", description="Code attempted in this step")
    error: str = Field(default="", description="Error/traceback encountered")
    diagnosis: str = Field(default="", description="Diagnostic analysis of the error")
    fix: str = Field(default="", description="Corrective action or resolved code")


class TrajectoryAuditContext(BaseAletheiaModel):
    """Audit payload linking a trajectory to an Advocate freeze diagnostic."""
    audit_type: str = Field(default="unknown", description="Audit category, e.g., advocate_freeze_diagnostic")
    branch_id: Optional[str] = Field(default=None, description="Branch identifier associated with the freeze event")
    frozen_node: Optional[str] = Field(default=None, description="Node ID that triggered the freeze")
    root_cause: str = Field(default="unknown", description="Primary root-cause classification")
    dominant_breach_mode: Optional[str] = Field(default=None, description="Mode with strongest accumulated drift")
    recommendation: str = Field(default="manual_review", description="Recommended remediation path")
    slr_breach_count: Optional[int] = Field(default=None, ge=0, description="Rolling SLR breach count at freeze time")
    slr_breach_sequence: List[float] = Field(default_factory=list, description="Contiguous SLR breach tail sequence")
    mode_weight_skew: Dict[str, float] = Field(default_factory=dict, description="Delta between current and stable mode weights")
    system_backpressure: Optional[float] = Field(default=None, validation_alias="field_pressure", description="System backpressure at freeze time")
    adversarial_cross_check: Optional[Dict[str, Any]] = Field(default=None, description="Adversarial lens verdict snapshot")
    artifact_path: Optional[str] = Field(default=None, description="Persisted advocate artifact path if available")


class Trajectory(BaseAletheiaModel):
    """
    Veteran-mode reasoning trajectory: serializes the attempt→error→diagnosis→fix sequence.
    Fed into the cosine-similarity sycophancy detector for conviction scoring.
    """
    trajectory_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique trajectory identifier")
    steps: List[TrajectoryStep] = Field(default_factory=list, description="Ordered sequence of diagnostic steps")
    confusion_matrix: Optional[List[List[float]]] = Field(default=None, description="Response signal matrix for sycophancy detection")
    response_vector: Optional[List[float]] = Field(default=None, description="Baseline reference vector for sycophancy detection")
    final_conviction: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Latent conviction after sycophancy analysis")
    labels: List[str] = Field(default_factory=list, description="Classification labels for this trajectory")
    timestamp: Optional[datetime] = Field(default=None, description="Trajectory creation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Auxiliary trajectory metadata")
    audit_context: Optional[TrajectoryAuditContext] = Field(default=None, description="Audit linkage for rejection-driven trajectories")
    failure_matrix_reference: Optional[str] = Field(default=None, description="Stable failure matrix identifier for this trajectory")


class EpistemicState(BaseAletheiaModel):
    state: str = Field(..., description="Epistemic state label")
    c_node: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0,1]")
    confidence: Optional[Dict[str, Any]] = None
    retry_budget: Optional[int] = 6
    depth: Optional[int] = 0
    branch_id: Optional[str] = Field(default="root", description="Branch identifier")
    evidence_refs: Optional[List[Any]] = Field(default_factory=list, description="Supporting evidence references")
    final_status: Optional[str] = Field(default=None, description="Terminal status label")

class TopologyCluster(BaseAletheiaModel):
    cluster_id: Optional[str] = None
    nodes: Optional[List[str]] = None
    edges: Optional[List[Dict[str, Any]]] = None
    child_ids: Optional[List[str]] = None
    node_ids: Optional[List[str]] = None
    parent_ids: Optional[List[str]] = None
    radius: Optional[int] = None
    node_count: Optional[int] = None
    edge_count: Optional[int] = None
    dependency_count: Optional[int] = None
    graph_density: Optional[float] = None
    cluster_stage: Optional[str] = None
    merge_basis: Optional[str] = None
    merge_confidence: float = Field(default=0.0, ge=0.0)
    preserved_contrast_terms: List[str] = Field(default_factory=list)
    compressed_from: List[str] = Field(default_factory=list)
    quarantine_reason: Optional[str] = None


# --- EpistemicReasoning: Strict 5 foundational reasoning vectors ---
class EpistemicReasoning(BaseAletheiaModel):
    intent: Optional[str] = Field(default="Derived execution intent")
    strategy: Optional[str] = Field(default="AST Translation")
    constraints: List[str] = Field(default_factory=list)
    execution_pattern: List[str] = Field(default_factory=list)
    failure_modes: List[str] = Field(default_factory=list)

class TeachingLayer(BaseAletheiaModel):
    skill_identity: Dict[str, Any] = Field(default_factory=dict)
    method_metadata: Dict[str, Any] = Field(default_factory=dict)
    reasoning_vectors: Optional[EpistemicReasoning] = None
    implementation_template: Dict[str, Any] = Field(default_factory=dict)
    heuristics: Optional[Dict[str, Any]] = None
    transformation_delta: Optional[Dict[str, Any]] = None

class AletheiaSkill(BaseAletheiaModel):
    node_id: str
    name: str
    file: str
    code_snippet: str
    imports: List[str]
    operator_type: str
    topology_cluster: Optional[TopologyCluster] = None
    teaching_layer: TeachingLayer
    dependencies: Optional[Dict[str, Any]] = None
    epistemic: Optional[EpistemicState] = None
    semantics: Optional[Dict[str, Any]] = None
    source_context: Optional[str] = None
    documentation_context: Optional[str] = None
    skill_type: Optional[str] = "execution"
    type: Optional[str] = None
    source_type: Optional[str] = "ast_code"
    identity: Optional[Dict[str, Any]] = None
    graph: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    skill_identity: Optional[Dict[str, Any]] = None
    state_and_execution: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    epistemic_core: Optional[Dict[str, Any]] = None
    validation_pass: Optional[bool] = False
    v_score: Optional[float] = 0.0
    # SIE unit for quality/alignment/flow calculations
    sie_node: Optional[SemanticReasoningNode] = None
    # ACS Engine v5.1 tracking fields (injected dynamically by evaluate_node)
    acs_handshake_sid: Optional[str] = None
    acs_violations: Optional[List[str]] = Field(default_factory=list)
    acs_audited: Optional[bool] = False
    identity_scope: Optional[IdentityScope] = None
    task_scope: Optional[TaskScope] = None
    thought_action_boundary: Optional[ThoughtActionBoundary] = None
    control_nodes: List[ControlNode] = Field(default_factory=list)
    governance_version: Optional[str] = None
    training_visibility_policy: Optional[Dict[str, Any]] = None
    canonical_mode: Optional[CanonicalMode] = None
    legacy_mode_alias: Optional[str] = None

    @field_validator('source_context', mode='before')
    @classmethod
    def coerce_source_context(cls, v):
        if isinstance(v, dict):
            import json
            return json.dumps(v)
        return v

    @field_validator('imports', mode='before')
    @classmethod
    def ensure_imports_list(cls, v):
        return v or []

    @field_validator('canonical_mode', mode='before')
    @classmethod
    def normalize_canonical_mode(cls, v):
        if v is None:
            return None
        return normalize_mode(v)

    @model_validator(mode='after')
    def check_operator_type(self):
        if not self.operator_type:
            raise ValueError('operator_type must not be empty')
        return self


class ProjectedNode(BaseAletheiaModel):
    """Strict Layer-3 projection contract at the semantic compiler boundary.

    Architectural posture: "accept the superset, validate the critical subset".
    Layer 3 strictly validates the properties it cares about (operator_type,
    teaching_layer, epistemic, code_snippet, etc.) and safely passes through
    upstream fields (topology_cluster, source_context, dependencies, ...)
    without rejecting them. Upstream layers own their own strict contracts
    (e.g., AletheiaSkill) and Layer-3 must not re-litigate their shape.
    """
    model_config = {"extra": "ignore", "populate_by_name": True, "strict": True}

    node_id: str
    name: str
    file: str
    code_snippet: str
    imports: List[str] = Field(default_factory=list)
    operator_type: str
    teaching_layer: Dict[str, Any]
    epistemic: Dict[str, Any]
    semantics: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    validation: Dict[str, Any] = Field(default_factory=dict)
    constraints: Optional[Dict[str, Any]] = Field(default_factory=dict)
    identity: Optional[Dict[str, Any]] = None
    graph: Optional[Dict[str, Any]] = None
    skill_type: Optional[str] = "theoretical_reasoning"
    source_type: Optional[str] = "ocr_document"
    execution_eligible: bool = True
    invariant_signals: Optional[Dict[str, Any]] = None


class ReasoningEdge(BaseAletheiaModel):
    """
    Typed edge in the cognitive DAG.
    Carries constraint metadata and dependency direction between CognitiveNodes.
    """
    source_id: str = Field(..., description="Source CognitiveNode ID")
    target_id: str = Field(..., description="Target CognitiveNode ID")
    edge_type: str = Field(default="dependency", description="Edge classification: dependency | constraint | evidence")
    constraints: List[Constraint] = Field(default_factory=list, description="Constraints attached to this edge")
    weight: float = Field(default=1.0, ge=0.0, description="Edge weight for PageRank and cascade calculations")
    transformation: str = Field(default="apply_rule", description="Transformation applied along this edge")
    constraint_ok: bool = Field(default=True, description="Whether all constraints on this edge are satisfied")
    slr: float = Field(default=0.0, ge=0.0, description="SIE-SLR value computed across this edge")
    relation_type: Optional[str] = None
    evidence_basis: Optional[str] = None
    source_domain: Optional[str] = None
    target_domain: Optional[str] = None
    cross_domain: bool = False
    edge_confidence: float = Field(default=0.0, ge=0.0)
    bridge_confidence: float = Field(default=0.0, ge=0.0)
    risk_level: Optional[str] = None
    quarantine_status: Optional[str] = None
    requires_review: bool = False
    lens_coordinates: List[LensCoordinate] = Field(default_factory=list)
    reason: Optional[str] = None


class CognitiveNode(BaseAletheiaModel):
    """
    DAG execution wrapper that separates cognitive routing context from the raw AletheiaSkill payload.
    Embeds the SIE unit, manages unified scoring (c_final), and tracks pipeline constraints.
    AletheiaSkill remains strictly focused on the source code payload.

    Spec §1.2: The complete wrapper for execution and scoring.
    """
    cognitive_id: str = Field(..., description="Unique DAG node identifier (may differ from skill.node_id for branched nodes)")
    skill: AletheiaSkill = Field(..., description="The wrapped source-code payload")
    sie_node: SemanticReasoningNode = Field(default_factory=SemanticReasoningNode, description="SIE computation unit")
    trajectory: Optional[Trajectory] = Field(default=None, description="Veteran-mode reasoning trajectory")
    constraints: List[Constraint] = Field(default_factory=list, description="Active constraints on this node")
    inbound_edges: List[str] = Field(default_factory=list, description="IDs of edges feeding into this node")
    outbound_edges: List[str] = Field(default_factory=list, description="IDs of edges leaving this node")
    c_final: float = Field(0.0, ge=0.0, le=1.0, description="Unified confidence: 0.4*s_sie + 0.3*s_acs + 0.2*s_topology + 0.1*s_validation")
    mode: str = Field(default="advocate", description="Active execution mode for this node")
    reasoning_type: str = Field(default="transform", description="Cognitive reasoning type: transform | validate | compose | derive")
    source: str = Field(default="repo", description="Origin source: repo | text | colab")
    acs_score: float = Field(0.0, ge=0.0, le=1.0, description="ACS governance score for this node")
    sie_score: float = Field(0.0, ge=0.0, description="Standalone SIE score (may differ from sie_node.s_sie after weighting)")
    sie_pass: bool = Field(default=True, description="Whether node passed the SIE quality gate")

    @computed_field
    @property
    def information_flow(self) -> float:
        """
        Spec §1.4: Explicit Information Flow (F_info) calculation.
        F_info = mode_scaling_factor * content_density * ||alignment_vector|| * constraint_factor

        Constraint factor gates to 0 if ANY constraint is invalid — hard enforcement.
        """
        alignment = self.sie_node.structural_alignment
        magnitude = alignment.magnitude
        constraint_factor = 1.0 if all(c.valid for c in self.constraints) else 0.0
        return round(self.sie_node.mode_scaling_factor * self.sie_node.content_density * magnitude * constraint_factor, 6)
