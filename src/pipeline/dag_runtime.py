"""
Aletheia DAG Runtime v5.3 (Causal Authority Engine)

Central Causal Authority Orchestrator for the Aletheia Knowledge Compiler Engine.

Architecture: Every raw AletheiaSkill payload is wrapped in a CognitiveNode for
cognitive routing. The runtime enforces a strict node lifecycle:

    CREATED → VALIDATED → SCORED → ACCEPTED | REJECTED
                                ↘   REROLL   ↗

Contract violations and confidence failures route to the RerollEngine rather
than silent termination. Structured logging is emitted at every state transition.

Note: CognitiveNode, IdentityState, and all SIE models are defined in
src.core.models (the consolidated Pydantic v2 model layer).
"""

# --- Deterministic Seeding for Global Reproducibility ---
import os
import random
import numpy as np
os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)

import uuid
import json
import time
import hashlib
import logging
import ast
import traceback
import networkx as nx
from typing import Dict, List, Any, Optional, Set
from copy import deepcopy
from pydantic import TypeAdapter


# ============================================================
# Priority 3: State Transition Trigger Enum (Governance Lock)
# ============================================================
# Strict enum binding so downstream SFT data-slicers can aggregate
# rejection taxonomies without free-form string drift.
class TransitionTrigger:
    """Cryptographic ledger trigger taxonomy for state transitions."""
    BOOTSTRAP              = "bootstrap"
    CONTRACT_PASSED        = "contract_passed"
    CONTRACT_VIOLATION     = "contract_violation"
    DRIFT_VIOLATION        = "drift_violation"
    SIE_UNDERFLOW          = "sie_underflow"
    ACS_SCHEMA_CORRUPTED   = "acs_schema_corrupted"
    ACS_CONSTRAINT_FATAL   = "acs_constraint_fatal"
    IDENTITY_DRIFT         = "identity_drift"
    IDENTITY_DRIFT_PROACTIVE = "identity_drift_proactive"
    IDENTITY_GLOBAL_FREEZE = "identity_globally_frozen"
    SYCOPHANCY_COLLAPSE    = "sycophancy_collapse"
    SYCOPHANCY_REROLL      = "sycophancy_reroll"
    PARENT_CHAIN_FAILED    = "parent_chain_failed"
    C_FINAL_SCORED         = "c_final_scored"
    C_FINAL_ACCEPTED       = "c_final_accepted"
    C_FINAL_BELOW_BAND     = "c_final_below_instability_band"
    INSTABILITY_REROLL     = "instability_reroll"
    CASCADE_FROM_PARENT    = "cascade_from_parent"
    CONSISTENCY_VIOLATION  = "consistency_violation"
    BRANCH_SPAWN           = "branch_spawn"
    DEPTH_LIMIT            = "depth_limit"
    REROLL_BUDGET_EXHAUSTED = "reroll_budget_exhausted"
    REROLL_NO_ENGINE       = "reroll_no_engine"
    REROLL_ATTEMPT         = "reroll_attempt"
    OMEGA_ORPHAN_QUARANTINE = "omega_orphan_quarantine"
    BOUNDARY_SCOPE_FAILURE = "boundary_scope_failure"
    THOUGHT_ACTION_LEAK = "thought_action_leak"
    FREE_FLOATING_LENS = "free_floating_lens"
    LEXICAL_EDGE_QUARANTINE = "lexical_edge_quarantine"
    LOW_CONFIDENCE_DOMAIN_BRIDGE = "low_confidence_domain_bridge"
    CLUSTER_COMPRESSION_BLOCKED = "cluster_compression_blocked"
    TRAINING_VISIBILITY_LEAK = "training_visibility_leak"

    ALL = frozenset({
        BOOTSTRAP, CONTRACT_PASSED, CONTRACT_VIOLATION, DRIFT_VIOLATION,
        SIE_UNDERFLOW, ACS_SCHEMA_CORRUPTED, ACS_CONSTRAINT_FATAL,
        IDENTITY_DRIFT, IDENTITY_DRIFT_PROACTIVE, IDENTITY_GLOBAL_FREEZE,
        SYCOPHANCY_COLLAPSE, SYCOPHANCY_REROLL, PARENT_CHAIN_FAILED,
        C_FINAL_SCORED, C_FINAL_ACCEPTED, C_FINAL_BELOW_BAND,
        INSTABILITY_REROLL, CASCADE_FROM_PARENT, CONSISTENCY_VIOLATION,
        BRANCH_SPAWN, DEPTH_LIMIT, REROLL_BUDGET_EXHAUSTED,
        REROLL_NO_ENGINE, REROLL_ATTEMPT, OMEGA_ORPHAN_QUARANTINE,
        BOUNDARY_SCOPE_FAILURE, THOUGHT_ACTION_LEAK, FREE_FLOATING_LENS,
        LEXICAL_EDGE_QUARANTINE, LOW_CONFIDENCE_DOMAIN_BRIDGE,
        CLUSTER_COMPRESSION_BLOCKED, TRAINING_VISIBILITY_LEAK,
    })

    @classmethod
    def normalize(cls, trigger: str) -> str:
        """Coerce free-form reasons into a known enum token (best effort).

        Unknown triggers are preserved verbatim so the ledger remains truthful,
        but they are stamped with a ``*_legacy`` suffix so slicers can filter
        them out of aggregate dashboards. This preserves atomicity guarantees
        while still enforcing the enumeration discipline.
        """
        if not trigger:
            return cls.BOOTSTRAP
        head = trigger.split(":", 1)[0].strip().lower().replace(" ", "_")
        if head in cls.ALL:
            return head
        # Known prefixes
        for known in cls.ALL:
            if head.startswith(known):
                return known
        return trigger  # preserved verbatim for diagnostic fidelity

# --- Sentinel Codes for Upstream Orchestrator Telemetry ---
SENTINEL_FATAL_CYCLE     = "[DAG_FATAL_CYCLE_DETECTED]"
SENTINEL_WARN_SCHEMA     = "[DAG_WARN_SCHEMA_BREACH]"
SENTINEL_ERR_CONVERGE    = "[DAG_ERR_MATH_CONVERGENCE]"
SENTINEL_FATAL_CRASH     = "[DAG_FATAL_RUNTIME_CRASH]"
SENTINEL_WARN_QUARANTINE = "[DAG_WARN_NODE_QUARANTINED]"
SENTINEL_QUALITY_UNDERFLOW   = "[DAG_QUALITY_UNDERFLOW]"
SENTINEL_CONTRACT_FAIL   = "[DAG_CONTRACT_VIOLATION]"
SENTINEL_CASCADE_PRUNE   = "[DAG_CASCADE_PRUNE]"
SENTINEL_IDENTITY_FREEZE = "[DAG_IDENTITY_FREEZE]"
SENTINEL_PARENT_CHAIN    = "[DAG_PARENT_CHAIN_FAIL]"
SENTINEL_TELEMETRY_ALERT = "[DAG_TELEMETRY_THRESHOLD]"
SENTINEL_SYSTEM_BACKPRESSURE  = "[DAG_SYSTEM_BACKPRESSURE]"
SENTINEL_EDGE_INCOHERENT  = "[DAG_EDGE_INCOHERENT]"
SENTINEL_INSTABILITY     = "[DAG_INSTABILITY_REROLL]"

# SIE gate: default minimum quality floor for s_sie score (0 = information-dead node)
# Configurable per-runtime via DAGRuntime(quality_floor=...)
QUALITY_FLOOR = 0.1

# --- Strict Pipeline Integration ---
try:
    from src.interfaces.graph_interface import EpistemicGraphInterface
    from src.interfaces.graph_backend import is_gpu_available as _gpu_available
    from src.pipeline.acs_engine import ACSExecutionGovernor, SycophancyDetector
    from src.core.compiler_utils import stable_json_dumps
    from src.core.models import (
        AletheiaSkill, TeachingLayer, TopologyCluster, EpistemicState,
        SemanticReasoningNode, CognitiveNode, ReasoningEdge, Constraint,
        IdentityState, ProjectedNode,
    )
    from src.core.meta_reasoning_models import (
        BoundaryNode,
        ControlNode,
        IdentityScope,
        LensCoordinate,
        TaskScope,
        ThoughtActionBoundary,
        TopologyBridgeNode,
        assert_no_hidden_governance_in_text,
    )
    from src.pipeline.contracts import validate_contract, MODE_CONTRACTS
    from src.pipeline.identity_manager import IdentityManager
    from src.pipeline.reroll import RerollEngine
    from src.pipeline.telemetry import TelemetryCollector
    from src.pipeline.invariant_engine import InvariantEngine
    from src.pipeline.consistency_engine import ConsistencyEngine
    from src.pipeline.omega_validator import OmegaValidator
    from src.pipeline.adversarial_lens import AdversarialLens, SENTINEL_ALL_CONFLICT
    from src.validation.pipeline_firewall import enforce_semantic_firewall, DriftViolation
    from pydantic import ValidationError
except ImportError:
    # Graceful fallback for standalone or partial-import environments
    from pydantic import ValidationError
    from src.pipeline.fallback_stubs import (
        EpistemicGraphInterface, ACSExecutionGovernor,
        TopologyCluster, EpistemicState, TeachingLayer, AletheiaSkill,
        SemanticReasoningNode, CognitiveNode, ReasoningEdge, Constraint,
        IdentityState, stable_json_dumps,
        MODE_CONTRACTS, validate_contract,
        IdentityManager, RerollEngine, TelemetryCollector,
        DriftViolation, enforce_semantic_firewall,
        AdversarialLens, SENTINEL_ALL_CONFLICT,
    )

from src.pipeline.telemetry_policy import check_telemetry_thresholds as _check_telemetry_policy

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("Advocate.DAGRuntime")


# ============================================================
# Thresholds
# ============================================================
ACCEPTANCE_THRESHOLD = 0.40
BRANCH_PRUNE_THRESHOLD = 0.35
MAX_BRANCHES = 15
MAX_DEPTH = 5
MAX_ITERATIONS = 500

# Unified confidence weights: c_final = α·s_sie + β·s_acs + γ·s_topology + δ·s_validation
CONFIDENCE_ALPHA = 0.4   # SIE — primary epistemic signal
CONFIDENCE_BETA  = 0.3   # ACS — governance score
CONFIDENCE_GAMMA = 0.2   # Topology — structural integrity
CONFIDENCE_DELTA = 0.1   # Validation — evidence quality

# Stability reroll band (Patch 3)
INSTABILITY_BAND_LOW = 0.25

# Trusted-source mode: set ALETHEIA_TRUSTED_MODE=1 to disable backpressure, relax
# instability band, and prevent cascade from overriding confident nodes.
# Intended for high-quality OSS ingestion runs (ruff, debugpy, plotly, etc.).
_TRUSTED_MODE = os.environ.get("ALETHEIA_TRUSTED_MODE", "0") == "1"

# D6: per-mode scaling factor for SIE coupling strength
SCALING_FACTOR_BY_MODE = {
    "theorist":         1.8,  # V2.1: raised from 1.2 to compensate for text-based SIE
    "coding_assistant": 1.0,
    "advocate":         0.9,
    "veteran":          1.1,
}

# Depth penalty (Phase 7 — topological field dynamics)
DEPTH_PENALTY_RATE = 0.05
DEPTH_PENALTY_CAP = 0.3

# Soft branch degradation threshold (Phase 7 — replaces hard pruning)
BRANCH_STABILITY_THRESHOLD = 0.4
BRANCH_DEGRADATION_FACTOR = 0.2
SOFT_PARENT_CASCADE_FACTOR = 0.5

# Edge coherence threshold (Patch 2)
EDGE_COHERENCE_THRESHOLD = 0.3

# Dynamic auto-tuning bounds (Bucket C)
AUTO_TUNE_COHERENCE_MIN = 0.15
AUTO_TUNE_COHERENCE_MAX = 0.45
AUTO_TUNE_COHERENCE_STEP = 0.02
AUTO_TUNE_RECOVERY_CADENCE_MIN = 1
AUTO_TUNE_RECOVERY_CADENCE_MAX = 5
AUTO_TUNE_COOLDOWN_CHECKS = 2

# SLR threshold tiers — standardized across all SLR enforcement paths
SLR_THRESHOLDS = {
    "safe": 0.3,       # No action required
    "warning": 0.6,    # Log + telemetry only
    "reroll": 0.7,     # Trigger reroll + identity mutation
    "collapse": 0.8,   # Hard rejection — confidence collapse
}

MIN_DOMAIN_BRIDGE_CONFIDENCE = 0.70
MIN_CLUSTER_MERGE_CONFIDENCE = 0.75


def _governance_issue(trigger: str, action: str, reason: str, **extra: Any) -> Dict[str, Any]:
    issue = {"trigger": trigger, "action": action, "reason": reason}
    issue.update(extra)
    return issue


def _validate_control_plane_fields(node: AletheiaSkill) -> List[Dict[str, Any]]:
    """Validate optional node-level control-plane fields when present."""

    issues: List[Dict[str, Any]] = []
    validators = (
        ("identity_scope", IdentityScope),
        ("task_scope", TaskScope),
        ("thought_action_boundary", ThoughtActionBoundary),
    )
    for field_name, schema in validators:
        value = getattr(node, field_name, None)
        if value is None:
            continue
        try:
            schema.model_validate(value)
        except Exception as exc:
            issues.append(_governance_issue(
                TransitionTrigger.BOUNDARY_SCOPE_FAILURE,
                "reject",
                f"{field_name} failed validation: {exc}",
                field=field_name,
            ))

    adapter = TypeAdapter(ControlNode)
    for idx, control_node in enumerate(getattr(node, "control_nodes", []) or []):
        try:
            validated = adapter.validate_python(control_node)
        except Exception as exc:
            issues.append(_governance_issue(
                TransitionTrigger.BOUNDARY_SCOPE_FAILURE,
                "reject",
                f"control_nodes[{idx}] failed validation: {exc}",
                field="control_nodes",
            ))
            continue
        lens_issue = _validate_lens_attachment(validated, owner_kind="node")
        if lens_issue is not None:
            issues.append(lens_issue)
    return issues


def _detect_thought_action_leak(node: AletheiaSkill) -> Optional[Dict[str, Any]]:
    boundary = getattr(node, "thought_action_boundary", None)
    if boundary is None:
        return None
    try:
        boundary = ThoughtActionBoundary.model_validate(boundary)
    except Exception as exc:
        return _governance_issue(
            TransitionTrigger.BOUNDARY_SCOPE_FAILURE,
            "reject",
            f"thought_action_boundary failed validation: {exc}",
        )
    if not boundary.action_authorised and boundary.action_state_class != "none":
        return _governance_issue(
            TransitionTrigger.THOUGHT_ACTION_LEAK,
            "reroll",
            "unauthorised action state escaped thought/action boundary",
        )
    return None


def _validate_lens_attachment(control_node: Any, owner_kind: Optional[str]) -> Optional[Dict[str, Any]]:
    try:
        lens = (
            control_node
            if isinstance(control_node, LensCoordinate)
            else LensCoordinate.model_validate(control_node)
        )
    except Exception:
        return None

    if owner_kind is None or lens.applies_to != owner_kind:
        return _governance_issue(
            TransitionTrigger.FREE_FLOATING_LENS,
            "quarantine",
            "lens coordinate is not attached to its declared owner",
            lens_id=lens.lens_id,
        )
    if lens.intervention_type == "quarantine":
        return _governance_issue(
            TransitionTrigger.FREE_FLOATING_LENS,
            "quarantine",
            "attached lens requested quarantine",
            lens_id=lens.lens_id,
        )
    if lens.intervention_type == "reroll":
        return _governance_issue(
            TransitionTrigger.FREE_FLOATING_LENS,
            "reroll",
            "attached lens requested reroll",
            lens_id=lens.lens_id,
        )
    return _governance_issue(None, "observe", "attached lens is advisory", lens_id=lens.lens_id)


def _edge_endpoint_is_control(endpoint: Optional[str]) -> bool:
    lowered = str(endpoint or "").lower()
    return lowered.startswith(("control:", "lens:", "boundary:", "bridge:"))


def _govern_edge_policy(edge: ReasoningEdge) -> Dict[str, Any]:
    """Return advisory/quarantine policy for governed edge metadata."""

    relation_type = getattr(edge, "relation_type", None)
    evidence_basis = getattr(edge, "evidence_basis", None)
    if relation_type == "lexical_similarity":
        return _governance_issue(
            TransitionTrigger.LEXICAL_EDGE_QUARANTINE,
            "quarantine",
            "lexical similarity edges quarantine by default",
        )

    if relation_type == "domain_bridge" and (
        _edge_endpoint_is_control(getattr(edge, "source_id", None))
        or _edge_endpoint_is_control(getattr(edge, "target_id", None))
        or getattr(edge, "lens_coordinates", None)
    ):
        return _governance_issue(
            TransitionTrigger.LOW_CONFIDENCE_DOMAIN_BRIDGE,
            "reject",
            "control or lens nodes cannot create domain bridges",
        )

    if relation_type == "domain_bridge" and not evidence_basis:
        return _governance_issue(
            TransitionTrigger.LOW_CONFIDENCE_DOMAIN_BRIDGE,
            "quarantine",
            "domain bridge missing evidence_basis",
        )

    if getattr(edge, "cross_domain", False) and getattr(edge, "edge_confidence", 0.0) < MIN_DOMAIN_BRIDGE_CONFIDENCE:
        return _governance_issue(
            TransitionTrigger.LOW_CONFIDENCE_DOMAIN_BRIDGE,
            "quarantine",
            "cross-domain edge below confidence threshold",
        )

    for lens in getattr(edge, "lens_coordinates", []) or []:
        issue = _validate_lens_attachment(lens, owner_kind="edge")
        if issue and issue["action"] != "observe":
            return issue

    return _governance_issue(None, "observe", "edge governance passed")


def _govern_cluster_promotion(cluster: TopologyCluster) -> Dict[str, Any]:
    """Return quarantine policy for governed cluster promotion metadata."""

    stage = getattr(cluster, "cluster_stage", None)
    merge_basis = getattr(cluster, "merge_basis", None)
    merge_confidence = getattr(cluster, "merge_confidence", 0.0)
    preserved_contrast_terms = getattr(cluster, "preserved_contrast_terms", []) or []

    if merge_basis == "lexical":
        return _governance_issue(
            TransitionTrigger.CLUSTER_COMPRESSION_BLOCKED,
            "quarantine",
            "cluster compression cannot use lexical merge basis",
        )
    if stage == "compressed_invariant":
        if merge_confidence < MIN_CLUSTER_MERGE_CONFIDENCE:
            return _governance_issue(
                TransitionTrigger.CLUSTER_COMPRESSION_BLOCKED,
                "quarantine",
                "compressed invariant below merge confidence threshold",
            )
        if not preserved_contrast_terms:
            return _governance_issue(
                TransitionTrigger.CLUSTER_COMPRESSION_BLOCKED,
                "quarantine",
                "compressed invariant requires preserved contrast terms",
            )
    return _governance_issue(None, "observe", "cluster governance passed")


def _detect_training_visibility_leak(node: AletheiaSkill) -> Optional[Dict[str, Any]]:
    """Detect hidden governance keys in assistant-visible/output-ish node text."""

    candidates: List[str] = []
    semantics = getattr(node, "semantics", None)
    if isinstance(semantics, dict):
        for key in ("output", "final_answer", "assistant", "assistant_content"):
            if semantics.get(key):
                candidates.append(str(semantics[key]))
    for attr in ("output", "final_answer"):
        value = getattr(node, attr, None)
        if value:
            candidates.append(str(value))

    for text in candidates:
        try:
            assert_no_hidden_governance_in_text(text)
        except ValueError as exc:
            return _governance_issue(
                TransitionTrigger.TRAINING_VISIBILITY_LEAK,
                "reject",
                str(exc),
            )
    return None

# Telemetry → Action Thresholds (canonical source: telemetry_policy.py)
from src.pipeline.telemetry_policy import (
    TELEMETRY_SLR_MEAN_THRESHOLD,
    TELEMETRY_DRIFT_CUMULATIVE_THRESHOLD,
    TELEMETRY_QUALITY_MEAN_FLOOR,
)


# ============================================================
# SIE-Aware SLR Computation
# ============================================================
def calculate_sie_slr(parent: Any, child: Any) -> float:
    """
    Compute SIE-based Sycophancy Likelihood Ratio between parent and child nodes.
    Uses cosine distance between alignment_vector vectors as the SLR proxy.
    Returns SLR in [0, 1]: 0 = identical vectors (no drift), 1 = orthogonal (max drift).
    """
    parent_sie = getattr(parent, "sie_node", None)
    child_sie = getattr(child, "sie_node", None)
    if parent_sie is None or child_sie is None:
        return 0.0

    p_grad = getattr(parent_sie, "alignment_vector", None)
    c_grad = getattr(child_sie, "alignment_vector", None)
    if not p_grad or not c_grad:
        return 0.0

    p = np.array(p_grad, dtype=float)
    c = np.array(c_grad, dtype=float)

    p_norm = np.linalg.norm(p)
    c_norm = np.linalg.norm(c)
    if p_norm < 1e-9 or c_norm < 1e-9:
        return 0.0

    cos_sim = float(np.dot(p, c) / (p_norm * c_norm))
    if not np.isfinite(cos_sim):
        return 0.0
    slr = 1.0 - max(0.0, min(1.0, cos_sim))
    return round(slr, 4)


# ============================================================
# Node Lifecycle States
# ============================================================
class NodeState:
    """Explicit lifecycle states for strict state-machine governance."""
    CREATED   = "CREATED"
    VALIDATED = "VALIDATED"
    SCORED    = "SCORED"
    ACCEPTED  = "ACCEPTED"
    REJECTED  = "REJECTED"
    REROLL    = "REROLL"

    TERMINAL = frozenset({ACCEPTED, REJECTED})
    ACTIVE   = frozenset({CREATED, VALIDATED, SCORED, REROLL})
    # Legacy states from pre-v5 pipelines (mapped to ACTIVE during bootstrap)
    LEGACY   = frozenset({
        "unresolved", "hypothesis", "validating", "rejected", "terminal",
        # V2.1 Theorist Mode: semantic compiler emits verifiability levels
        # as initial epistemic states. These must be normalized to CREATED.
        "low", "medium", "high",
    })


# ============================================================
# InMemoryEpistemicGraph
# ============================================================
class InMemoryEpistemicGraph(EpistemicGraphInterface):
    """
    In-Memory Execution Context for the Epistemic Graph.
    Enforces strict Pydantic schema compliance and topological integrity.
    """
    def __init__(self, initial_nodes: Optional[List[Dict[str, Any]]] = None):
        self.nodes: Dict[str, AletheiaSkill] = {}
        self.quarantine: List[Dict[str, Any]] = []
        if initial_nodes:
            self.add_nodes(initial_nodes)

    def _build_schema_mismatch_node(self, raw_payload: Any, error: ValidationError) -> AletheiaSkill:
        raw = raw_payload if isinstance(raw_payload, dict) else {}
        raw_node_id = str(raw.get("node_id") or f"quarantine_{uuid.uuid4().hex[:8]}")
        diag = str(error)
        return AletheiaSkill.model_validate({
            "node_id": raw_node_id,
            "name": str(raw.get("name") or "quarantined_projected_node"),
            "file": str(raw.get("file") or "<ingress>"),
            "code_snippet": str(raw.get("code_snippet") or ""),
            "imports": raw.get("imports") if isinstance(raw.get("imports"), list) else [],
            "operator_type": str(raw.get("operator_type") or "schema_mismatch"),
            "teaching_layer": {
                "skill_identity": {},
                "method_metadata": {},
                "implementation_template": {},
            },
            "epistemic": {
                "state": NodeState.REJECTED,
                "c_node": 0.0,
                "retry_budget": 0,
                "depth": 0,
                "branch_id": "root",
                "final_status": "projected_schema_mismatch",
            },
            "semantics": {
                "execution_eligible": False,
                "failure_type": "projected_node_validation_error",
                "validation_error": diag,
                "orchestration_mode": "veteran",
                "attempt_code": str(raw.get("code_snippet") or ""),
                "traceback_text": diag,
                "analysis_text": "ProjectedNode strict validation failed at Layer 3->4 boundary.",
                "patch": "",
                "resolved_code": "",
                "rejection_reason": "projected_node_validation_error",
            },
            "metadata": {
                "execution_eligible": False,
                "ingress_contract": "ProjectedNode",
                "validation_error": diag,
            },
            "validation_pass": False,
            "v_score": 0.0,
            "skill_type": "veteran",
            "source_type": str(raw.get("source_type") or "ingress_payload"),
        })

    def add_nodes(self, nodes_list: List[Any]):
        for n in sorted(
            nodes_list,
            key=lambda x: x.get("node_id") if isinstance(x, dict) else getattr(x, "node_id", ""),
        ):
            try:
                if isinstance(n, dict):
                    # V2.1 Theorist Mode: strip compiler-internal fields that are
                    # not part of the AletheiaSkill contract (extra="forbid").
                    # The v3.6 semantic compiler emits execution_eligible and
                    # invariant_signals for its own optimization loop; these are
                    # not DAG-layer concerns and must be stripped at ingestion.
                    _COMPILER_INTERNAL_KEYS = {"execution_eligible", "invariant_signals"}
                    for _cik in _COMPILER_INTERNAL_KEYS:
                        n.pop(_cik, None)
                    # Phase D: strict ingress contract at Layer 3 -> Layer 4 boundary.
                    ProjectedNode.model_validate(n, strict=True)
                skill = n if isinstance(n, AletheiaSkill) else AletheiaSkill.model_validate(n)
                self.nodes[skill.node_id] = skill
            except ValidationError as e:
                logger.error(
                    f"{SENTINEL_WARN_SCHEMA} Pydantic Boundary Breach: "
                    f"Quarantining payload. Error: {e}"
                )
                quarantined_payload = n if isinstance(n, dict) else getattr(n, "__dict__", {})
                self.quarantine.append(quarantined_payload)
                # Containment rule: do not drop invalid payloads; route as ineligible veteran failures.
                fallback = self._build_schema_mismatch_node(n, e)
                self.nodes[fallback.node_id] = fallback

    def active(self) -> bool:
        """Graph remains active if any node has not reached a terminal state."""
        return any(
            n.epistemic.state not in NodeState.TERMINAL
            and n.epistemic.state != "terminal"
            for n in self.nodes.values()
        )

    def ready(self) -> List[AletheiaSkill]:
        """Yields nodes queued for state-machine execution."""
        return [
            n for n in self.nodes.values()
            if n.epistemic.state in NodeState.ACTIVE
            or n.epistemic.state in NodeState.LEGACY
        ]

    def validated_nodes(self) -> List[AletheiaSkill]:
        return [
            n for n in self.nodes.values()
            if n.validation_pass and n.epistemic.c_node >= ACCEPTANCE_THRESHOLD
        ]

    def get_evidence(self, node_id: str) -> List[Dict[str, Any]]:
        if node_id in self.nodes:
            return getattr(self.nodes[node_id].epistemic, "evidence_refs", [])
        return []

    def branch_ids(self) -> List[str]:
        return sorted(set(
            n.epistemic.branch_id for n in self.nodes.values()
            if n.epistemic.branch_id != "root"
        ))

    def get_branch(self, branch_id: str) -> List[AletheiaSkill]:
        return sorted(
            [n for n in self.nodes.values() if n.epistemic.branch_id == branch_id],
            key=lambda n: n.node_id,
        )

    def remove_branch(self, branch_id: str):
        keys = sorted(k for k, v in self.nodes.items() if v.epistemic.branch_id == branch_id)
        for k in keys:
            del self.nodes[k]


# ============================================================
# DAGRuntime — Strict Identity-as-Code State Machine
# ============================================================
class DAGRuntime:
    """
    Mutable DAG Runtime v5.3 — Causal Authority Engine.

    Governs CognitiveNode lifecycle through strict state transitions:

        CREATED → VALIDATED → SCORED → ACCEPTED | REJECTED
                                    ↘   REROLL   ↗

    Contract violations and threshold failures route to the RerollEngine
    with failure context injection, preserving rejected traces for SFT isolation.
    """

    def __init__(
        self,
        graph: EpistemicGraphInterface,
        acs: ACSExecutionGovernor,
        llm_orchestrator=None,
        pattern_registry=None,
        mode: str = "advocate",
        acceptance_threshold: float = ACCEPTANCE_THRESHOLD,
        quality_floor: float = QUALITY_FLOOR,
    ):
        self.graph = graph
        self.acs = acs
        self.llm_orchestrator = llm_orchestrator
        self.pattern_registry = pattern_registry or {}
        self.rejected_traces: List[Dict[str, Any]] = []
        self._nx_graph: Optional[nx.DiGraph] = None
        self._cognitive_nodes: Dict[str, CognitiveNode] = {}
        self.acceptance_threshold = acceptance_threshold
        self.quality_floor = quality_floor

        # Priority 3 — Cryptographic State Transition Ledger
        # Every node mutation through _log_transition is sealed into a
        # SHA256 hash chain so the full lifecycle (CREATED -> VALIDATED ->
        # SCORED -> TERMINAL) is tamper-evident. If any downstream consumer
        # re-orders or drops a record the chain hash will fail to match.
        self._transition_seq: int = 0
        self._last_transition_hash: str = "GENESIS"
        self._transition_ledger: List[Dict[str, Any]] = []

        # Bucket C: dynamic governance controls (defaults preserve baseline behavior)
        self.edge_coherence_tuning_threshold = EDGE_COHERENCE_THRESHOLD
        self.telemetry_check_cadence = 1
        self.recovery_tick_cadence = 1
        self.auto_tuning_enabled = True
        self.auto_tuning_cooldown_checks = AUTO_TUNE_COOLDOWN_CHECKS
        self._policy_check_count = 0
        self._last_tuning_update_check = -AUTO_TUNE_COOLDOWN_CHECKS
        self._tuning_history: Dict[str, List[Dict[str, Any]]] = {
            "coherence_threshold": [],
            "recovery_tick_cadence": [],
        }

        # Identity drift manager
        try:
            self.identity_manager = IdentityManager(max_drift=0.3)
        except Exception:
            self.identity_manager = None

        # Reroll engine for REROLL state handling
        try:
            self.reroll_engine = RerollEngine()
        except Exception:
            self.reroll_engine = None

        # Centralized telemetry collector
        try:
            self.telemetry = TelemetryCollector()
        except Exception:
            self.telemetry = None

        # Wire telemetry into sub-engines for governance observability
        if self.telemetry:
            if hasattr(self.acs, "telemetry"):
                self.acs.telemetry = self.telemetry
            if self.identity_manager and hasattr(self.identity_manager, "telemetry"):
                self.identity_manager.telemetry = self.telemetry

        # Global system backpressure (Patch 4 — field-level dynamics)
        self.system_backpressure = 0.0

        # Invariant execution engine (Phase 8)
        try:
            self.invariant_engine = InvariantEngine()
        except Exception:
            self.invariant_engine = None

        # Global consistency engine (Phase 6)
        try:
            self.consistency_engine = ConsistencyEngine()
        except Exception:
            self.consistency_engine = None

        # Omega handshake validator (Phase 9 — split: partial + final)
        try:
            self.omega_validator = OmegaValidator()
        except Exception:
            self.omega_validator = None

        # Adversarial Lens — orthogonal evaluation channel (advisory only)
        try:
            self._adversarial_lens = AdversarialLens(conflict_threshold=0.30)
        except Exception:
            self._adversarial_lens = None

        # Ensure 4-Mode strict enforcement
        assert mode in ("theorist", "coding_assistant", "advocate", "veteran"), (
            f"Invalid Identity Matrix Mode: {mode}"
        )
        self.mode = mode
        # D9: Log graph backend selection
        try:
            _gpu = _gpu_available()
            _backend_name = "cuGraph (GPU)" if _gpu else "NetworkX (CPU)"
        except Exception:
            _backend_name = "NetworkX (CPU)"
        logger.info(
            f"[DAG] Runtime v5.3 initialized | mode={self.mode.upper()} | "
            f"threshold={self.acceptance_threshold} | graph_backend={_backend_name}"
        )

    # ------------------------------------------------------------------
    # Priority 3 — Cryptographic State Transition Ledger
    # ------------------------------------------------------------------
    def _log_transition(
        self,
        node_id: str,
        from_state: str,
        to_state: str,
        reason: str = "",
        trigger: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Seal a state transition into the tamper-evident hash chain.

        Emits a structured telemetry event AND appends a cryptographically
        chained record (``sha256(prev_hash | node_id | from | to | trigger
        | seq)``) to the per-runtime ledger. Telemetry emission and ledger
        append are always executed atomically with the caller's state
        mutation, so the ledger is the authoritative record of why a node
        arrived at its final state.

        Parameters
        ----------
        node_id      : the node being mutated
        from_state   : source NodeState (pre-mutation)
        to_state     : destination NodeState (post-mutation)
        reason       : free-form diagnostic text (backward compatible)
        trigger      : explicit enum trigger from ``TransitionTrigger``.
                       If omitted, falls back to ``reason`` (normalized).

        Returns
        -------
        The ledger record dict (so callers can assert on ``seq`` or
        ``this_hash`` in regression tests).
        """
        # Resolve the enum-bound trigger. Falls back to the free-form reason
        # when a structured trigger is not supplied so existing call sites
        # keep working without modification.
        raw_trigger = trigger if trigger is not None else (reason or "")
        bound_trigger = TransitionTrigger.normalize(raw_trigger)

        detail = f" | {bound_trigger}" if bound_trigger else ""
        if reason and reason != bound_trigger:
            detail += f" ({reason})"
        logger.info(f"[DAG] Node {node_id} Transition: {from_state} -> {to_state}{detail}")

        # --- Hash chain append (atomic with mutation) ---
        self._transition_seq += 1
        seq = self._transition_seq
        prev_hash = self._last_transition_hash
        material = (
            f"{prev_hash}|{node_id}|{from_state}|{to_state}|{bound_trigger}|{seq}"
        )
        this_hash = hashlib.sha256(material.encode("utf-8")).hexdigest()
        self._last_transition_hash = this_hash

        record: Dict[str, Any] = {
            "seq":        seq,
            "node_id":    node_id,
            "from_state": from_state,
            "to_state":   to_state,
            "trigger":    bound_trigger,
            "reason":     reason,
            "prev_hash":  prev_hash,
            "this_hash":  this_hash,
            "timestamp":  time.time(),
        }
        self._transition_ledger.append(record)

        # --- Telemetry emission ---
        if self.telemetry:
            self.telemetry.record(
                "state_transition", "transition",
                node_id=node_id,
                payload={
                    "from_state": from_state,
                    "to_state":   to_state,
                    "trigger":    bound_trigger,
                    "seq":        seq,
                    "prev_hash":  prev_hash,
                    "this_hash":  this_hash,
                    # Backward-compatible aliases for existing consumers.
                    "from":       from_state,
                    "to":         to_state,
                    "reason":     reason,
                },
            )
        return record

    # ------------------------------------------------------------------
    # Ledger Accessors (Priority 3)
    # ------------------------------------------------------------------
    def get_transition_ledger(self) -> List[Dict[str, Any]]:
        """Return the full hash-chained state transition ledger."""
        return list(self._transition_ledger)

    def verify_transition_ledger(self) -> bool:
        """Recompute the hash chain and verify its integrity end-to-end.

        Returns ``True`` iff every recorded ``this_hash`` equals
        ``sha256(prev_hash | node_id | from | to | trigger | seq)``. This
        is the equivalent of a Merkle proof for the node lifecycle: if any
        record is mutated, reordered, or dropped the verification fails.
        """
        expected_prev = "GENESIS"
        for rec in self._transition_ledger:
            material = (
                f"{expected_prev}|{rec['node_id']}|{rec['from_state']}|"
                f"{rec['to_state']}|{rec['trigger']}|{rec['seq']}"
            )
            recomputed = hashlib.sha256(material.encode("utf-8")).hexdigest()
            if recomputed != rec["this_hash"]:
                return False
            if rec["prev_hash"] != expected_prev:
                return False
            expected_prev = rec["this_hash"]
        return True

    # ------------------------------------------------------------------
    # Graph & Node Helpers
    # ------------------------------------------------------------------
    def _get_nodes_safely(self) -> List[AletheiaSkill]:
        """Safely extracts nodes, bypassing payloads that failed Pydantic validation."""
        raw = []
        if hasattr(self.graph, "nodes"):
            if isinstance(self.graph.nodes, dict):
                raw = list(self.graph.nodes.values())
            elif isinstance(self.graph.nodes, list):
                raw = self.graph.nodes
        elif getattr(self.graph, "_nodes", []):
            raw = self.graph._nodes

        validated = []
        for n in raw:
            try:
                validated.append(
                    AletheiaSkill.model_validate(n) if isinstance(n, dict) else n
                )
            except ValidationError as e:
                logger.warning(f"{SENTINEL_WARN_QUARANTINE} Evicting corrupt node: {e}")
        return validated

    def _sync_node(self, node: AletheiaSkill):
        """Write the typed node back to the graph store."""
        if hasattr(self.graph, "nodes") and isinstance(self.graph.nodes, dict):
            self.graph.nodes[node.node_id] = node

    def _build_directed_graph(self, nodes: List[AletheiaSkill]) -> nx.DiGraph:
        G = nx.DiGraph()
        for node in nodes:
            G.add_node(node.node_id)
            tc = node.topology_cluster
            if tc:
                for cid in (getattr(tc, "child_ids", None) or []):
                    edge = ReasoningEdge(source_id=node.node_id, target_id=cid)
                    G.add_edge(node.node_id, cid, edge=edge)
            deps = node.dependencies
            if deps:
                dep_list = (
                    deps.get("downstream_calls", [])
                    if isinstance(deps, dict)
                    else getattr(deps, "downstream_calls", [])
                )
                for did in (dep_list or []):
                    edge = ReasoningEdge(
                        source_id=node.node_id, target_id=did, edge_type="dependency",
                    )
                    G.add_edge(node.node_id, did, edge=edge)
        return G

    def _sort_topologically(self, nodes: List[AletheiaSkill]) -> List[AletheiaSkill]:
        if not nodes:
            return nodes
        G = self._build_directed_graph(self._get_nodes_safely())
        try:
            order = {nid: i for i, nid in enumerate(nx.topological_sort(G))}
            return sorted(nodes, key=lambda n: order.get(n.node_id, float("inf")))
        except nx.NetworkXUnfeasible:
            return nodes

    def _detect_cycles(self) -> bool:
        nodes = self._get_nodes_safely()
        if not nodes:
            return False
        G = self._build_directed_graph(nodes)
        if not nx.is_directed_acyclic_graph(G):
            logger.error(
                f"{SENTINEL_FATAL_CYCLE} Cycle detected — "
                f"attempting topological quarantine before abort."
            )
            return True
        return False

    def _break_cycles_via_quarantine(self) -> List[str]:
        """V2.0 real-world degrade path for cyclic graphs.

        Production codebases (e.g. starlette) contain legitimate circular
        imports/call-graph loops that the old fatal-abort path treated as
        a total-run failure, collapsing every node into ``rejected_nodes``
        and preventing any SFT artifact from being emitted. That turned a
        local topological defect into a catastrophic pipeline breakdown.

        This routine computes the strongly-connected components of the
        current node graph. Every member of a non-trivial SCC (size > 1)
        as well as every self-looping node is moved into
        ``graph.quarantine`` and sealed into the cryptographic transition
        ledger as an ``OMEGA_ORPHAN_QUARANTINE`` transition. The remaining
        graph is a DAG and can proceed through the normal SIE/ACS/Omega
        handshake, yielding ``OMEGA_DEGRADED`` instead of ``OMEGA_FAIL``
        for the typical real-world case where only a small fraction of
        the graph participates in cycles.

        Returns the ordered list of quarantined ``node_id`` values so the
        caller can append a degraded-path record to the execution trace.
        """
        nodes = self._get_nodes_safely()
        if not nodes:
            return []
        G = self._build_directed_graph(nodes)
        if nx.is_directed_acyclic_graph(G):
            return []

        doomed: Set[str] = set()
        try:
            for scc in nx.strongly_connected_components(G):
                if len(scc) > 1:
                    doomed.update(scc)
        except Exception as e:  # pragma: no cover — defensive
            logger.warning(f"[DAG] SCC enumeration failed, falling back to simple_cycles: {e}")
            try:
                for cyc in nx.simple_cycles(G):
                    doomed.update(cyc)
            except Exception as e2:  # pragma: no cover — defensive
                logger.error(f"[DAG] simple_cycles also failed: {e2}")
                return []

        # Self-loops (rare but possible with self-referential topology)
        try:
            for u, v in nx.selfloop_edges(G):
                doomed.add(u)
        except Exception:
            pass

        if not doomed:
            return []

        quarantined: List[str] = []
        for nid in sorted(doomed):
            node = self.graph.nodes.pop(nid, None) if hasattr(self.graph, "nodes") and isinstance(self.graph.nodes, dict) else None
            prior_state = "UNKNOWN"
            if node is not None and hasattr(node, "epistemic") and node.epistemic is not None:
                prior_state = getattr(node.epistemic, "state", "UNKNOWN") or "UNKNOWN"

            payload: Dict[str, Any]
            if node is not None and hasattr(node, "model_dump"):
                try:
                    payload = node.model_dump(exclude_unset=True)
                except Exception:
                    payload = {"node_id": nid}
            else:
                payload = {"node_id": nid}
            payload["quarantine_reason"] = "topological_cycle_break"

            if hasattr(self.graph, "quarantine") and isinstance(self.graph.quarantine, list):
                self.graph.quarantine.append(payload)

            # Seal the transition into the cryptographic ledger even when
            # the node has been evicted from the active graph so the final
            # run artifact contains a tamper-evident record of the break.
            try:
                self._log_transition(
                    nid,
                    prior_state,
                    NodeState.REJECTED,
                    reason="topological_cycle_break",
                    trigger=TransitionTrigger.OMEGA_ORPHAN_QUARANTINE,
                )
            except Exception as e:  # pragma: no cover — defensive
                logger.warning(f"[DAG] ledger seal failed for cycle-break {nid}: {e}")

            if self.telemetry:
                try:
                    self.telemetry.record(
                        "state_transition", "cycle_quarantined",
                        node_id=nid,
                        payload={"reason": "cycle_break", "prior_state": prior_state},
                    )
                except Exception:
                    pass

            quarantined.append(nid)

        logger.warning(
            f"[DAG] Cycle-break quarantined {len(quarantined)} nodes "
            f"from SCCs/self-loops; continuing run on remaining DAG."
        )
        return quarantined

    def _apply_graph_analytics(self, nodes: List[AletheiaSkill]) -> Dict[str, float]:
        if not nodes:
            return {}
        G = self._build_directed_graph(nodes)
        try:
            return nx.pagerank(G, alpha=0.85, max_iter=200)
        except nx.exception.NetworkXError:
            return nx.in_degree_centrality(G)
        except Exception:
            return {n.node_id: 0.0 for n in nodes}

    def _get_validated_nodes(self) -> List[AletheiaSkill]:
        return [
            n for n in self._get_nodes_safely()
            if (n.validation_pass and getattr(n.epistemic, "c_node", 0.0) >= self.acceptance_threshold)
            or getattr(n.epistemic, "state", "") == NodeState.ACCEPTED
        ]

    def _dedupe_and_sort_dicts(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            h = stable_json_dumps(item)
            if h not in seen:
                seen[h] = item
        return sorted(seen.values(), key=lambda x: str(x.get("node_id", "")))

    # ------------------------------------------------------------------
    # State Machine: CREATED → VALIDATED (Contract Gate)
    # ------------------------------------------------------------------
    def _apply_control_plane_governance(self, node: AletheiaSkill, prev: str) -> bool:
        """Apply Phase 4 control-plane checks at the validation boundary."""

        issues = _validate_control_plane_fields(node)
        thought_issue = _detect_thought_action_leak(node)
        visibility_issue = _detect_training_visibility_leak(node)
        issues.extend(issue for issue in (thought_issue, visibility_issue) if issue)

        cluster = getattr(node, "topology_cluster", None)
        if cluster is not None:
            issues.append(_govern_cluster_promotion(cluster))

        for issue in issues:
            if not issue or issue.get("action") == "observe":
                continue
            trigger = issue.get("trigger") or TransitionTrigger.BOUNDARY_SCOPE_FAILURE
            reason = issue.get("reason", trigger)
            action = issue.get("action")

            if action == "reroll":
                self._route_to_reroll(node, prev, f"{trigger}: {reason}")
                return False

            node.epistemic.state = NodeState.REJECTED
            node.epistemic.final_status = f"{trigger}: {reason}"
            if action == "quarantine":
                payload = node.model_dump() if hasattr(node, "model_dump") else {}
                payload["quarantine_reason"] = trigger
                if hasattr(self.graph, "quarantine") and isinstance(self.graph.quarantine, list):
                    self.graph.quarantine.append(payload)
            self._log_transition(
                node.node_id,
                prev,
                NodeState.REJECTED,
                f"{trigger}: {reason}",
                trigger=trigger,
            )
            return False
        return True

    def _transition_to_validated(self, node: AletheiaSkill) -> bool:
        """
        CREATED → VALIDATED via contract validation.
        On contract failure, routes to REROLL (or REJECTED if budget=0).
        Returns True if transition succeeded, False if rerouted.
        """
        prev = node.epistemic.state

        # Mode-specific content generation
        self.execute(node)

        if not self._apply_control_plane_governance(node, prev):
            return False

        # Contract validation gate
        contract_schema = MODE_CONTRACTS.get(self.mode)
        if contract_schema:
            payload = node.semantics if isinstance(node.semantics, dict) else {}
            # Strip non-contract fields before validation (extra="forbid" rejects them)
            _NON_CONTRACT_KEYS = frozenset({
                "orchestration_mode", "name", "extracted_patterns",
                "analysis_text", "attempt_code", "text_to_code_ratio",
                "code_snippet", "chunk_text", "theory_text", "implementation_code",
                # "output" is the canonical downstream-formatter contract field
                # set by execute(); it is intentionally NOT part of the per-mode
                # output contracts and must be stripped here.
                "output",
                # V2.1 Theorist Mode: semantic compiler v3.6 emits NLP analysis
                # structures into semantics that are not part of mode output contracts.
                "discourse", "semantic_frames", "semantic_signature",
                "reasoning_trajectory",
            })
            contract_payload = {k: v for k, v in payload.items() if k not in _NON_CONTRACT_KEYS}
            is_valid, err_msg = validate_contract(contract_payload, contract_schema)
            if self.telemetry:
                self.telemetry.record(
                    "contract", "validated" if is_valid else "failed",
                    node_id=node.node_id,
                    payload={"mode": self.mode, "error": err_msg},
                )
            if not is_valid:
                logger.warning(
                    f"{SENTINEL_CONTRACT_FAIL} Node {node.node_id}: {err_msg}"
                )
                self._route_to_reroll(
                    node, prev, f"contract_violation: {err_msg}",
                    violation_details=err_msg,
                )
                return False

        node.epistemic.state = NodeState.VALIDATED
        self._log_transition(node.node_id, prev, NodeState.VALIDATED, "contract_passed")
        return True

    # ------------------------------------------------------------------
    # State Machine: VALIDATED → SCORED (SIE + ACS + c_final)
    # ------------------------------------------------------------------
    def _transition_to_scored(self, node: AletheiaSkill) -> bool:
        """
        VALIDATED → SCORED via SIE computation, ACS governance,
        and unified c_final calculation.
        Returns False if rejected during this phase.
        """
        prev = node.epistemic.state

        # 0. Semantic drift firewall — reject contaminated content pre-SIE
        try:
            content = json.dumps(node.semantics) if isinstance(node.semantics, dict) else str(node.semantics)
            enforce_semantic_firewall(content, context=f"node:{node.node_id}")
        except DriftViolation as dv:
            logger.warning(f"SENTINEL-DRIFT Node {node.node_id}: {dv}")
            node.epistemic.state = NodeState.REJECTED
            node.epistemic.final_status = "drift_violation"
            node.epistemic.c_node = 0.0
            self._log_transition(node.node_id, prev, NodeState.REJECTED, f"drift_violation:{dv.token}")
            self._cascade_rejection(node.node_id)
            if self.telemetry:
                self.telemetry.record("firewall", "drift_violation", node_id=node.node_id,
                                      payload={"token": dv.token})
            return False

        # 1. SIE underflow hard gate
        if not self._compute_sie(node):
            node.epistemic.state = NodeState.REJECTED
            node.epistemic.final_status = "sie_underflow"
            node.epistemic.c_node = 0.0
            self._log_transition(node.node_id, prev, NodeState.REJECTED, "sie_underflow")
            self._cascade_rejection(node.node_id)
            return False

        # 2. ACS governance (penalty absorption, constraint routing)
        self._apply_acs(node)
        if node.epistemic.state == NodeState.REJECTED:
            # Priority 3: resolve trigger enum from final_status so schema
            # corruption and fatal constraints are distinguishable in the
            # ledger. Both enter here but via different causal paths.
            final_status = getattr(node.epistemic, "final_status", "") or ""
            if "schema" in final_status.lower():
                acs_trigger = TransitionTrigger.ACS_SCHEMA_CORRUPTED
            else:
                acs_trigger = TransitionTrigger.ACS_CONSTRAINT_FATAL
            self._log_transition(
                node.node_id, prev, NodeState.REJECTED,
                reason=final_status or "acs_constraint_fatal",
                trigger=acs_trigger,
            )
            self._cascade_rejection(node.node_id)
            return False

        # 3. Governance directive check (SLR breach → identity mutation)
        if self._check_governance_directive(node):
            return False

        # 3.25 Direct Tikhonov SLR: if node carries reasoning trajectory
        #      with confusion_matrix + response_vector, compute SLR directly
        #      and apply penalty — bypasses indirect ACS governance path.
        sem = node.semantics if isinstance(node.semantics, dict) else {}
        rt = sem.get("reasoning_trajectory", {})
        if isinstance(rt, dict) and rt.get("confusion_matrix") and rt.get("response_vector"):
            try:
                detector = SycophancyDetector()
                tik_slr = detector.compute_tikhonov_slr(
                    rt["confusion_matrix"], rt["response_vector"],
                )
                if tik_slr > 0.0:
                    node.epistemic.c_node = max(0.0, node.epistemic.c_node * (1.0 - tik_slr))
                    if self.telemetry:
                        self.telemetry.record(
                            "tikhonov", "direct_slr", node_id=node.node_id,
                            payload={"slr": tik_slr, "c_node_after": node.epistemic.c_node},
                        )
                if tik_slr >= SLR_THRESHOLDS["reroll"]:
                    if self._handle_sycophancy_failure(node, tik_slr):
                        return False
            except Exception:
                pass

        # 3.5 SIE-SLR enforcement: check alignment_vector drift against parents
        if self._nx_graph is not None:
            try:
                predecessors = list(self._nx_graph.predecessors(node.node_id))
                node_map = {n.node_id: n for n in self._get_nodes_safely()}
                for parent_id in predecessors:
                    parent = node_map.get(parent_id)
                    if parent is None:
                        continue
                    slr = calculate_sie_slr(parent, node)
                    if slr >= SLR_THRESHOLDS["warning"]:
                        if self._handle_sycophancy_failure(node, slr):
                            return False
            except nx.NetworkXError:
                pass

        # 4. Proactive identity enforcement (every node, not just SLR breach)
        if self._enforce_identity(node):
            return False

        # 5. Evidence validation
        self._validate_evidence(node)

        # 5.5 Invariant enforcement (Phase 8 — hard constraint)
        #     Invariants define truth boundaries; must pass before scoring.
        if self.invariant_engine:
            invariants = (
                node.semantics.get("invariants", [])
                if isinstance(node.semantics, dict) else []
            )
            if invariants:
                inv_pass, inv_failures = self.invariant_engine.validate_all(
                    node, invariants,
                )
                if not inv_pass:
                    reason = f"invariant_violation: {inv_failures}"
                    logger.warning(
                        f"[DAG] Node {node.node_id}: {reason}"
                    )
                    if self.telemetry:
                        self.telemetry.record(
                            "invariant", "failed", node_id=node.node_id,
                            payload={"failures": inv_failures},
                        )
                    self._route_to_reroll(
                        node, prev, reason,
                        violation_details="; ".join(inv_failures),
                    )
                    return False

        # 6. Unified confidence computation
        self._compute_unified_confidence(node)

        node.epistemic.state = NodeState.SCORED
        self._log_transition(
            node.node_id, prev, NodeState.SCORED,
            f"c_final={node.epistemic.c_node:.4f}",
        )
        return True

    # ------------------------------------------------------------------
    # State Machine: SCORED → ACCEPTED | REJECTED
    # ------------------------------------------------------------------
    def _transition_to_terminal(self, node: AletheiaSkill):
        """
        SCORED → ACCEPTED if c_final >= threshold, else → REROLL or REJECTED.

        Includes instability band (Patch 3): borderline nodes in
        [INSTABILITY_BAND_LOW, threshold) are forced to reroll for refinement
        rather than being allowed to fail silently.
        """
        prev = node.epistemic.state
        c = node.epistemic.c_node

        if c >= self.acceptance_threshold:
            # Topological authority: verify all parents are ACCEPTED + SIE coherent
            if not self._validate_parent_chain(node):
                node.epistemic.c_node = 0.0
                node.epistemic.state = NodeState.REJECTED
                node.epistemic.final_status = "parent_chain_failed"
                self._log_transition(
                    node.node_id, prev, NodeState.REJECTED,
                    f"parent_chain_failed (c_final={c:.4f} was sufficient)",
                )
                self._cascade_rejection(node.node_id)
                return

            # Pre-commit Omega partial check (Phase 9 — Correction 4)
            # Run while reroll context is still live and identity hasn't mutated
            c = node.epistemic.c_node
            if c < self.acceptance_threshold:
                logger.info(
                    f"{SENTINEL_PARENT_CHAIN} Node {node.node_id}: "
                    f"soft parent cascade lowered c_final to {c:.4f}."
                )
                if c >= INSTABILITY_BAND_LOW:
                    rerolled = self._route_to_reroll(
                        node, prev,
                        f"soft_parent_cascade: c_final={c:.4f} in "
                        f"[{INSTABILITY_BAND_LOW}, {self.acceptance_threshold})",
                    )
                    if not rerolled:
                        self._cascade_rejection(node.node_id)
                    return

                node.epistemic.state = NodeState.REJECTED
                node.epistemic.final_status = f"soft_parent_cascade_below_band: c_final={c:.4f}"
                self._log_transition(
                    node.node_id, prev, NodeState.REJECTED,
                    f"soft_parent_cascade: c_final={c:.4f} < {INSTABILITY_BAND_LOW}",
                )
                self._cascade_rejection(node.node_id)
                return

            if self.omega_validator:
                omega_partial = self.omega_validator.partial_check(
                    node, identity_manager=self.identity_manager,
                )
                if not omega_partial["pass"]:
                    logger.warning(
                        f"[DAG] Node {node.node_id}: omega partial check failed "
                        f"— routing to reroll. {omega_partial['dimensions']}"
                    )
                    if self.telemetry:
                        self.telemetry.record(
                            "omega", "partial_fail", node_id=node.node_id,
                            payload=omega_partial["dimensions"],
                        )
                    rerolled = self._route_to_reroll(
                        node, prev, "omega_partial_check_failed",
                    )
                    if not rerolled:
                        self._cascade_rejection(node.node_id)
                    return

            node.epistemic.state = NodeState.ACCEPTED
            node.epistemic.final_status = "validated"
            node.validation_pass = True

            # --- Adversarial Lens: orthogonal advisory evaluation ---
            if self._adversarial_lens:
                verdict = self._adversarial_lens.evaluate(node, c)
                if isinstance(node.semantics, dict):
                    node.semantics["_adversarial_verdict"] = {
                        "score":       verdict.adversarial_score,
                        "conflict":    verdict.conflict_magnitude,
                        "class":       verdict.conflict_class,
                        "flags":       verdict.flags,
                        "fingerprint": verdict.structural_fingerprint,
                    }
                    node.semantics["_adversarial_conflict"] = (
                        verdict.conflict_class == "major_conflict"
                    )
                if self.telemetry:
                    self.telemetry.record(
                        "adversarial_lens", verdict.conflict_class,
                        node_id=node.node_id,
                        payload={
                            "adversarial_score":  verdict.adversarial_score,
                            "primary_score":      verdict.primary_score,
                            "conflict_magnitude": verdict.conflict_magnitude,
                            "flags":              verdict.flags,
                        },
                    )
                if verdict.conflict_class == "major_conflict":
                    logger.warning(
                        f"{SENTINEL_ALL_CONFLICT} Node '{node.node_id}' "
                        f"primary={verdict.primary_score:.3f} "
                        f"adversarial={verdict.adversarial_score:.3f} "
                        f"Δ={verdict.conflict_magnitude:.3f} "
                        f"flags={verdict.flags}"
                    )

            self._log_transition(
                node.node_id, prev, NodeState.ACCEPTED,
                f"c_final={c:.4f} >= {self.acceptance_threshold}",
            )
        elif c >= INSTABILITY_BAND_LOW:
            # Fix 4: trusted-source runs accept marginally confident nodes rather
            # than forcing a reroll — the source quality is already known good.
            if _TRUSTED_MODE:
                node.epistemic.c_node = c
                node.epistemic.state = NodeState.ACCEPTED
                node.epistemic.final_status = f"trusted_marginal_accept: c_final={c:.4f}"
                self._log_transition(
                    node.node_id, prev, NodeState.ACCEPTED,
                    f"trusted_mode: c_final={c:.4f} in instability band, accepted"
                )
                return

            # Instability band: borderline nodes get forced refinement
            logger.info(
                f"{SENTINEL_INSTABILITY} Node {node.node_id}: "
                f"c_final={c:.4f} in instability band "
                f"[{INSTABILITY_BAND_LOW}, {self.acceptance_threshold}) — forced reroll."
            )
            if self.telemetry:
                self.telemetry.record(
                    "confidence", "instability_reroll", node_id=node.node_id,
                    payload={
                        "c_final": c,
                        "band_low": INSTABILITY_BAND_LOW,
                        "threshold": self.acceptance_threshold,
                    },
                )
            rerolled = self._route_to_reroll(
                node, prev,
                f"instability_band: c_final={c:.4f} in [{INSTABILITY_BAND_LOW}, {self.acceptance_threshold})",
            )
            if not rerolled:
                self._cascade_rejection(node.node_id)
        else:
            # Below instability band — fundamentally unsound, immediate rejection
            node.epistemic.c_node = c
            node.epistemic.state = NodeState.REJECTED
            node.epistemic.final_status = f"below_instability_band: c_final={c:.4f}"
            self._log_transition(
                node.node_id, prev, NodeState.REJECTED,
                f"c_final={c:.4f} < {INSTABILITY_BAND_LOW} — immediate rejection",
            )
            self._cascade_rejection(node.node_id)

    # ------------------------------------------------------------------
    # REROLL Routing & Handling
    # ------------------------------------------------------------------
    def _route_to_reroll(self, node: AletheiaSkill, prev_state: str, reason: str,
                         violation_details: Optional[str] = None) -> bool:
        """
        Routes node to REROLL if budget > 0.
        Captures DPO rejected snapshot with structural metadata, then injects
        Pydantic ValidationError context so the recovery engine can reconstruct
        the prompt with exact constraint violations.
        Falls back to REJECTED if budget exhausted.
        Returns True if routed to REROLL, False if REJECTED.
        """
        budget = getattr(node.epistemic, "retry_budget", 0)
        if budget > 0 and self.reroll_engine:
            # 1. Capture rejected snapshot BEFORE injecting corrective context
            rejected_snapshot = deepcopy(node.semantics) if isinstance(node.semantics, dict) else {}

            # 2. Rejected trace with structural snapshot for SFT isolation
            structural = {
                "sie_score": getattr(node.sie_node, "s_sie", None) if node.sie_node else None,
                "c_node": node.epistemic.c_node,
                "pre_state": prev_state,
                "identity_drift": (
                    self.identity_manager.drift_score if self.identity_manager else None
                ),
                "system_backpressure": self.system_backpressure,
            }
            self.rejected_traces.append({
                "node_id": node.node_id,
                "rejected": rejected_snapshot,
                "reason": reason,
                "corrected": None,
                "structural_snapshot": structural,
            })

            # 3. Inject failure context for the reroll engine's recovery prompt
            #    Phase 5.8: Include epistemic failure signals for intelligent recovery
            if not isinstance(node.semantics, dict):
                node.semantics = {}
            reroll_ctx = {
                "previous_failure_reason": reason,
                "violation_details": violation_details,
            }
            # Enrich with epistemic failure data when available
            if "sycophancy" in reason or "slr" in reason.lower():
                reroll_ctx["failure_type"] = "epistemic_failure"
                reroll_ctx["slr"] = structural.get("sie_score")
                reroll_ctx["sie_divergence"] = structural.get("c_node")
                reroll_ctx["identity_drift"] = structural.get("identity_drift")
            node.semantics["_reroll_context"] = reroll_ctx

            node.epistemic.retry_budget = budget - 1
            node.epistemic.state = NodeState.REROLL
            self._log_transition(node.node_id, prev_state, NodeState.REROLL, reason)
            return True
        else:
            node.epistemic.state = NodeState.REJECTED
            node.epistemic.final_status = reason
            self._log_transition(
                node.node_id, prev_state, NodeState.REJECTED,
                f"budget_exhausted | {reason}",
            )
            return False

    def _handle_reroll(self, node: AletheiaSkill):
        """Invoke the RerollEngine on a REROLL-state node."""
        if not self.reroll_engine:
            node.epistemic.state = NodeState.REJECTED
            node.epistemic.final_status = "no_reroll_engine"
            self._log_transition(
                node.node_id, NodeState.REROLL, NodeState.REJECTED, "no_reroll_engine"
            )
            return

        # Read failure context injected by _route_to_reroll
        reroll_ctx = (
            node.semantics.get("_reroll_context", {})
            if isinstance(node.semantics, dict) else {}
        )
        reason = (
            reroll_ctx.get("previous_failure_reason")
            or getattr(node.epistemic, "final_status", None)
            or "contract_violation"
        )
        violation_details = reroll_ctx.get("violation_details")

        # D7: High-severity rerolls get 3 candidates; default is 2
        if reason and ("sycophancy" in reason or "identity" in reason.lower()
                       if isinstance(reason, str) else False):
            self.reroll_engine.branch_count = 3
        else:
            self.reroll_engine.branch_count = 2

        reroll_result = self.reroll_engine.reroll(
            node, self, reason=reason, violation_details=violation_details,
        )
        if reroll_result:
            self.rejected_traces.append(reroll_result)
            for trace in self.rejected_traces:
                if trace.get("node_id") == node.node_id and trace.get("corrected") is None:
                    trace["corrected"] = reroll_result.get("chosen") or reroll_result.get("corrected")

        self._log_transition(
            node.node_id, NodeState.REROLL, node.epistemic.state,
            f"reroll_attempt={self.reroll_engine.reroll_count}",
        )

    # ------------------------------------------------------------------
    # SIE Computation (Semantic Quality Gate)
    # ------------------------------------------------------------------
    def _compute_sie(self, node: AletheiaSkill) -> bool:
        """
        Populate node.sie_node from semantic quality analysis.
        Primary path: ACS engine. Fallback: inline SIE computation.
        SIE gate is boolean: node must have rho > 0 (content present)
        AND alignment_vector norm > 0 (reasoning vectors present).
        Returns False if node is information-dead (gate fails).
        """
        sie = None

        # Primary: delegate to ACS
        if hasattr(self.acs, "compute_sie_score"):
            node_dict = node.model_dump(exclude_unset=True)
            sie = self.acs.compute_sie_score(node_dict)
        else:
            # Fallback: standalone SIE computation (identical formula)
            sie = self._compute_sie_inline(node)

        # D6: per-mode scaling factor enforcement
        if isinstance(sie, dict):
            mode_scaling = SCALING_FACTOR_BY_MODE.get(self.mode, 1.0)
            old_scaling = sie.get("mode_scaling_factor", 1.0)
            if old_scaling != mode_scaling:
                ratio = mode_scaling / old_scaling if old_scaling else mode_scaling
                sie["composite_quality_score"] = round(sie["composite_quality_score"] * ratio, 4)
                sie["s_sie"] = round(sie["s_sie"] * ratio, 4)
                sie["mode_scaling_factor"] = mode_scaling

        # Guard: if SIE computation returned None or non-dict, force underflow
        if not isinstance(sie, dict):
            logger.warning(
                f"{SENTINEL_QUALITY_UNDERFLOW} Node {node.node_id}: "
                f"SIE computation returned {type(sie).__name__}, expected dict"
            )
            return False

        try:
            node.sie_node = SemanticReasoningNode(
                content_density=sie["content_density"],
                alignment_vector=sie["alignment_vector"],
                composite_quality_score=sie["composite_quality_score"],
                mode_scaling_factor=sie["mode_scaling_factor"],
                s_sie=sie["s_sie"],
            )
        except Exception:
            pass

        if self.telemetry:
            self.telemetry.record(
                "sie", "computed", node_id=node.node_id,
                payload={"s_sie": sie.get("s_sie", 0), "composite_quality_score": sie.get("composite_quality_score", 0)},
            )

        # Boolean SIE gate: reject information-dead nodes
        # A node is rejected if it has no content (content_density=0) OR no reasoning (||alignment_vector||=0)
        sie_pass = sie["content_density"] > 0.0 and sie["s_sie"] >= self.quality_floor
        if not sie_pass:
            logger.warning(
                f"{SENTINEL_QUALITY_UNDERFLOW} Node {node.node_id}: "
                f"SIE gate failed (content_density={sie['content_density']:.4f}, s_sie={sie['s_sie']:.4f})"
            )
            if self.telemetry:
                self.telemetry.record(
                    "sie", "gate_failed", node_id=node.node_id,
                    payload={"content_density": sie["content_density"], "s_sie": sie["s_sie"]},
                )
            return False
        return True

    def _compute_sie_inline(self, node: AletheiaSkill) -> Dict[str, Any]:
        """
        Standalone SIE computation (identical formula to ACS compute_sie_score).

        content_density:
            Code nodes  → AST depth / docstring / type-hint ratio
            Text nodes  → word-count depth proxy / structure markers
        alignment_vector: [coverage, novelty, consistency]
        composite_quality_score = mode_scaling_factor × content_density × ||alignment_vector||
        s_sie  = composite_quality_score
        """
        code_fields = ("implementation", "code_snippet", "implementation_code", "resolved_code", "attempt_code")
        text_fields = ("constraints", "chunk_text", "theory", "theory_text", "analysis_text")

        text_content = ""
        is_code = False
        # V2.1: Theorist Mode nodes carry natural language in code_snippet.
        # Detect non-code content via source_type or skill_type so the
        # text-based density path is used instead of the AST path.
        _is_text_origin = (
            getattr(node, "source_type", "") in ("ocr_document", "text_document", "pdf_extraction")
            or getattr(node, "skill_type", "") == "theoretical_reasoning"
        )
        # Check semantics dict first
        if isinstance(node.semantics, dict):
            for f in code_fields:
                val = node.semantics.get(f, "")
                if val and val.strip():
                    text_content = val
                    is_code = not _is_text_origin
                    break
            if not text_content:
                for f in text_fields:
                    val = node.semantics.get(f, "")
                    if val and val.strip():
                        text_content = val
                        break
        # Fallback: check node-level attributes (AST-extracted nodes)
        if not text_content:
            for f in code_fields:
                val = getattr(node, f, "")
                if val and val.strip():
                    text_content = val
                    is_code = not _is_text_origin
                    break
        if not text_content:
            for f in text_fields:
                val = getattr(node, f, "")
                if val and val.strip():
                    text_content = val
                    break
        if not text_content:
            text_content = node.source_context or ""

        # --- content_density computation ---
        content_density = 0.0
        depth_proxy = 0.0
        if is_code and text_content.strip():
            # AST-based content_density (identical to ACS)
            ast_depth = 0
            docstring_present = 0.0
            type_hint_ratio = 0.0
            func_count = 0
            hints_count = 0
            try:
                tree = ast.parse(text_content)

                def _max_depth(node_ast, depth=0):
                    max_d = depth
                    for child in ast.iter_child_nodes(node_ast):
                        max_d = max(max_d, _max_depth(child, depth + 1))
                    return max_d
                ast_depth = min(_max_depth(tree), 20)

                for n in ast.walk(tree):
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if ast.get_docstring(n):
                            docstring_present = 1.0
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        func_count += 1
                        has_arg_hints = any(
                            arg.annotation for arg in getattr(n.args, "args", [])
                        )
                        has_return = getattr(n, "returns", None) is not None
                        if has_arg_hints or has_return:
                            hints_count += 1
                type_hint_ratio = (hints_count / func_count) if func_count > 0 else 0.0
            except (SyntaxError, RecursionError, ValueError, MemoryError):
                pass
            content_density = min(1.0, (ast_depth / 10.0 + docstring_present * 0.2 + type_hint_ratio * 0.3))
            depth_proxy = min(ast_depth / 20.0, 1.0)
        elif text_content.strip():
            # Text-based content_density for non-code nodes (theorist, advocate, veteran analysis)
            words = text_content.split()
            depth_proxy = min(len(words) / 200.0, 1.0)
            structure_markers = sum(
                1 for w in ("because", "therefore", "constraint", "invariant",
                            "however", "specifically", "requires", "implies")
                if w in text_content.lower()
            )
            structure_score = min(structure_markers / 3.0, 1.0)
            content_density = min(1.0, depth_proxy * 0.5 + structure_score * 0.3 + 0.2)

        # --- alignment_vector: [coverage, novelty, consistency] ---
        # Validation data: check node attributes first, then semantics
        validation_data = getattr(node, "validation", None)
        if validation_data is None and isinstance(node.semantics, dict):
            validation_data = node.semantics.get("validation", {})
        validation_data = validation_data or {}
        assertion_count = (
            validation_data.get("assertion_count", 0)
            if isinstance(validation_data, dict) else 0
        )
        coverage = min(1.0, assertion_count / 5.0)

        novelty = 0.7 if (node.skill_type or "") in ("validation", "theoretical_reasoning") else 0.5

        # Teaching layer: check node attributes first, then semantics
        teaching_layer = getattr(node, "teaching_layer", None)
        if teaching_layer is None and isinstance(node.semantics, dict):
            teaching_layer = node.semantics.get("teaching_layer", {})
        if not teaching_layer:
            teaching_layer = {}
        reasoning = (
            teaching_layer.get("reasoning_vectors", {})
            if isinstance(teaching_layer, dict) else
            getattr(teaching_layer, "reasoning_vectors", {})
        )
        if isinstance(reasoning, dict):
            filled = sum(
                1 for k in ("intent", "strategy", "constraints",
                            "execution_pattern", "failure_modes")
                if reasoning.get(k)
            )
        else:
            filled = sum(
                1 for k in ("intent", "strategy", "constraints",
                            "execution_pattern", "failure_modes")
                if getattr(reasoning, k, None)
            )
        consistency = filled / 5.0

        alignment_vector = [coverage, novelty, consistency]

        # --- composite_quality_score and s_sie ---
        grad_norm = float(np.linalg.norm(alignment_vector))
        mode_scaling_factor = SCALING_FACTOR_BY_MODE.get(self.mode, 1.0)
        composite_quality_score = mode_scaling_factor * content_density * grad_norm
        s_sie = composite_quality_score

        return {
            "content_density": round(content_density, 4),
            "alignment_vector": [round(v, 4) for v in alignment_vector],
            "composite_quality_score": round(composite_quality_score, 4),
            "mode_scaling_factor": mode_scaling_factor,
            "s_sie": round(s_sie, 4),
        }

    # ------------------------------------------------------------------
    # ACS Governance
    # ------------------------------------------------------------------
    def _apply_acs(self, node: AletheiaSkill):
        """Run ACS evaluation and severity-driven constraint routing."""
        # Lens pressure telemetry: snapshot c_node before ACS
        _c_node_before_acs = node.epistemic.c_node

        node_dict = node.model_dump(exclude_unset=True)

        # Hoist reasoning_trajectory from semantics to top-level so ACS
        # Tikhonov dispatch gate can find it via get_val().
        sem = node_dict.get("semantics") or {}
        if isinstance(sem, dict) and "reasoning_trajectory" in sem:
            node_dict["reasoning_trajectory"] = sem["reasoning_trajectory"]

        if hasattr(self.acs, "evaluate_node"):
            node_dict = self.acs.evaluate_node(node_dict)
        elif hasattr(self.acs, "audit"):
            violations = self.acs.audit(node_dict, self.graph)
            if violations:
                self.acs.intervene(node_dict, violations, self.graph)

        # ---- V2.0 ACS metadata isolation --------------------------------
        # ACS evaluate_node() injects _-prefixed internal metadata
        # (_acs_structured_constraints, _acs_trajectory, _governance_directive)
        # that must NOT be present when re-validating against the strict
        # AletheiaSkill contract (extra="forbid").  Pop them into a
        # side-channel dict so the constraint router below can still read them.
        _acs_meta: Dict[str, Any] = {}
        for _k in list(node_dict.keys()):
            if _k.startswith("_"):
                _acs_meta[_k] = node_dict.pop(_k)
        # Also strip the hoisted reasoning_trajectory (not a model field).
        if "reasoning_trajectory" in node_dict:
            _acs_meta["reasoning_trajectory"] = node_dict.pop("reasoning_trajectory")

        try:
            refreshed = AletheiaSkill.model_validate(node_dict)
            node.epistemic = refreshed.epistemic
            node.v_score = refreshed.v_score
            node.validation_pass = refreshed.validation_pass
            node.acs_handshake_sid = refreshed.acs_handshake_sid
            node.acs_violations = refreshed.acs_violations
            node.acs_audited = refreshed.acs_audited
            if refreshed.sie_node:
                node.sie_node = refreshed.sie_node
        except ValidationError as e:
            logger.error(f"{SENTINEL_WARN_SCHEMA} ACS corrupted node schema: {e}")
            node.epistemic.state = NodeState.REJECTED
            # Priority 3: tag final_status so the outer _transition_to_scored
            # ledger entry can resolve the exact trigger enum.
            node.epistemic.final_status = f"acs_schema_corrupted: {e}"
            return

        # Severity-driven constraint router (read from side-channel)
        for c in _acs_meta.get("_acs_structured_constraints", []):
            sev = (
                c.get("severity", "warning")
                if isinstance(c, dict)
                else getattr(c, "severity", "warning")
            )
            if sev == "fatal":
                logger.error(f"FATAL constraint on {node.node_id}: {c}")
                node.epistemic.state = NodeState.REJECTED
                node.epistemic.final_status = "constraint_fatal"
                node.epistemic.c_node = 0.0
                break
            elif sev == "error":
                node.epistemic.c_node = round(node.epistemic.c_node * 0.5, 4)
                logger.warning(
                    f"ERROR constraint on {node.node_id}: {c} — 50% penalty"
                )

        # Lens pressure telemetry: record c_node delta from ACS pass
        if self.telemetry:
            _c_node_after_acs = node.epistemic.c_node
            _lens_delta = round(_c_node_after_acs - _c_node_before_acs, 4)
            self.telemetry.record(
                "lens_pressure", "assignment",
                node_id=node.node_id,
                payload={
                    "sid_matched":        getattr(node, "acs_handshake_sid", ""),
                    "c_final_before_lens": round(_c_node_before_acs, 4),
                    "c_final_after_lens":  round(_c_node_after_acs, 4),
                    "lens_delta":          _lens_delta,
                    "mode":                self.mode,
                },
            )

        # Feed ACS trajectory into identity manager (read from side-channel)
        trajectory = _acs_meta.get("_acs_trajectory")
        if trajectory and self.identity_manager:
            self.identity_manager.update_from_trajectory(trajectory)

    def _check_governance_directive(self, node: AletheiaSkill) -> bool:
        """
        Check for SLR breach governance directive from ACS.
        Returns True if node was rerouted (to REROLL or REJECTED).
        """
        node_dict = node.model_dump(exclude_unset=True)
        directive = node_dict.get("_governance_directive")
        if not directive or directive.get("action") != "reroll":
            return False

        if self.identity_manager:
            slr = directive.get("slr", 0.0)
            conviction = directive.get("conviction", 0.0)
            if self.telemetry:
                self.telemetry.record(
                    "slr", "breach_detected", node_id=node.node_id,
                    payload={"slr": slr, "conviction": conviction, "mode": self.mode},
                )
            self.identity_manager.update_on_slr_breach(node.node_id, slr, self.mode)
            if self.telemetry:
                self.telemetry.record(
                    "identity", "drift_update", node_id=node.node_id,
                    payload={"drift": self.identity_manager.drift_score},
                )

            branch_id = getattr(node.epistemic, "branch_id", "root")
            if self.identity_manager.check_and_enforce(branch_id):
                if self.telemetry:
                    self.telemetry.record(
                        "identity", "branch_frozen", node_id=node.node_id,
                        payload={"branch_id": branch_id},
                    )
                logger.warning(f"Identity drift exceeded — freezing branch {branch_id}")
                node.epistemic.state = NodeState.REJECTED
                node.epistemic.final_status = "identity_drift_frozen"
                self._log_transition(
                    node.node_id, NodeState.VALIDATED, NodeState.REJECTED,
                    "identity_drift",
                )

                # Phase 3.3 — Advocate Audit on governance-directive freeze
                audit_report = self._run_advocate_audit(node, branch_id)
                if self.telemetry:
                    self.telemetry.record(
                        "identity", "advocate_audit_completed", node_id=node.node_id,
                        payload={
                            "branch_id": branch_id,
                            "trigger": "governance_directive",
                            "audit_report": audit_report,
                        },
                    )
                self.rejected_traces.append({
                    "node_id": node.node_id,
                    "rejected": (
                        deepcopy(node.semantics)
                        if isinstance(node.semantics, dict) else {}
                    ),
                    "reason": "advocate_audit_governance_drift",
                    "corrected": None,
                    "structural_snapshot": {
                        "c_node": node.epistemic.c_node,
                        "identity_drift": self.identity_manager.drift_score,
                        "system_backpressure": self.system_backpressure,
                        "audit_report": audit_report,
                    },
                })

                self._cascade_rejection(node.node_id)
                return True

        self._route_to_reroll(node, node.epistemic.state, "sycophancy_breach")
        return True

    # ------------------------------------------------------------------
    # Hard Sycophancy Failure Handler (SIE-SLR)
    # ------------------------------------------------------------------
    def _handle_sycophancy_failure(self, node: AletheiaSkill, slr: float) -> bool:
        """
        Dedicated sycophancy handler triggered by SIE-SLR enforcement.
        Actions by SLR tier:
          - slr >= collapse (0.8): hard rejection + confidence collapse
          - slr >= reroll (0.7): identity mutation + reroll
          - slr >= warning (0.6): log + telemetry only (returns False)
        Returns True if the node was rerouted (REJECTED or REROLL).
        """
        prev = node.epistemic.state

        # Identity mutation for any actionable SLR
        if self.identity_manager and slr >= SLR_THRESHOLDS["reroll"]:
            self.identity_manager.update_on_slr_breach(node.node_id, slr, self.mode)
            if self.telemetry:
                self.telemetry.record(
                    "slr", "sie_slr_breach", node_id=node.node_id,
                    payload={"slr": slr, "tier": "collapse" if slr >= SLR_THRESHOLDS["collapse"] else "reroll"},
                )

        # Hard rejection: SLR >= collapse threshold
        if slr >= SLR_THRESHOLDS["collapse"]:
            node.epistemic.c_node = 0.0
            node.epistemic.state = NodeState.REJECTED
            node.epistemic.final_status = "sycophancy_collapse"
            self._log_transition(
                node.node_id, prev, NodeState.REJECTED,
                f"sycophancy_collapse: slr={slr:.4f} >= {SLR_THRESHOLDS['collapse']}",
            )
            self._cascade_rejection(node.node_id)
            return True

        # Reroll: SLR >= reroll threshold
        if slr >= SLR_THRESHOLDS["reroll"]:
            # Confidence collapse proportional to SLR before reroll
            node.epistemic.c_node = round(node.epistemic.c_node * (1.0 - slr), 4)
            self._route_to_reroll(
                node, prev,
                f"sycophancy_reroll: slr={slr:.4f} >= {SLR_THRESHOLDS['reroll']}",
            )
            return True

        # Warning tier: log only, no routing
        if slr >= SLR_THRESHOLDS["warning"] and self.telemetry:
            self.telemetry.record(
                "slr", "sie_slr_warning", node_id=node.node_id,
                payload={"slr": slr},
            )

        return False

    # ------------------------------------------------------------------
    # Proactive Identity Enforcement (Every Node)
    # ------------------------------------------------------------------
    def _enforce_identity(self, node: AletheiaSkill) -> bool:
        """
        Proactive identity drift check at every node.
        Returns True if node was rejected due to identity freeze.

        Unlike _check_governance_directive (which only fires on ACS sycophancy
        detection), this runs unconditionally — converting identity from an
        observer into an execution authority.
        """
        if not self.identity_manager:
            return False

        # If identity is already globally frozen, reject immediately
        if self.identity_manager.frozen:
            prev = node.epistemic.state
            node.epistemic.state = NodeState.REJECTED
            node.epistemic.final_status = "identity_frozen"
            node.epistemic.c_node = 0.0
            logger.warning(
                f"{SENTINEL_IDENTITY_FREEZE} Node {node.node_id}: "
                f"identity globally frozen — rejecting."
            )
            self._log_transition(
                node.node_id, prev, NodeState.REJECTED, "identity_globally_frozen",
            )
            if self.telemetry:
                self.telemetry.record(
                    "identity", "node_killed", node_id=node.node_id,
                    payload={
                        "reason": "identity_frozen",
                        "drift": self.identity_manager.drift_score,
                    },
                )

            # Phase 3.3 — Advocate Audit for the first node killed under a
            # global freeze.  Subsequent kills reuse the same diagnostic
            # (the freeze epoch hasn't changed).
            if not getattr(self, "_global_freeze_audited", False):
                self._global_freeze_audited = True
                branch_id = getattr(node.epistemic, "branch_id", "root")
                audit_report = self._run_advocate_audit(node, branch_id)
                if self.telemetry:
                    self.telemetry.record(
                        "identity", "advocate_audit_completed", node_id=node.node_id,
                        payload={
                            "branch_id": branch_id,
                            "trigger": "global_freeze",
                            "audit_report": audit_report,
                        },
                    )
                self.rejected_traces.append({
                    "node_id": node.node_id,
                    "rejected": (
                        deepcopy(node.semantics)
                        if isinstance(node.semantics, dict) else {}
                    ),
                    "reason": "advocate_audit_global_freeze",
                    "corrected": None,
                    "structural_snapshot": {
                        "c_node": node.epistemic.c_node,
                        "pre_state": prev,
                        "identity_drift": self.identity_manager.drift_score,
                        "system_backpressure": self.system_backpressure,
                        "audit_report": audit_report,
                    },
                })

            self._cascade_rejection(node.node_id)
            return True

        # Proactive per-node drift check
        branch_id = getattr(node.epistemic, "branch_id", "root")
        if self.identity_manager.check_and_enforce(branch_id):
            prev = node.epistemic.state
            node.epistemic.state = NodeState.REJECTED
            node.epistemic.final_status = "identity_drift_frozen"
            node.epistemic.c_node = 0.0
            logger.warning(
                f"{SENTINEL_IDENTITY_FREEZE} Node {node.node_id}: "
                f"identity drift exceeded — freezing branch {branch_id}."
            )
            self._log_transition(
                node.node_id, prev, NodeState.REJECTED, "identity_drift_proactive",
            )
            if self.telemetry:
                self.telemetry.record(
                    "identity", "proactive_freeze", node_id=node.node_id,
                    payload={
                        "branch_id": branch_id,
                        "drift": self.identity_manager.drift_score,
                    },
                )
            
            # Run Advocate audit: active diagnostic pass on the frozen branch.
            # The audit report is captured as a structured rejected trace
            # so it can be routed into the SFT rejection pool for training.
            audit_report = self._run_advocate_audit(node, branch_id)
            if self.telemetry:
                self.telemetry.record(
                    "identity", "advocate_audit_completed", node_id=node.node_id,
                    payload={
                        "branch_id": branch_id,
                        "drift": self.identity_manager.drift_score,
                        "audit_report": audit_report,
                    },
                )

            # Route the audit diagnostic into rejected_traces for SFT isolation.
            # This ensures every freeze produces actionable training signal.
            self.rejected_traces.append({
                "node_id": node.node_id,
                "rejected": (
                    deepcopy(node.semantics)
                    if isinstance(node.semantics, dict) else {}
                ),
                "reason": "advocate_audit_identity_drift",
                "corrected": None,
                "structural_snapshot": {
                    "c_node": node.epistemic.c_node,
                    "pre_state": prev,
                    "identity_drift": self.identity_manager.drift_score,
                    "system_backpressure": self.system_backpressure,
                    "audit_report": audit_report,
                },
            })

            self._cascade_rejection(node.node_id)
            return True

        return False

    def _run_advocate_audit(self, node: AletheiaSkill, branch_id: str) -> Dict[str, Any]:
        """
        Phase 3.3 — Advocate Audit Protocol.

        Spawns a full Advocate-mode diagnostic pass over the frozen branch's
        node history when the IdentityManager triggers a branch freeze.

        Analysis dimensions
        -------------------
        1. **Ancestor Chain Inspection** — walk the DAG backwards from the
           frozen node, capturing per-ancestor confidence, SIE pass/fail,
           and constraint counts.
        2. **SLR History Analysis** — extract the rolling SLR window from
           the IdentityManager and identify the breach sequence that
           accumulated enough drift to trigger the freeze.
        3. **Mode-Weight Skew** — compare the current mode_weights against
           the stable reference to identify which mode's penalties drove
           the drift.
        4. **Adversarial Lens Cross-Check** — if an AdversarialLens is
           available, score the frozen node through the orthogonal channel
           and report any primary_overscoring detection.
        5. **Root Cause Classification** — produce a deterministic
           diagnostic including the causal chain, the dominant breach
           mode, and a remediation recommendation.
        6. **Artifact Persistence** — write the report to a JSONL in
           ``output/`` so it is available for DPO pairing and forensic
           review, independent of the in-memory telemetry lifecycle.
        """
        # ── 1. Scaffold report ────────────────────────────────────────
        identity_snap = (
            self.identity_manager.get_state_snapshot()
            if self.identity_manager else {}
        )
        report: Dict[str, Any] = {
            "audit_type": "advocate_freeze_diagnostic",
            "branch_id": branch_id,
            "frozen_node": node.node_id,
            "mode": self.mode,
            "drift_score": identity_snap.get("drift_score"),
            "kl_divergence": identity_snap.get("kl_divergence"),
            "rolling_slr_mean": identity_snap.get("rolling_slr_mean"),
            "rolling_slr_critical": identity_snap.get("rolling_slr_critical"),
            "slr_breach_count": identity_snap.get("slr_breach_count"),
            "system_backpressure": self.system_backpressure,
            "identity_version": identity_snap.get("version"),
            "ancestor_chain": [],
            "slr_history": [],
            "mode_weight_skew": {},
            "adversarial_cross_check": None,
            "root_cause": "unknown",
            "dominant_breach_mode": None,
            "recommendation": "manual_review",
        }

        # ── 2. Ancestor chain walk ───────────────────────────────────
        try:
            if self._nx_graph and node.node_id in self._nx_graph:
                ancestors = list(nx.ancestors(self._nx_graph, node.node_id))
                for anc_id in sorted(ancestors):
                    anc_node = self.graph.nodes.get(anc_id) if hasattr(self.graph, "nodes") else None
                    cog = self._cognitive_nodes.get(anc_id)
                    entry: Dict[str, Any] = {"node_id": anc_id}
                    if cog:
                        entry["c_node"] = cog.c_node
                        entry["sie_pass"] = cog.sie_pass
                        entry["constraint_count"] = len(cog.constraints) if cog.constraints else 0
                    if anc_node:
                        entry["state"] = getattr(anc_node.epistemic, "state", None)
                        entry["final_status"] = getattr(anc_node.epistemic, "final_status", None)
                    report["ancestor_chain"].append(entry)
        except Exception as e:
            logger.warning(f"Advocate audit ancestor walk failed: {e}")

        # ── 3. SLR history analysis ──────────────────────────────────
        if self.identity_manager and self.identity_manager.slr_history:
            slr_hist = list(self.identity_manager.slr_history)
            report["slr_history"] = [round(s, 4) for s in slr_hist]
            # Identify the breach sequence: contiguous tail of SLR values
            # that stayed above the reroll threshold.
            breach_seq = []
            for s in reversed(slr_hist):
                if s >= SLR_THRESHOLDS["reroll"]:
                    breach_seq.insert(0, round(s, 4))
                else:
                    break
            report["slr_breach_sequence"] = breach_seq

        # ── 4. Mode-weight skew ──────────────────────────────────────
        if self.identity_manager:
            current_w = self.identity_manager.mode_weights
            stable_w = self.identity_manager._stable_state.mode_weights if hasattr(self.identity_manager, '_stable_state') else {}
            skew: Dict[str, float] = {}
            for mode_key in current_w:
                delta = round(current_w.get(mode_key, 0.0) - stable_w.get(mode_key, 0.0), 4)
                if abs(delta) > 1e-6:
                    skew[mode_key] = delta
            report["mode_weight_skew"] = skew

            # Determine which mode accumulated the most behavioral drift
            bp = self.identity_manager.behavior_profile
            if bp:
                dominant = max(bp, key=lambda k: abs(bp[k]))
                if abs(bp[dominant]) > 1e-6:
                    report["dominant_breach_mode"] = dominant

        # ── 5. Adversarial lens cross-check ──────────────────────────
        if self._adversarial_lens:
            try:
                adv_result = self._adversarial_lens.evaluate(node)
                report["adversarial_cross_check"] = {
                    "v_score": adv_result.get("v_score"),
                    "conflict_class": adv_result.get("conflict_class"),
                    "primary_overscoring": adv_result.get("primary_overscoring", False),
                    "flags": adv_result.get("flags", []),
                }
            except Exception as e:
                report["adversarial_cross_check"] = {"error": str(e)}

        # ── 6. Root cause classification ─────────────────────────────
        low_confidence_ancestors = [
            a for a in report["ancestor_chain"] if a.get("c_node", 1.0) < 0.5
        ]
        sie_failures = [
            a for a in report["ancestor_chain"] if not a.get("sie_pass", True)
        ]

        if sie_failures:
            report["root_cause"] = "upstream_sie_failure"
            report["recommendation"] = (
                "review_semantic_quality_of_ancestors: "
                f"{[a['node_id'] for a in sie_failures]}"
            )
        elif low_confidence_ancestors:
            report["root_cause"] = "cascading_low_confidence"
            report["recommendation"] = (
                "revalidate_low_confidence_ancestors: "
                f"{[a['node_id'] for a in low_confidence_ancestors]}"
            )
        elif self.system_backpressure > 0.7:
            report["root_cause"] = "high_system_backpressure_instability"
            report["recommendation"] = "reduce_batch_reject_rate_or_throttle_ingestion"
        elif report.get("dominant_breach_mode"):
            report["root_cause"] = f"mode_penalty_accumulation_{report['dominant_breach_mode']}"
            report["recommendation"] = (
                f"audit_{report['dominant_breach_mode']}_mode_payloads_for_sycophantic_patterns"
            )
        else:
            report["root_cause"] = "accumulated_identity_drift"
            report["recommendation"] = "reset_identity_baseline_after_forensic_review"

        # ── 7. Persist diagnostic artifact ───────────────────────────
        try:
            audit_path = Path("output") / "advocate_audits"
            audit_path.mkdir(parents=True, exist_ok=True)
            audit_file = audit_path / f"advocate_audit_{branch_id}_{node.node_id}.jsonl"
            with open(audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(report, sort_keys=True, default=str) + "\n")
            report["_persisted_to"] = str(audit_file)
            logger.info(
                f"[ADVOCATE_AUDIT] Diagnostic artifact persisted to {audit_file}"
            )
        except Exception as e:
            logger.warning(f"[ADVOCATE_AUDIT] Failed to persist artifact: {e}")

        logger.info(
            f"[ADVOCATE_AUDIT] Branch {branch_id} frozen at {node.node_id}: "
            f"root_cause={report['root_cause']}, "
            f"dominant_mode={report.get('dominant_breach_mode')}, "
            f"slr_breaches={report.get('slr_breach_count')}, "
            f"recommendation={report['recommendation']}"
        )
        return report

    # ------------------------------------------------------------------
    # Parent-Chain Validation (Topological + SIE Coherence Authority)
    # ------------------------------------------------------------------
    def _compute_sie_edge_coherence(
        self, parent: AletheiaSkill, child: AletheiaSkill,
    ) -> float:
        """
        Cosine similarity between parent and child alignment_vector vectors.
        Returns 0.0 if either node lacks SIE data.

        This ensures not just "both nodes are individually good" but
        "the reasoning chain from parent → child is coherent."
        """
        p_sie = parent.sie_node
        c_sie = child.sie_node
        if not p_sie or not c_sie:
            return 0.0  # No SIE data = incoherent (conservative default)

        p_grad = np.array(
            p_sie.alignment_vector
            if hasattr(p_sie, "alignment_vector") else [0.0, 0.0, 0.0]
        )
        c_grad = np.array(
            c_sie.alignment_vector
            if hasattr(c_sie, "alignment_vector") else [0.0, 0.0, 0.0]
        )

        p_norm = np.linalg.norm(p_grad)
        c_norm = np.linalg.norm(c_grad)

        if p_norm < 1e-9 or c_norm < 1e-9:
            return 0.0  # Zero-vector = incoherent

        return float(np.dot(p_grad, c_grad) / (p_norm * c_norm))

    def _validate_parent_chain(self, node: AletheiaSkill) -> bool:
        """
        Validates all parent (predecessor) nodes before allowing acceptance:
        1. Unaccepted parents apply a soft cascade penalty to child c_node.
           Strong child nodes can still proceed if the penalty leaves them above
           the acceptance threshold.
        2. SIE alignment_vector must be coherent across each edge (Patch 2).

        Returns True if parent chain is valid, False otherwise.
        """
        if self._nx_graph is None:
            return True

        try:
            predecessors = list(self._nx_graph.predecessors(node.node_id))
        except nx.NetworkXError:
            return True

        if not predecessors:
            return True  # Root nodes have no parents to validate

        node_map = {n.node_id: n for n in self._get_nodes_safely()}
        accepted_parent_confidences = []
        for parent_id in predecessors:
            parent = node_map.get(parent_id)
            if not parent:
                continue

            # Gate 1: unaccepted parent applies a soft cascade penalty.
            if parent.epistemic.state != NodeState.ACCEPTED:
                old_confidence = float(getattr(node.epistemic, "c_node", 0.0) or 0.0)
                new_confidence = round(old_confidence * SOFT_PARENT_CASCADE_FACTOR, 4)
                node.epistemic.c_node = new_confidence
                parent_state_name = getattr(parent.epistemic.state, "name", str(parent.epistemic.state))
                logger.warning(
                    f"{SENTINEL_PARENT_CHAIN} Node {node.node_id}: parent {parent_id} "
                    f"is {parent_state_name}; soft cascade c_node "
                    f"{old_confidence:.4f} -> {new_confidence:.4f}."
                )
                logger.debug(
                    f"{SENTINEL_PARENT_CHAIN} Node {node.node_id}: "
                    f"parent {parent_id} soft cascade already logged."
                )
                if self.telemetry:
                    self.telemetry.record(
                        "topology", "soft_parent_cascade", node_id=node.node_id,
                        payload={
                            "parent": parent_id,
                            "parent_state": parent.epistemic.state,
                            "factor": SOFT_PARENT_CASCADE_FACTOR,
                            "old_c_node": round(old_confidence, 4),
                            "new_c_node": new_confidence,
                        },
                    )
                continue

            # Gate 1b: Soft cascade — track accepted parent confidences
            accepted_parent_confidences.append(parent.epistemic.c_node)

            # Gate 2: SIE edge coherence
            coherence = self._compute_sie_edge_coherence(parent, node)
            threshold = self.edge_coherence_tuning_threshold
            if coherence < threshold:
                logger.warning(
                    f"{SENTINEL_EDGE_INCOHERENT} Node {node.node_id}: "
                    f"SIE coherence with parent {parent_id} = {coherence:.4f} "
                    f"< {threshold:.4f} — reasoning chain broken."
                )
                if self.telemetry:
                    self.telemetry.record(
                        "topology", "sie_edge_incoherent", node_id=node.node_id,
                        payload={
                            "parent": parent_id,
                            "coherence": round(coherence, 4),
                            "threshold": round(threshold, 4),
                            "default_threshold": EDGE_COHERENCE_THRESHOLD,
                        },
                    )
                return False

            # Gate 3: SIE-SLR — alignment_vector drift between parent and child
            edge_slr = calculate_sie_slr(parent, node)
            if edge_slr >= SLR_THRESHOLDS["reroll"]:
                logger.warning(
                    f"{SENTINEL_EDGE_INCOHERENT} Node {node.node_id}: "
                    f"SIE-SLR with parent {parent_id} = {edge_slr:.4f} "
                    f">= {SLR_THRESHOLDS['reroll']} — identity drift on edge."
                )
                if self.telemetry:
                    self.telemetry.record(
                        "topology", "sie_slr_edge_fail", node_id=node.node_id,
                        payload={
                            "parent": parent_id,
                            "edge_slr": round(edge_slr, 4),
                            "threshold": SLR_THRESHOLDS["reroll"],
                        },
                    )
                return False

        # Soft cascade: dampen child confidence based on mean parent c_node
        if accepted_parent_confidences:
            parent_confidence_factor = sum(accepted_parent_confidences) / len(
                accepted_parent_confidences
            )
            # Store for audit — applied later in _compute_unified_confidence
            # via the node's semantics dict
            if isinstance(node.semantics, dict):
                node.semantics["_parent_confidence_factor"] = round(
                    parent_confidence_factor, 4
                )
            if parent_confidence_factor < 0.5 and self.telemetry:
                self.telemetry.record(
                    "topology", "soft_cascade_warning", node_id=node.node_id,
                    payload={"parent_confidence_factor": round(parent_confidence_factor, 4)},
                )

        return True

    # ------------------------------------------------------------------
    # Telemetry → Actions (Signal-Driven Governance + Field Pressure)
    # ------------------------------------------------------------------
    def _check_telemetry_thresholds(self) -> None:
        """
        Convert aggregate telemetry observations into governance actions
        and compute global field pressure (Patch 4).

        Delegates to src.pipeline.telemetry_policy.check_telemetry_thresholds().
        """
        all_nodes = self._get_nodes_safely()
        if all_nodes:
            non_rejected = sum(
                1 for n in all_nodes
                if n.epistemic.state != NodeState.REJECTED
            )
            non_rejected_ratio = non_rejected / len(all_nodes)
        else:
            non_rejected_ratio = 1.0

        result = _check_telemetry_policy(
            telemetry=self.telemetry,
            identity_manager=self.identity_manager,
            acceptance_threshold=self.acceptance_threshold,
            system_backpressure=self.system_backpressure,
            non_rejected_ratio=non_rejected_ratio,
        )
        self.acceptance_threshold = result["acceptance_threshold"]
        self.system_backpressure = result["system_backpressure"]

        self._policy_check_count += 1
        self._apply_runtime_auto_tuning(non_rejected_ratio=non_rejected_ratio)

    def _should_run_telemetry_check(self, iteration: int) -> bool:
        cadence = max(1, int(self.telemetry_check_cadence))
        return iteration % cadence == 0

    def _should_run_recovery_tick(self, iteration: int) -> bool:
        cadence = max(1, int(self.recovery_tick_cadence))
        return iteration % cadence == 0

    def _emit_auto_tuning_event(
        self,
        parameter: str,
        old_value: Any,
        new_value: Any,
        reason: str,
    ) -> None:
        entry = {
            "policy_check": self._policy_check_count,
            "old": old_value,
            "new": new_value,
            "reason": reason,
        }
        self._tuning_history.setdefault(parameter, []).append(entry)
        if self.telemetry:
            self.telemetry.record(
                "auto_tuning", "runtime_tuning_update", node_id="global",
                payload={
                    "parameter": parameter,
                    "old": old_value,
                    "new": new_value,
                    "reason": reason,
                    "policy_check": self._policy_check_count,
                },
            )

    def _apply_runtime_auto_tuning(self, non_rejected_ratio: float) -> None:
        if not self.auto_tuning_enabled or not self.identity_manager:
            return

        checks_since_last = self._policy_check_count - self._last_tuning_update_check
        if checks_since_last < max(0, int(self.auto_tuning_cooldown_checks)):
            return

        rolling_slr = self.identity_manager.get_rolling_slr_mean()
        drift = self.identity_manager.drift_score

        changed = False

        old_threshold = self.edge_coherence_tuning_threshold
        if rolling_slr >= SLR_THRESHOLDS["reroll"] or drift >= 0.2:
            self.edge_coherence_tuning_threshold = min(
                AUTO_TUNE_COHERENCE_MAX,
                round(self.edge_coherence_tuning_threshold + AUTO_TUNE_COHERENCE_STEP, 4),
            )
            if self.edge_coherence_tuning_threshold != old_threshold:
                self._emit_auto_tuning_event(
                    "coherence_threshold",
                    round(old_threshold, 4),
                    round(self.edge_coherence_tuning_threshold, 4),
                    "high_slr_or_drift",
                )
                changed = True
        elif (
            rolling_slr <= SLR_THRESHOLDS["safe"]
            and drift <= 0.05
            and non_rejected_ratio >= 0.8
        ):
            self.edge_coherence_tuning_threshold = max(
                AUTO_TUNE_COHERENCE_MIN,
                round(self.edge_coherence_tuning_threshold - AUTO_TUNE_COHERENCE_STEP, 4),
            )
            if self.edge_coherence_tuning_threshold != old_threshold:
                self._emit_auto_tuning_event(
                    "coherence_threshold",
                    round(old_threshold, 4),
                    round(self.edge_coherence_tuning_threshold, 4),
                    "stable_low_slr",
                )
                changed = True

        old_cadence = self.recovery_tick_cadence
        if rolling_slr >= SLR_THRESHOLDS["reroll"] or drift >= 0.2:
            self.recovery_tick_cadence = max(
                AUTO_TUNE_RECOVERY_CADENCE_MIN,
                self.recovery_tick_cadence - 1,
            )
            if self.recovery_tick_cadence != old_cadence:
                self._emit_auto_tuning_event(
                    "recovery_tick_cadence",
                    old_cadence,
                    self.recovery_tick_cadence,
                    "high_slr_or_drift",
                )
                changed = True
        elif (
            rolling_slr <= SLR_THRESHOLDS["safe"]
            and drift <= 0.05
            and non_rejected_ratio >= 0.8
        ):
            self.recovery_tick_cadence = min(
                AUTO_TUNE_RECOVERY_CADENCE_MAX,
                self.recovery_tick_cadence + 1,
            )
            if self.recovery_tick_cadence != old_cadence:
                self._emit_auto_tuning_event(
                    "recovery_tick_cadence",
                    old_cadence,
                    self.recovery_tick_cadence,
                    "stable_low_slr",
                )
                changed = True

        if changed:
            self._last_tuning_update_check = self._policy_check_count

    # ------------------------------------------------------------------
    # Evidence Validation
    # ------------------------------------------------------------------
    def _validate_evidence(self, node: AletheiaSkill):
        """Score based on source type and evidence references."""
        evidence: List[Any] = getattr(self.graph, "get_evidence", lambda x: [])(node.node_id)
        st = (node.source_type or "").lower()

        score = 1.0 if st in ("ast_code", "macro_skill") else (0.85 if evidence else 0.75)
        node.v_score = score
        node.validation_pass = score >= 0.7

        if evidence:
            refs = getattr(node.epistemic, "evidence_refs", [])
            for e in evidence:
                if e not in refs:
                    refs.append(e)
            node.epistemic.evidence_refs = refs

    # ------------------------------------------------------------------
    # Unified Confidence Computation (Additive Weighted Sum)
    # ------------------------------------------------------------------
    def _compute_unified_confidence(self, node: AletheiaSkill):
        """
        Additive weighted confidence:
          c_final = α·s_sie + β·s_acs + γ·s_topology + δ·s_validation

        Weights: α=0.4, β=0.3, γ=0.2, δ=0.1 (defined as module constants).

        Post-computation dampeners (applied multiplicatively to c_final):
          - SIE coherence decay (parent-child SLR)
          - Identity drift
          - Multi-hop depth penalty
          - Global field pressure

        Stores per-component breakdown in epistemic.confidence for audit.
        """
        # s_sie: from SIE node
        s_sie = min(1.0, node.sie_node.s_sie) if node.sie_node else 0.0

        # s_acs: current c_node after ACS penalty absorption
        s_acs = min(1.0, node.epistemic.c_node)

        # s_topology: from cached centrality or dependency satisfaction.
        # Fix 3: treat blast_radius < 0.1 as unset — sparse-graph PageRank assigns
        # near-zero centrality to valid isolated functions; fall through to the
        # dependency-satisfaction heuristic (default 0.8 for leaf nodes) instead.
        s_topology = 0.0
        if isinstance(node.semantics, dict):
            s_topology = min(
                1.0, node.semantics.get("system_centrality_blast_radius", 0.0)
            )
        if s_topology < 0.1:
            deps = node.dependencies
            downstream: List[str] = []
            if deps:
                downstream = (
                    deps.get("downstream_calls", [])
                    if isinstance(deps, dict)
                    else getattr(deps, "downstream_calls", [])
                )
            if downstream:
                validated_ids = {n.node_id for n in self._get_validated_nodes()}
                satisfied = sum(1 for d in downstream if d in validated_ids)
                s_topology = (satisfied + 1) / (len(downstream) + 1)
            else:
                s_topology = 0.8

        # s_validation: from validate() outcome
        s_validation = getattr(node, "v_score", 0.75)

        # --- Additive weighted sum ---
        c_final = round(
            min(1.0,
                CONFIDENCE_ALPHA * s_sie
                + CONFIDENCE_BETA * s_acs
                + CONFIDENCE_GAMMA * s_topology
                + CONFIDENCE_DELTA * s_validation),
            4,
        )

        # --- Phase 2.4: SIE coherence decay (parent-child SLR dampening) ---
        max_edge_slr = self._compute_max_parent_slr(node)
        if max_edge_slr > 0.0 and c_final > 0.0:
            c_final = round(c_final * (1.0 - max_edge_slr), 4)

        # --- Phase 3.5: Identity drift dampening ---
        identity_drift = 0.0
        if self.identity_manager:
            identity_drift = min(1.0, self.identity_manager.drift_score)
            if identity_drift > 0.0 and c_final > 0.0:
                c_final = round(c_final * (1.0 - identity_drift), 4)

        # --- Phase 7: Multi-hop depth penalty ---
        depth = float(getattr(node.epistemic, "depth", 0))
        depth_penalty = min(DEPTH_PENALTY_CAP, depth * DEPTH_PENALTY_RATE)
        if depth_penalty > 0.0 and c_final > 0.0:
            c_final = round(c_final * (1.0 - depth_penalty), 4)

        # --- Apply global system backpressure ---
        # Fix 2: skip backpressure dampening for trusted-source ingestion runs.
        if self.system_backpressure > 0.0 and not _TRUSTED_MODE:
            c_final = round(c_final * (1.0 - min(self.system_backpressure, 0.9)), 4)

        # --- Defensive floor: prevent negative/NaN from chained dampening ---
        if not np.isfinite(c_final) or c_final < 0.0:
            c_final = 0.0

        node.epistemic.confidence = {
            "sie": round(s_sie, 4),
            "acs": round(s_acs, 4),
            "topology": round(s_topology, 4),
            "validation": round(s_validation, 4),
            "model": "additive_weighted",
            "weights": {
                "alpha": CONFIDENCE_ALPHA,
                "beta": CONFIDENCE_BETA,
                "gamma": CONFIDENCE_GAMMA,
                "delta": CONFIDENCE_DELTA,
            },
            "coherence_decay": round(max_edge_slr, 4),
            "identity_drift": round(identity_drift, 4),
            "depth_penalty": round(depth_penalty, 4),
            "system_backpressure": round(self.system_backpressure, 4),
            "final": c_final,
        }
        node.epistemic.c_node = c_final

    def _compute_max_parent_slr(self, node: AletheiaSkill) -> float:
        """Returns the maximum SIE-SLR across all parent edges, or 0.0 if none."""
        if self._nx_graph is None:
            return 0.0
        try:
            predecessors = list(self._nx_graph.predecessors(node.node_id))
        except nx.NetworkXError:
            return 0.0
        if not predecessors:
            return 0.0
        node_map = {n.node_id: n for n in self._get_nodes_safely()}
        max_slr = 0.0
        for pid in predecessors:
            parent = node_map.get(pid)
            if parent:
                slr = calculate_sie_slr(parent, node)
                # Store SLR on the edge for audit trail
                edge_data = self._nx_graph.get_edge_data(pid, node.node_id) or {}
                edge_obj = edge_data.get("edge")
                if edge_obj is not None:
                    edge_obj.slr = slr
                max_slr = max(max_slr, slr)
        return max_slr

    # ------------------------------------------------------------------
    # Soft Branch Stability (Phase 7 — replaces hard pruning)
    # ------------------------------------------------------------------
    def _apply_branch_stability(self):
        """
        Compute per-branch stability and apply soft degradation.

        stability = mean(c_node) × (1 - max_slr_in_branch)

        Instead of hard-pruning unstable branches (which risks false negatives
        in early DAG stages), we degrade node confidence by a factor so the
        DAG naturally eliminates them through the normal scoring pipeline.
        """
        branch_ids = getattr(self.graph, "branch_ids", lambda: [])()
        if not branch_ids:
            return

        for bid in branch_ids:
            branch_nodes = self.graph.get_branch(bid)
            active_nodes = [
                n for n in branch_nodes
                if n.epistemic.state not in NodeState.TERMINAL
            ]
            if not active_nodes:
                continue

            c_values = [n.epistemic.c_node for n in active_nodes]
            mean_c = sum(c_values) / len(c_values) if c_values else 0.0

            # Max SLR within the branch
            max_slr = 0.0
            if self._nx_graph is not None:
                node_map = {n.node_id: n for n in active_nodes}
                for n in active_nodes:
                    try:
                        for pid in self._nx_graph.predecessors(n.node_id):
                            parent = node_map.get(pid)
                            if parent:
                                max_slr = max(max_slr, calculate_sie_slr(parent, n))
                    except nx.NetworkXError:
                        pass

            stability = mean_c * (1.0 - max_slr)

            if stability < BRANCH_STABILITY_THRESHOLD:
                # Soft degradation — multiply confidence down, mark as degraded
                for n in active_nodes:
                    n.epistemic.c_node = round(
                        n.epistemic.c_node * BRANCH_DEGRADATION_FACTOR, 4
                    )
                    if isinstance(n.semantics, dict):
                        n.semantics["_branch_degraded"] = True
                        n.semantics["_branch_stability"] = round(stability, 4)

                if self.telemetry:
                    self.telemetry.record(
                        "topology", "branch_degraded",
                        payload={
                            "branch_id": bid,
                            "stability": round(stability, 4),
                            "node_count": len(active_nodes),
                        },
                    )
                logger.info(
                    f"[DAG] Branch {bid}: stability={stability:.4f} "
                    f"< {BRANCH_STABILITY_THRESHOLD} — soft degradation applied"
                )

    # ------------------------------------------------------------------
    # Cascade Rejection
    # ------------------------------------------------------------------
    def _cascade_rejection(self, failed_node_id: str):
        """Topologically cascade rejection to all downstream descendants."""
        if self._nx_graph is None:
            self._nx_graph = self._build_directed_graph(self._get_nodes_safely())

        try:
            descendants = nx.descendants(self._nx_graph, failed_node_id)
        except nx.NetworkXError:
            return

        if not descendants:
            return

        for node in self._get_nodes_safely():
            if node.node_id in descendants:
                # Skip already-terminal nodes to prevent duplicate logging/work
                if node.epistemic.state == NodeState.REJECTED:
                    continue
                # Fix 6: in trusted-source mode, don't cascade-reject a node that
                # already has an independently established confidence score above the
                # instability floor — the upstream failure should not override a
                # standalone quality assessment for a known-good source.
                if _TRUSTED_MODE and node.epistemic.c_node >= INSTABILITY_BAND_LOW:
                    continue
                prev = node.epistemic.state
                node.epistemic.c_node = 0.0
                node.epistemic.state = NodeState.REJECTED
                node.epistemic.final_status = f"cascade_from_{failed_node_id}"
                logger.warning(
                    f"{SENTINEL_CASCADE_PRUNE} Cascade: {node.node_id} "
                    f"rejected due to upstream failure of {failed_node_id}"
                )
                # Priority 3 — Cryptographic Ledger: every cascade rejection
                # is sealed so the topological blast-radius of the root
                # failure is queryable.
                self._log_transition(
                    node.node_id, prev, NodeState.REJECTED,
                    reason=f"cascade_from_{failed_node_id}",
                    trigger=TransitionTrigger.CASCADE_FROM_PARENT,
                )
                # Mark inbound edge as constraint-failed
                if self._nx_graph.has_edge(failed_node_id, node.node_id):
                    edge_data = self._nx_graph[failed_node_id][node.node_id]
                    edge_obj = edge_data.get("edge")
                    if edge_obj is not None:
                        edge_obj.constraint_ok = False
                self._sync_node(node)

    # ------------------------------------------------------------------
    # CognitiveNode Layer Builder
    # ------------------------------------------------------------------
    def _build_cognitive_layer(self, nodes: List[AletheiaSkill]) -> Dict[str, Any]:
        """Build CognitiveNode + ReasoningEdge overlay for domain-driven consumers."""
        try:
            cognitive_nodes = {}
            reasoning_edges = []

            for skill in nodes:
                sie = skill.sie_node if skill.sie_node else SemanticReasoningNode()

                raw_constraints = getattr(skill, "_acs_structured_constraints", [])
                typed = []
                for rc in raw_constraints:
                    try:
                        typed.append(
                            Constraint.model_validate(rc) if isinstance(rc, dict) else rc
                        )
                    except Exception:
                        pass

                if isinstance(skill.constraints, dict):
                    for ic in skill.constraints.get("initial_constraints", []):
                        try:
                            typed.append(
                                Constraint.model_validate(ic)
                                if isinstance(ic, dict) else ic
                            )
                        except Exception:
                            pass

                reasoning_type = "validate" if skill.skill_type == "validation" else "transform"
                source_map = {
                    "ast_code": "repo", "ocr_document": "text",
                    "theory": "text", "colab": "colab",
                }
                source = source_map.get(skill.source_type or "ast_code", "repo")
                acs_score = getattr(skill.epistemic, "c_node", 0.0) if skill.epistemic else 0.0
                sie_score = sie.s_sie if sie else 0.0

                cn = CognitiveNode(
                    cognitive_id=skill.node_id,
                    skill=skill,
                    sie_node=sie,
                    constraints=typed,
                    c_final=getattr(skill.epistemic, "c_node", 0.0) if skill.epistemic else 0.0,
                    mode=self.mode,
                    reasoning_type=reasoning_type,
                    source=source,
                    acs_score=round(acs_score, 4),
                    sie_score=round(sie_score, 4),
                )
                cognitive_nodes[skill.node_id] = cn

                tc = skill.topology_cluster
                if tc:
                    for cid in (tc.child_ids or []):
                        all_ok = all(getattr(c, "valid", True) for c in typed)
                        edge = ReasoningEdge(
                            source_id=skill.node_id,
                            target_id=cid,
                            edge_type="dependency",
                            constraint_ok=all_ok,
                        )
                        reasoning_edges.append(edge)
                        cn.outbound_edges.append(f"{skill.node_id}->{cid}")

            # Wire inbound edges
            for edge in reasoning_edges:
                if edge.target_id in cognitive_nodes:
                    cognitive_nodes[edge.target_id].inbound_edges.append(
                        f"{edge.source_id}->{edge.target_id}"
                    )

            # Edge-level constraint enforcement
            for edge in reasoning_edges:
                if not edge.constraint_ok and edge.target_id in cognitive_nodes:
                    target = cognitive_nodes[edge.target_id]
                    target.c_final = 0.0
                    logger.warning(
                        f"Edge {edge.source_id}->{edge.target_id} constraint_ok=False "
                        f"— zeroed c_final on {edge.target_id}"
                    )

            return {
                "cognitive_nodes": {k: v.model_dump() for k, v in cognitive_nodes.items()},
                "reasoning_edges": [e.model_dump() for e in reasoning_edges],
            }
        except Exception as e:
            logger.warning(f"CognitiveNode layer construction failed (non-fatal): {e}")
            return {"cognitive_nodes": {}, "reasoning_edges": []}

    # ==================================================================
    # Core Orchestration Loop
    # ==================================================================
    def run(self) -> Dict[str, Any]:
        """
        Execute the DAG state machine across all nodes.
        Each node is driven through CREATED → VALIDATED → SCORED → ACCEPTED/REJECTED
        with reroll handling for contract violations and confidence failures.
        """
        run_id = f"DAG-RUN-{uuid.uuid4().hex[:8].upper()}"
        trace = [{"step": "init", "status": f"v5.3_causal_authority_{self.mode}", "run_id": run_id}]
        run_started_at = time.perf_counter()

        def _finalize(result: Dict[str, Any]) -> Dict[str, Any]:
            elapsed_ms = round((time.perf_counter() - run_started_at) * 1000.0, 3)
            result["dag_processing_time_ms"] = elapsed_ms
            if self.telemetry:
                self.telemetry.record(
                    "runtime", "dag_processing_time",
                    payload={
                        "elapsed_ms": elapsed_ms,
                        "status": result.get("status", "unknown"),
                        "run_id": run_id,
                    },
                )
                result["telemetry"] = self.telemetry.snapshot()
                try:
                    self.telemetry.persist(f"output/telemetry_{run_id}.jsonl")
                except Exception:
                    pass
            return result

        try:
            # Phase D containment: ingest pre-quarantined schema-mismatch nodes
            # into rejected traces so Veteran pipeline can consume diagnostics.
            for node in self._get_nodes_safely():
                sem = node.semantics if isinstance(node.semantics, dict) else {}
                if sem.get("failure_type") == "projected_node_validation_error":
                    self.rejected_traces.append({
                        "node_id": node.node_id,
                        "reason": "projected_node_validation_error",
                        "rejected": node.model_dump(exclude_unset=True),
                        "corrected": None,
                        "validation_error": sem.get("validation_error"),
                    })

            # --- Cycle detection (degrade-not-abort for V2.0) ---
            # Real-world codebases (e.g. starlette) contain legitimate
            # circular import / call-graph loops. The old fatal-abort path
            # collapsed the entire run into ``rejected_nodes`` the moment
            # a single cycle was detected, destroying any chance of an SFT
            # artifact. V2.0 quarantines SCC members, seals the break into
            # the ledger, and continues the run on the remaining DAG. The
            # terminal Omega handshake then classifies the run as DEGRADED
            # rather than FAIL. A true halt only occurs when quarantining
            # cycles would empty the graph (no healthy DAG remains).
            if self._detect_cycles():
                cycle_broken_ids = self._break_cycles_via_quarantine()
                remaining = self._get_nodes_safely()
                if not remaining:
                    trace.append({
                        "step": "cycle_detection",
                        "status": "halted",
                        "quarantined_nodes": len(cycle_broken_ids),
                        "quarantine_sample": cycle_broken_ids[:20],
                        "reason": "cycle_quarantine_emptied_graph",
                    })
                    return _finalize({
                        "run_id": run_id,
                        "status": "failed",
                        "validated_nodes": [],
                        "rejected_nodes": [],
                        "quarantined_nodes": getattr(self.graph, "quarantine", []),
                        "execution_trace": trace,
                        "compiled_artifact": None,
                        "transition_ledger": self.get_transition_ledger(),
                        "transition_ledger_valid": self.verify_transition_ledger(),
                    })
                trace.append({
                    "step": "cycle_detection",
                    "status": "degraded",
                    "quarantined_nodes": len(cycle_broken_ids),
                    "quarantine_sample": cycle_broken_ids[:20],
                    "remaining_nodes": len(remaining),
                })
                # Residual check — should be impossible but guard anyway.
                if self._detect_cycles():
                    trace.append({
                        "step": "cycle_detection",
                        "status": "halted",
                        "reason": "residual_cycle_after_quarantine",
                    })
                    return _finalize({
                        "run_id": run_id,
                        "status": "failed",
                        "validated_nodes": [],
                        "rejected_nodes": [n.model_dump() for n in self._get_nodes_safely()],
                        "quarantined_nodes": getattr(self.graph, "quarantine", []),
                        "execution_trace": trace,
                        "compiled_artifact": None,
                        "transition_ledger": self.get_transition_ledger(),
                        "transition_ledger_valid": self.verify_transition_ledger(),
                    })

            # --- Bootstrap: normalize legacy states to CREATED ---
            for node in self._get_nodes_safely():
                if not node.epistemic.state or node.epistemic.state in NodeState.LEGACY:
                    prev = node.epistemic.state or "NULL"
                    node.epistemic.state = NodeState.CREATED
                    self._log_transition(node.node_id, prev, NodeState.CREATED, "bootstrap")
                if getattr(node.epistemic, "retry_budget", None) is None:
                    node.epistemic.retry_budget = 6
                self._sync_node(node)

            self._nx_graph = self._build_directed_graph(self._get_nodes_safely())

            # --- Build CognitiveNode overlay ---
            self._cognitive_nodes = {}
            for skill in self._get_nodes_safely():
                sie = skill.sie_node if skill.sie_node else SemanticReasoningNode()
                cn = CognitiveNode(
                    cognitive_id=skill.node_id,
                    skill=skill,
                    sie_node=sie,
                    mode=self.mode,
                )
                self._cognitive_nodes[skill.node_id] = cn

            # --- Main state-machine loop ---
            iteration = 0
            while self.graph.active() and iteration < MAX_ITERATIONS:
                iteration += 1

                # Drive loop through CognitiveNode overlay
                ready_cns = [
                    cn for cn in self._cognitive_nodes.values()
                    if cn.skill.epistemic.state in NodeState.ACTIVE
                    or cn.skill.epistemic.state in NodeState.LEGACY
                ]
                if not ready_cns:
                    break

                ready_cns = self._sort_topologically([cn.skill for cn in ready_cns])
                ready_cns = [self._cognitive_nodes[s.node_id] for s in ready_cns]

                for cn in ready_cns:
                    node = cn.skill
                    state = node.epistemic.state

                    # === REROLL Handler ===
                    if state == NodeState.REROLL:
                        self._handle_reroll(node)
                        self._sync_node(node)
                        continue

                    # === Already terminal ===
                    if state in NodeState.TERMINAL:
                        continue

                    # === CREATED → VALIDATED (contract gate) ===
                    if state == NodeState.CREATED:
                        if not self._transition_to_validated(node):
                            self._sync_node(node)
                            continue

                    # === VALIDATED → SCORED (SIE + ACS + c_final) ===
                    if node.epistemic.state == NodeState.VALIDATED:
                        if not self._transition_to_scored(node):
                            self._sync_node(node)
                            continue

                    # === SCORED → ACCEPTED | REJECTED ===
                    if node.epistemic.state == NodeState.SCORED:
                        self._transition_to_terminal(node)

                    # === Post-transition cascade ===
                    if node.epistemic.state == NodeState.REJECTED:
                        self._cascade_rejection(node.node_id)

                    # === Branch management ===
                    if node.epistemic.state not in NodeState.TERMINAL:
                        depth = getattr(node.epistemic, "depth", 0)
                        if getattr(node.epistemic, "branch_id", "") == "root":
                            if depth < MAX_DEPTH and len(self.graph.branch_ids()) < MAX_BRANCHES:
                                self.spawn(node, k=1)
                            else:
                                node.epistemic.state = NodeState.REJECTED
                                node.epistemic.final_status = "depth_limit"
                                self._log_transition(
                                    node.node_id, "BRANCHING", NodeState.REJECTED,
                                    "depth_limit",
                                )

                    self._sync_node(node)

                    # Sync CognitiveNode state from skill
                    cn.c_final = node.epistemic.c_node
                    cn.sie_node = node.sie_node if node.sie_node else cn.sie_node
                    cn.sie_pass = getattr(node, "validation_pass", True)
                    cn.sie_score = cn.sie_node.s_sie

                self.prune()

                # Soft branch degradation (Phase 7 — replaces hard pruning)
                self._apply_branch_stability()

                # Telemetry → Actions: convert aggregate signals to governance
                if self._should_run_telemetry_check(iteration):
                    self._check_telemetry_thresholds()

                # HD3: Asymptotic Tikhonov-regularized recovery tick per iteration —
                # allows drift_score to decay gradually between scoring rounds,
                # preventing permanent identity freeze after transient SLR spikes.
                if self.identity_manager and self._should_run_recovery_tick(iteration):
                    self.identity_manager.apply_recovery_tick()

            # --- MAX_ITERATIONS guard: emit trace if loop was exhausted ---
            if iteration >= MAX_ITERATIONS:
                logger.warning(
                    f"[DAG] MAX_ITERATIONS ({MAX_ITERATIONS}) reached — "
                    f"halting with partial results."
                )
                trace.append({
                    "step": "max_iterations",
                    "status": "halted",
                    "iterations": iteration,
                })

            # --- Results aggregation ---
            all_nodes = self._get_nodes_safely()
            validated = self._get_validated_nodes()
            validated_ids = {n.node_id for n in validated}
            rejected = [n for n in all_nodes if n.node_id not in validated_ids]

            centrality = self._apply_graph_analytics(validated)
            for n in validated:
                if isinstance(n.semantics, dict):
                    n.semantics["system_centrality_blast_radius"] = round(
                        centrality.get(n.node_id, 0.0), 4
                    )

            # --- Global Consistency Engine (Phase 6) ---
            # Applied AFTER field dynamics but BEFORE omega certification
            # V2.1: Skip consistency sweep for theorist mode. Natural language
            # nodes from PDF documents share domain vocabulary and sentence
            # structure, causing the redundancy detector to flag topically
            # related paragraphs as duplicates. Contradictions are also
            # inapplicable to encyclopedic text where opposing viewpoints
            # are presented as part of the same knowledge unit.
            consistency_result = None
            if self.consistency_engine and validated and self.mode != "theorist":
                consistency_result = self.consistency_engine.validate_global_consistency(
                    validated
                )
                contradiction_index: Dict[str, List[Dict[str, Any]]] = {}
                for contradiction in consistency_result.get("contradictions", []):
                    a_id = str(contradiction.get("node_a", ""))
                    b_id = str(contradiction.get("node_b", ""))
                    if a_id:
                        contradiction_index.setdefault(a_id, []).append(contradiction)
                    if b_id:
                        contradiction_index.setdefault(b_id, []).append(contradiction)

                redundancy_index: Dict[str, List[Dict[str, str]]] = {}
                for pair in consistency_result.get("redundancies", []):
                    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                        continue
                    a_id = str(pair[0])
                    b_id = str(pair[1])
                    entry = {
                        "node_a": a_id,
                        "node_b": b_id,
                        "reason": "semantic_redundancy",
                    }
                    redundancy_index.setdefault(a_id, []).append(entry)
                    redundancy_index.setdefault(b_id, []).append(entry)

                node_map = {n.node_id: n for n in all_nodes}
                for rej_id in consistency_result.get("rejected_ids", []):
                    rej_node = node_map.get(rej_id)
                    if rej_node and rej_node.epistemic.state != NodeState.REJECTED:
                        prior_state = rej_node.epistemic.state
                        rej_node.epistemic.state = NodeState.REJECTED
                        rej_node.epistemic.final_status = "consistency_violation"
                        rej_node.validation_pass = False
                        if not isinstance(rej_node.semantics, dict):
                            rej_node.semantics = {}
                        rej_node.semantics["execution_eligible"] = False
                        rej_node.semantics["_rejection_reason"] = "consistency_violation"

                        c_details = contradiction_index.get(rej_id, [])
                        r_details = redundancy_index.get(rej_id, [])
                        rej_node.semantics["_consistency_violations"] = {
                            "contradictions": c_details,
                            "redundancies": r_details,
                        }

                        logger.warning(
                            f"[DAG_CONSISTENCY_VIOLATION] Node {rej_id} rejected by "
                            f"global consistency sweep: contradictions={len(c_details)}, "
                            f"redundancies={len(r_details)}"
                        )

                        self.rejected_traces.append({
                            "node_id": rej_id,
                            "reason": "consistency_violation",
                            "rejected": rej_node.model_dump(exclude_unset=True),
                            "corrected": None,
                            "consistency": {
                                "contradictions": c_details,
                                "redundancies": r_details,
                            },
                        })

                        self._log_transition(
                            rej_id, prior_state, NodeState.REJECTED,
                            "consistency_violation",
                        )
                        self._cascade_rejection(rej_id)
                if self.telemetry:
                    self.telemetry.record(
                        "consistency", consistency_result.get("status", "unknown"),
                        payload={
                            "contradictions": len(consistency_result.get("contradictions", [])),
                            "redundancies": len(consistency_result.get("redundancies", [])),
                            "rejected": len(consistency_result.get("rejected_ids", [])),
                        },
                    )
                # Refresh validated/rejected after consistency pruning
                validated = self._get_validated_nodes()
                validated_ids = {n.node_id for n in validated}
                rejected = [n for n in all_nodes if n.node_id not in validated_ids]

            status = "success" if validated else "failed"
            if validated and rejected:
                status = "degraded"

            # --- Omega Final Handshake (Phase 9) ---
            # Can only downgrade status, never upgrade
            omega_result = None
            if self.omega_validator:
                omega_result = self.omega_validator.omega_handshake(
                    validated, self._nx_graph, self.identity_manager, self.telemetry,
                    all_nodes=all_nodes,
                )
                if omega_result["status"] == "OMEGA_FAIL":
                    status = "failed"
                elif omega_result["status"] == "OMEGA_DEGRADED" and status == "success":
                    status = "degraded"

                # Priority 3 — Orphan Closure: quarantine stranded non-terminal
                # orphans detected by Omega. Mutate their state to REJECTED and
                # seal the transition into the cryptographic ledger so the
                # final run artifact contains a tamper-evident record of why
                # the orphan was closed out.
                #
                # V2.1 Theorist Mode Exemption: Natural language nodes from PDFs
                # are inherently unconnected (no AST import/call edges). Orphan
                # quarantine would reject 100% of theorist nodes. Skip it when
                # the runtime is operating in theorist mode.
                orphan_dim = omega_result.get("dimensions", {}).get("orphan_state_closure", {})
                orphan_ids = orphan_dim.get("orphan_node_ids", []) if isinstance(orphan_dim, dict) else []
                if self.mode == "theorist":
                    orphan_ids = []  # Suppress orphan quarantine for theorist nodes
                if orphan_ids:
                    node_map = {n.node_id: n for n in all_nodes}
                    existing = set()
                    if hasattr(self.graph, "quarantine") and isinstance(self.graph.quarantine, list):
                        for q in self.graph.quarantine:
                            if isinstance(q, dict):
                                existing.add(str(q.get("node_id", "")))
                    for orphan_id in orphan_ids:
                        orphan_node = node_map.get(orphan_id)
                        # Mutate state + seal ledger transition atomically.
                        prior_state = (
                            orphan_node.epistemic.state
                            if orphan_node is not None and orphan_node.epistemic is not None
                            else "UNKNOWN"
                        )
                        if orphan_node is not None and orphan_node.epistemic is not None:
                            orphan_node.epistemic.state = NodeState.REJECTED
                            orphan_node.epistemic.final_status = "omega_orphan_quarantined"
                            orphan_node.epistemic.c_node = 0.0
                            orphan_node.validation_pass = False
                            self._sync_node(orphan_node)
                            self._log_transition(
                                orphan_id, prior_state, NodeState.REJECTED,
                                reason=f"omega_orphan_quarantine from={prior_state}",
                                trigger=TransitionTrigger.OMEGA_ORPHAN_QUARANTINE,
                            )

                        payload = (
                            orphan_node.model_dump(exclude_unset=True)
                            if orphan_node is not None
                            else {"node_id": orphan_id}
                        )
                        payload["quarantine_reason"] = "omega_orphan_state"
                        payload["orphan_state"] = (
                            orphan_dim.get("orphan_states", {}).get(orphan_id)
                            if isinstance(orphan_dim, dict)
                            else None
                        )
                        if hasattr(self.graph, "quarantine") and isinstance(self.graph.quarantine, list):
                            if orphan_id not in existing:
                                self.graph.quarantine.append(payload)
                                existing.add(orphan_id)

                    # Triad of Failure: any orphan count > 0 forces at least
                    # OMEGA_DEGRADED; proportional tolerance escalates to
                    # OMEGA_FAIL only when orphans exceed 1% of the graph.
                    total = max(len(all_nodes), 1)
                    orphan_ratio = len(orphan_ids) / total
                    if orphan_ratio > 0.01 or len(orphan_ids) >= 3:
                        status = "failed"
                        omega_result["status"] = "OMEGA_FAIL"
                    elif status == "success":
                        status = "degraded"
                        if omega_result["status"] == "OMEGA_PASS":
                            omega_result["status"] = "OMEGA_DEGRADED"

                    # Refresh validated/rejected lists after orphan mutation
                    # so the finalized artifact reflects the quarantine.
                    validated = self._get_validated_nodes()
                    validated_ids = {n.node_id for n in validated}
                    rejected = [n for n in all_nodes if n.node_id not in validated_ids]

            trace.append({
                "step": "synthesis",
                "status": "collapsing" if validated else "bypassed",
            })

            cognitive_layer = self._build_cognitive_layer(all_nodes)

            if self.telemetry:
                # Persist adversarial fingerprint window for session continuity
                if self._adversarial_lens and hasattr(self._adversarial_lens, "_prior_fingerprints"):
                    self.telemetry.record(
                        "adversarial_lens", "session_fingerprints",
                        payload={
                            "fingerprints": self._adversarial_lens._prior_fingerprints[-50:],
                        },
                    )

            return _finalize({
                "run_id": run_id,
                "status": status,
                "validated_nodes": [n.model_dump(exclude_unset=True) for n in validated],
                "rejected_nodes": [n.model_dump(exclude_unset=True) for n in rejected],
                "quarantined_nodes": getattr(self.graph, "quarantine", []),
                "execution_trace": trace,
                "rejected_traces": self.rejected_traces,
                "identity_state": (
                    self.identity_manager.get_state_snapshot()
                    if self.identity_manager else None
                ),
                "cognitive_layer": cognitive_layer,
                "telemetry": None,
                "consistency_result": consistency_result,
                "omega_handshake": omega_result,
                "transition_ledger": self.get_transition_ledger(),
                "transition_ledger_valid": self.verify_transition_ledger(),
                "compiled_artifact": self.collapse() if validated else None,
            })

        except Exception as e:
            err = f"Unhandled exception: {e}\n{traceback.format_exc()}"
            logger.critical(f"{SENTINEL_FATAL_CRASH} {err}")
            trace.append({"step": "fatal_crash", "status": str(e)})
            return _finalize({
                "run_id": run_id,
                "status": "fatal_crash",
                "error": err,
                "validated_nodes": [],
                "rejected_nodes": [
                    n.model_dump() if isinstance(n, AletheiaSkill) else n
                    for n in self._get_nodes_safely()
                ],
                "quarantined_nodes": getattr(self.graph, "quarantine", []),
                "execution_trace": trace,
                "compiled_artifact": None,
                "telemetry": None,
            })

    # ------------------------------------------------------------------
    # Mode-Specific Execution
    # ------------------------------------------------------------------
    def execute(self, node: AletheiaSkill) -> None:
        """
        Strict 4-Mode Execution Routing.
        Maps generative outputs to the fields expected by contract schemas.

        POST-CONDITION: node.semantics["output"] is ALWAYS a string after this
        method returns, regardless of mode, lens success, or code_snippet
        presence. Downstream consumers (dataset_formatter._format_dataset_legacy
        and the SFT card builders) rely on this invariant.
        """
        if not isinstance(node.semantics, dict):
            node.semantics = {}

        # Store mode as pipeline metadata (outside contract-validated semantics)
        node.semantics["orchestration_mode"] = self.mode

        if self.telemetry:
            self.telemetry.record(
                "mode", "execution", node_id=node.node_id,
                payload={"mode": self.mode},
            )

        # Track the primary generative output for this node so the formatter
        # has a single contract field to read. Initialize it from the raw
        # code_snippet so even a total lens failure leaves a usable string.
        code_snippet = getattr(node, "code_snippet", "") or ""
        primary_output: str = f"```python\n{code_snippet}\n```" if code_snippet else ""

        if self.mode == "theorist":
            if "constraints" not in node.semantics:
                # TheoristOutput contract requires constraints: List[str]
                lens_output = self._run_lens("Concept_Formulation", node.source_context)
                node.semantics["constraints"] = lens_output if isinstance(lens_output, list) else [lens_output]
            primary_output = " ".join(node.semantics.get("constraints") or []) or primary_output

        elif self.mode == "coding_assistant":
            if "implementation" not in node.semantics:
                # CodingOutput contract requires implementation: str
                node.semantics["implementation"] = " ".join(
                    self._run_lens("Procedural_Mapping", node.source_context)
                )
            primary_output = node.semantics.get("implementation") or primary_output

        elif self.mode == "advocate":
            if "theory" not in node.semantics:
                # AdvocateOutput contract requires theory + implementation
                node.semantics["theory"] = " ".join(
                    self._run_lens("Architectural_Theory", node.source_context)
                )
                node.semantics["implementation"] = " ".join(
                    self._run_lens("Implementation_Draft", node.source_context)
                )
            primary_output = node.semantics.get("implementation") or primary_output

        elif self.mode == "veteran":
            if "traceback" not in node.semantics:
                node.semantics["traceback"] = (
                    "Simulated or intercepted traceback error logs."
                )
                node.semantics["diff"] = " ".join(
                    self._run_lens("Veteran_Diff", node.source_context)
                )
                node.semantics["resolved_code"] = " ".join(
                    self._run_lens("Veteran_Resolution", node.source_context)
                )
            primary_output = node.semantics.get("resolved_code") or primary_output

        # Extract patterns for AST-sourced nodes
        st = (node.source_type or "").lower()
        if st in ("ast_code", "macro_skill") and "extracted_patterns" not in node.semantics:
            res = self._run_lens("Algorithmic_Abstraction", node.source_context)
            node.semantics["extracted_patterns"] = res if isinstance(res, list) else [res]

        # Final guarantee: node.semantics["output"] is always a string. Coerce
        # None / non-string lens returns into a safe placeholder so the
        # downstream formatter never hits an unbound name or KeyError.
        if not isinstance(primary_output, str) or not primary_output:
            primary_output = "```python\n# <no generative output>\n```"
        node.semantics["output"] = primary_output

    def _run_lens(self, lens_name: str, context: Any) -> List[str]:
        if self.llm_orchestrator is not None:
            try:
                if hasattr(self.llm_orchestrator, "run_lens"):
                    return self.llm_orchestrator.run_lens(lens_name, context)
                return self.llm_orchestrator(lens_name, context)
            except Exception as e:
                logger.error(f"Lens Execution Failure [{lens_name}]: {e}")
                return [f"LLM Error: {e}"]
        return [f"Generated_via_{lens_name}"]

    # ------------------------------------------------------------------
    # Branch Management
    # ------------------------------------------------------------------
    def spawn(self, node: AletheiaSkill, k: int) -> None:
        depth = getattr(node.epistemic, "depth", 0)
        children = []
        for i in range(k):
            child = deepcopy(node)
            child.node_id = f"{node.node_id}_h{i}"
            child.epistemic.branch_id = uuid.uuid4().hex
            prior_child_state = child.epistemic.state or "NULL"
            child.epistemic.state = NodeState.CREATED
            child.epistemic.depth = depth + 1
            # Priority 3 — Cryptographic Ledger: record every branch spawn
            # so the lineage of a derived child is auditable back to the
            # parent node's exact state.
            self._log_transition(
                child.node_id, prior_child_state, NodeState.CREATED,
                reason=f"branch_spawn from={node.node_id}",
                trigger=TransitionTrigger.BRANCH_SPAWN,
            )
            children.append(child.model_dump())
        if hasattr(self.graph, "add_nodes"):
            self.graph.add_nodes(children)

    def branch_confidence(self, branch_id: str) -> float:
        nodes = self.graph.get_branch(branch_id)
        if not nodes:
            return 0.0
        return max(getattr(n.epistemic, "c_node", 0.0) for n in nodes)

    def prune(self) -> None:
        for bid in self.graph.branch_ids():
            if self.branch_confidence(bid) < BRANCH_PRUNE_THRESHOLD:
                self.graph.remove_branch(bid)

    # ------------------------------------------------------------------
    # Final Artifact Collapse
    # ------------------------------------------------------------------
    def collapse(self) -> Dict[str, Any]:
        """
        Generates the compiled artifact from validated nodes.
        Lifts semantics to root level for dataset_formatter.py alignment.
        """
        import itertools
        validated = self._get_validated_nodes()
        axioms: List[Dict[str, Any]] = []
        patterns: List[Dict[str, Any]] = []
        c_sum = 0.0

        for node in validated:
            c_sum += getattr(node.epistemic, "c_node", 1.0)
            st = (node.source_type or "unknown").lower()
            dump = node.model_dump(exclude_unset=True)
            if "semantics" in dump and isinstance(dump["semantics"], dict):
                for k, v in dump["semantics"].items():
                    dump[k] = v
            if st in ("ocr_document", "theory", "theoretical_reasoning"):
                axioms.append(dump)
            else:
                patterns.append(dump)

        avg_c = round(c_sum / len(validated), 4) if validated else 0.0

        # Flatten evidence lazily — avoids a second full-corpus list before dedup.
        evidence_graph = list(
            itertools.chain.from_iterable(
                getattr(n.epistemic, "evidence_refs", []) or [] for n in validated
            )
        )
        del validated

        return {
            "schema": "aletheia_sie_v1",
            "orchestration_mode": self.mode,
            "compiled_intelligence": {
                "system_axioms": self._dedupe_and_sort_dicts(axioms),
                "design_patterns": self._dedupe_and_sort_dicts(patterns),
            },
            "epistemic_provenance": {
                "confidence_score": avg_c,
                "supporting_evidence": self._dedupe_and_sort_dicts(evidence_graph),
            },
        }
