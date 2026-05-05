"""
Aletheia Fallback Stubs — Graceful degradation layer.

When the full pipeline packages (src.core.models, src.pipeline.contracts, etc.)
are unavailable, these minimal Pydantic stubs allow dag_runtime.py to bootstrap
in standalone or partial-import environments without crashing.

Extracted from dag_runtime.py (Sprint 4a — risk-002).
"""

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict, field_validator


# ── Graph / ACS ──────────────────────────────────────────────────────

class EpistemicGraphInterface:
    pass


class ACSExecutionGovernor:
    def evaluate_node(self, node: Any) -> Any:
        return node


# ── Core Pydantic models ─────────────────────────────────────────────

class TopologyCluster(BaseModel):
    downstream_calls: List[str] = Field(default_factory=list)
    upstream_callers: List[str] = Field(default_factory=list)


class EpistemicState(BaseModel):
    state: str = "unresolved"
    c_node: float = 0.0
    confidence: Optional[Dict[str, Any]] = None
    retry_budget: int = 6
    depth: int = 0
    branch_id: str = "root"
    evidence_refs: List[Any] = Field(default_factory=list)
    final_status: Optional[str] = None


class TeachingLayer(BaseModel):
    pass


class AletheiaSkill(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    node_id: str
    source_context: str = ""
    source_type: str = "unknown"
    skill_type: str = "unknown"
    semantics: dict = {}
    epistemic: EpistemicState = Field(default_factory=EpistemicState)
    dependencies: TopologyCluster = Field(
        default_factory=TopologyCluster, alias="topology_cluster",
    )
    validation_pass: bool = False
    v_score: float = 0.0
    acs_handshake_sid: str = ""
    acs_violations: List[str] = Field(default_factory=list)
    acs_audited: bool = False
    sie_node: Any = None
    constraints: Any = None
    topology_cluster: Any = None

    @field_validator('source_context', mode='before')
    @classmethod
    def coerce_source_context(cls, v):
        if isinstance(v, dict):
            return json.dumps(v)
        return v if v is not None else ""


class SemanticReasoningNode(BaseModel):
    content_density: float = 0.0
    alignment_vector: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    composite_quality_score: float = 0.0
    mode_scaling_factor: float = 1.0
    s_sie: float = 0.0


class CognitiveNode(BaseModel):
    cognitive_id: str
    skill: Any = None
    sie_node: Any = None
    constraints: List[Any] = Field(default_factory=list)
    inbound_edges: List[str] = Field(default_factory=list)
    outbound_edges: List[str] = Field(default_factory=list)
    c_final: float = 0.0
    mode: str = "advocate"
    reasoning_type: str = "transform"
    source: str = "repo"
    acs_score: float = 0.0
    sie_score: float = 0.0


class ReasoningEdge(BaseModel):
    source_id: str
    target_id: str
    edge_type: str = "dependency"
    constraint_ok: bool = True
    weight: float = 1.0


class Constraint(BaseModel):
    type: str = "structural"
    description: str = ""
    valid: bool = True
    severity: str = "warning"
    tags: List[str] = Field(default_factory=list)


class IdentityState(BaseModel):
    drift_score: float = 0.0
    frozen: bool = False


# ── Utility stubs ────────────────────────────────────────────────────

def stable_json_dumps(obj):
    return json.dumps(obj, sort_keys=True)


MODE_CONTRACTS: Dict[str, Any] = {}


def validate_contract(output, schema):
    return (True, None)


# ── Engine stubs ─────────────────────────────────────────────────────

class IdentityManager:
    def __init__(self, **kw):
        self.drift_score = 0.0

    def update_on_slr_breach(self, *a, **kw):
        pass

    def check_and_enforce(self, *a, **kw):
        return False

    def update_from_trajectory(self, *a):
        pass

    def get_state_snapshot(self):
        return {}


class RerollEngine:
    def __init__(self):
        self.reroll_count = 0

    def reroll(self, *a, **kw):
        return None


class TelemetryCollector:
    def __init__(self):
        pass

    def record(self, *a, **kw):
        pass

    def snapshot(self):
        return {}

    def persist(self, *a):
        pass


# ── Adversarial Lens stub ─────────────────────────────────────────────

class AdversarialLens:
    def __init__(self, **kw):
        pass

    def evaluate(self, node, primary_c_final):
        from dataclasses import dataclass, field as df
        @dataclass
        class _Verdict:
            node_id: str = ""
            adversarial_score: float = 0.0
            primary_score: float = 0.0
            conflict_magnitude: float = 0.0
            conflict_class: str = "agreement"
            flags: list = df(default_factory=list)
            structural_fingerprint: str = ""
        return _Verdict(node_id=getattr(node, 'node_id', ''), primary_score=primary_c_final)


SENTINEL_ALL_CONFLICT = "[ALL-CONFLICT]"


# ── Firewall stubs ───────────────────────────────────────────────────

class DriftViolation(Exception):
    pass


def enforce_semantic_firewall(text, context=""):
    pass
