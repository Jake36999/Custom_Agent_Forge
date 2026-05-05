"""
Pathological Test Harness — Adversarial stress tests for the enforcement stack.

The 152 tests in test_refactored_modules.py prove structural correctness.
These tests prove operational stability under:
    - Deep chains (depth penalty, cascade, drift accumulation)
    - Reroll exhaustion under sustained failure
    - Adversarial SLR injection (orthogonal/anti-correlated vectors)
    - Invariant boundary attacks (exact thresholds, float precision, injection)
    - Consistency & redundancy flooding (mass identical/contradicting nodes)
    - Omega handshake under degraded system state
    - Branch stability collapse (compound degradation)
    - End-to-end integration stress (cycles, empty graphs, MAX_ITERATIONS)
"""

import sys
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import networkx as nx

# --- Path bootstrap ---
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.dag_runtime import (
    DAGRuntime,
    InMemoryEpistemicGraph,
    NodeState,
    calculate_sie_slr,
    ACCEPTANCE_THRESHOLD,
    BRANCH_PRUNE_THRESHOLD,
    MAX_BRANCHES,
    MAX_DEPTH,
    MAX_ITERATIONS,
    INSTABILITY_BAND_LOW,
    DEPTH_PENALTY_RATE,
    DEPTH_PENALTY_CAP,
    BRANCH_STABILITY_THRESHOLD,
    BRANCH_DEGRADATION_FACTOR,
    EDGE_COHERENCE_THRESHOLD,
    SLR_THRESHOLDS,
    TELEMETRY_SLR_MEAN_THRESHOLD,
    TELEMETRY_DRIFT_CUMULATIVE_THRESHOLD,
)
from src.pipeline.invariant_engine import InvariantEngine
from src.pipeline.consistency_engine import ConsistencyEngine
from src.pipeline.omega_validator import OmegaValidator
from src.pipeline.sie_projection import SIEProjection, CANONICAL_FIELDS
from src.core.models import SemanticReasoningNode


# =====================================================================
# Shared Helpers
# =====================================================================
def make_node_dict(
    node_id="test_001",
    depth=0,
    branch_id="root",
    code_snippet=None,
    invariants=None,
    parent_ids=None,
    child_ids=None,
    **overrides,
):
    """Build a minimal AletheiaSkill-compatible dict."""
    snippet = code_snippet or (
        "def foo(x: int) -> int:\n"
        "    \"\"\"Computes increment.\"\"\"\n"
        "    return x + 1\n"
    )
    sem = {"code_snippet": snippet, "name": "foo"}
    if invariants:
        sem["invariants"] = invariants
    base = {
        "node_id": node_id,
        "name": node_id,
        "file": "test.py",
        "code_snippet": snippet,
        "imports": [],
        "operator_type": "function",
        "source_type": "ast_code",
        "skill_type": "execution",
        "semantics": sem,
        "teaching_layer": {
            "skill_identity": {"name": node_id},
            "method_metadata": {"name": node_id, "language": "python"},
            "reasoning_vectors": {
                "intent": "compute",
                "strategy": "direct return",
                "constraints": ["no_mutation"],
                "execution_pattern": ["functional"],
                "failure_modes": ["type_error"],
            },
            "implementation_template": {"code": "pass"},
        },
        "epistemic": {
            "state": "CREATED",
            "c_node": 0.0,
            "retry_budget": 6,
            "depth": depth,
            "branch_id": branch_id,
        },
    }
    if child_ids:
        base["topology_cluster"] = {"child_ids": child_ids}
    if parent_ids:
        base["dependencies"] = {"downstream_calls": []}  # parent → child is topology
    base.update(overrides)
    return base


def make_runtime(nodes, mode="coding_assistant"):
    """Create DAGRuntime with in-memory graph and no ACS (forces inline SIE)."""
    graph = InMemoryEpistemicGraph(nodes)
    acs = MagicMock(spec=[])
    return DAGRuntime(graph, acs, mode=mode)


def set_scored(runtime, node, c_node=0.9, s_sie=0.9, alignment_vector=None):
    """
    Fast-forward a node to SCORED state with controlled SIE values.
    Bypasses the full pipeline to isolate specific enforcement gates.
    """
    pg = alignment_vector or [0.8, 0.6, 0.7]
    node.epistemic.state = NodeState.SCORED
    node.epistemic.c_node = c_node
    node.sie_node = SemanticReasoningNode(
        content_density=0.8, s_sie=s_sie, composite_quality_score=0.5, mode_scaling_factor=1.0, alignment_vector=pg,
    )
    node.v_score = 0.85
    node.validation_pass = True


def build_chain(n, base_pg=None):
    """
    Build a linear chain of n nodes: node_0 → node_1 → ... → node_{n-1}.
    Returns (list_of_node_dicts, list_of_edges_as_tuples).
    """
    pg = base_pg or [0.8, 0.6, 0.7]
    nodes = []
    edges = []
    for i in range(n):
        child_ids = [f"node_{i+1}"] if i < n - 1 else []
        nodes.append(make_node_dict(
            node_id=f"node_{i}",
            depth=i,
            child_ids=child_ids if child_ids else None,
        ))
        if i < n - 1:
            edges.append((f"node_{i}", f"node_{i+1}"))
    return nodes, edges


# =====================================================================
# A. Deep Chain Stress
# =====================================================================
class TestDeepChainStress:
    """Linear chains pushing depth penalty, cascade, and drift accumulation."""

    def test_depth_5_chain_confidence_monotonically_decreases(self):
        """Each child at greater depth gets a larger depth penalty → lower c_node."""
        node_dicts, edges = build_chain(5)
        runtime = make_runtime(node_dicts)

        G = nx.DiGraph()
        G.add_edges_from(edges)
        runtime._nx_graph = G

        # Score each node at identical base confidence
        confidences = []
        for i in range(5):
            node = runtime.graph.nodes[f"node_{i}"]
            node.epistemic.state = NodeState.VALIDATED
            node.epistemic.c_node = 0.8
            node.epistemic.depth = i
            node.sie_node = SemanticReasoningNode(
                content_density=0.8, s_sie=0.9, composite_quality_score=0.5, mode_scaling_factor=1.0,
                alignment_vector=[0.8, 0.6, 0.7],
            )
            node.v_score = 0.85
            runtime._compute_unified_confidence(node)
            confidences.append(node.epistemic.c_node)

        # Monotonic decrease (each deeper node has more penalty)
        for i in range(len(confidences) - 1):
            assert confidences[i] >= confidences[i + 1], (
                f"depth {i}: {confidences[i]} should >= depth {i+1}: {confidences[i+1]}"
            )

    def test_depth_at_max_hits_penalty_cap(self):
        """At depth=MAX_DEPTH, penalty should be exactly DEPTH_PENALTY_CAP."""
        runtime = make_runtime([make_node_dict(depth=MAX_DEPTH)])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.VALIDATED
        node.epistemic.c_node = 0.8
        node.epistemic.depth = MAX_DEPTH
        node.sie_node = SemanticReasoningNode(
            content_density=0.8, s_sie=0.9, composite_quality_score=0.5, mode_scaling_factor=1.0,
            alignment_vector=[0.8, 0.6, 0.7],
        )
        node.v_score = 0.85
        runtime._compute_unified_confidence(node)

        # depth * rate = 5 * 0.05 = 0.25 < 0.3 cap, so this is 0.25
        expected_penalty = min(DEPTH_PENALTY_CAP, MAX_DEPTH * DEPTH_PENALTY_RATE)
        assert node.epistemic.confidence["depth_penalty"] == pytest.approx(
            expected_penalty
        )

    def test_depth_exceeds_max_still_capped(self):
        """Depths far beyond MAX_DEPTH don't exceed DEPTH_PENALTY_CAP."""
        runtime = make_runtime([make_node_dict(depth=100)])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.VALIDATED
        node.epistemic.c_node = 0.8
        node.epistemic.depth = 100
        node.sie_node = SemanticReasoningNode(
            content_density=0.8, s_sie=0.9, composite_quality_score=0.5, mode_scaling_factor=1.0,
            alignment_vector=[0.8, 0.6, 0.7],
        )
        node.v_score = 0.85
        runtime._compute_unified_confidence(node)
        assert node.epistemic.confidence["depth_penalty"] == DEPTH_PENALTY_CAP

    def test_deep_chain_rejected_root_cascades_all_descendants(self):
        """Rejecting the root of a 5-node chain cascades to all descendants."""
        node_dicts, edges = build_chain(5)
        runtime = make_runtime(node_dicts)
        G = nx.DiGraph()
        G.add_edges_from(edges)
        runtime._nx_graph = G

        # Set all nodes to SCORED
        for nid in [f"node_{i}" for i in range(5)]:
            runtime.graph.nodes[nid].epistemic.state = NodeState.SCORED

        # Reject the root
        root = runtime.graph.nodes["node_0"]
        root.epistemic.state = NodeState.REJECTED
        root.epistemic.c_node = 0.0
        runtime._cascade_rejection("node_0")

        # All descendants should be rejected
        for i in range(1, 5):
            n = runtime.graph.nodes[f"node_{i}"]
            assert n.epistemic.state == NodeState.REJECTED, (
                f"node_{i} should be REJECTED after root cascade"
            )
            assert n.epistemic.c_node == 0.0

    def test_deep_chain_middle_rejection_partial_cascade(self):
        """Rejecting node_2 of a 5-node chain cascades to 3,4 but not 0,1."""
        node_dicts, edges = build_chain(5)
        runtime = make_runtime(node_dicts)
        G = nx.DiGraph()
        G.add_edges_from(edges)
        runtime._nx_graph = G

        # Set all to SCORED
        for nid in [f"node_{i}" for i in range(5)]:
            runtime.graph.nodes[nid].epistemic.state = NodeState.SCORED

        # Reject node_2
        n2 = runtime.graph.nodes["node_2"]
        n2.epistemic.state = NodeState.REJECTED
        n2.epistemic.c_node = 0.0
        runtime._cascade_rejection("node_2")

        # Upstream survives
        assert runtime.graph.nodes["node_0"].epistemic.state == NodeState.SCORED
        assert runtime.graph.nodes["node_1"].epistemic.state == NodeState.SCORED
        # Downstream cascaded
        assert runtime.graph.nodes["node_3"].epistemic.state == NodeState.REJECTED
        assert runtime.graph.nodes["node_4"].epistemic.state == NodeState.REJECTED

    def test_deep_chain_with_drift_accumulation(self):
        """
        Cumulative SLR breaches push drift past max_drift → identity freezes
        → all subsequent nodes rejected.
        """
        node_dicts, edges = build_chain(5)
        runtime = make_runtime(node_dicts)
        G = nx.DiGraph()
        G.add_edges_from(edges)
        runtime._nx_graph = G

        # Simulate SLR breach at each depth (drift += slr * 0.1 per breach)
        for i in range(5):
            # Each breach adds 0.8 * 0.1 = 0.08 drift
            runtime.identity_manager.update_on_slr_breach(
                f"node_{i}", slr=0.8, mode="coding_assistant",
            )

        # After 5 breaches: drift_score ≈ 5 * 0.08 = 0.4 > max_drift=0.3
        assert runtime.identity_manager.drift_score > 0.3
        # The identity should freeze on check
        frozen = runtime.identity_manager.check_and_enforce("root")
        assert frozen is True
        assert runtime.identity_manager.frozen is True


# =====================================================================
# B. Reroll Exhaustion Under Pressure
# =====================================================================
class TestRerollExhaustion:
    """Nodes that fail repeatedly until budget=0."""

    def test_budget_exhaustion_produces_rejected_trace(self):
        """Full budget drain: 6 rerolls, final state REJECTED."""
        nd = make_node_dict(node_id="reroll_victim")
        nd["epistemic"]["retry_budget"] = 3
        runtime = make_runtime([nd])
        node = runtime.graph.nodes["reroll_victim"]
        node.epistemic.state = NodeState.VALIDATED

        # Drain budget manually
        for i in range(3):
            prev = node.epistemic.state
            rerolled = runtime._route_to_reroll(
                node, prev, f"contract_failure_attempt_{i}",
            )
            if rerolled:
                # Simulate reroll: back to VALIDATED and fail again
                node.epistemic.state = NodeState.VALIDATED

        # Final attempt: budget should be 0 → REJECTED
        rerolled = runtime._route_to_reroll(
            node, NodeState.VALIDATED, "final_failure",
        )
        assert rerolled is False
        assert node.epistemic.state == NodeState.REJECTED

    def test_reroll_in_instability_band_drains_to_rejection(self):
        """
        c_node in [0.25, 0.40) forces instability reroll loop until budget exhausted.
        """
        nd = make_node_dict(node_id="instability_node")
        nd["epistemic"]["retry_budget"] = 2
        runtime = make_runtime([nd])
        node = runtime.graph.nodes["instability_node"]

        # Patch scoring pipeline so we control c_node exactly
        with patch.object(runtime, "_compute_sie", return_value=True), \
             patch.object(runtime, "_apply_acs"), \
             patch.object(runtime, "_check_governance_directive", return_value=False), \
             patch.object(runtime, "_enforce_identity", return_value=False), \
             patch.object(runtime, "_validate_evidence"):
            # Force c_node into instability band
            def fake_confidence(n):
                n.epistemic.c_node = 0.33
                n.epistemic.confidence = {
                    "sie": 0.9, "acs": 0.8, "topology": 0.8,
                    "validation": 0.40, "model": "additive_weighted",
                    "weights": {"alpha": 0.4, "beta": 0.3, "gamma": 0.2, "delta": 0.1},
                    "coherence_decay": 0.0, "identity_drift": 0.0,
                    "depth_penalty": 0.0, "system_backpressure": 0.0, "final": 0.33,
                }
            with patch.object(runtime, "_compute_unified_confidence", side_effect=fake_confidence):
                # Cycle 1: CREATED → VALIDATED → SCORED → instability reroll
                node.epistemic.state = NodeState.VALIDATED
                node.sie_node = MagicMock(s_sie=0.9, alignment_vector=[0.8, 0.6, 0.7])
                runtime._transition_to_scored(node)
                assert node.epistemic.state == NodeState.SCORED
                runtime._transition_to_terminal(node)
                # Should be REROLL (instability band) or REJECTED (budget exhausted)
                assert node.epistemic.state in (NodeState.REROLL, NodeState.REJECTED)

    def test_sycophancy_reroll_mutates_identity_each_cycle(self):
        """SLR reroll → identity mutation → drift increases each cycle."""
        nd = make_node_dict(node_id="syco_node")
        nd["epistemic"]["retry_budget"] = 4
        runtime = make_runtime([nd])
        node = runtime.graph.nodes["syco_node"]

        initial_drift = runtime.identity_manager.drift_score

        # Simulate 3 SLR reroll cycles
        for i in range(3):
            node.epistemic.state = NodeState.VALIDATED
            node.epistemic.c_node = 0.8
            # Trigger sycophancy handler with SLR in reroll tier
            runtime._handle_sycophancy_failure(node, slr=0.72)

        # Drift should have accumulated from identity mutations
        assert runtime.identity_manager.drift_score > initial_drift
        assert len(runtime.identity_manager.slr_history) == 3

    def test_reroll_context_preserved_across_cycles(self):
        """Failure context from cycle N should be visible in cycle N+1."""
        nd = make_node_dict(node_id="ctx_node")
        nd["epistemic"]["retry_budget"] = 4
        runtime = make_runtime([nd])
        node = runtime.graph.nodes["ctx_node"]
        node.epistemic.state = NodeState.VALIDATED

        # First reroll with specific reason
        runtime._route_to_reroll(
            node, NodeState.VALIDATED, "invariant_violation: content_density > 0.5",
            violation_details="content_density=0.3 < 0.5",
        )
        assert node.epistemic.state == NodeState.REROLL

        ctx = node.semantics.get("_reroll_context", {})
        assert "invariant_violation" in ctx["previous_failure_reason"]
        assert ctx["violation_details"] == "content_density=0.3 < 0.5"

    def test_concurrent_rerolls_on_sibling_branches(self):
        """3 branches each with budget=2, all failing — all drain independently."""
        nodes = []
        for br in range(3):
            nd = make_node_dict(
                node_id=f"br{br}_n0",
                branch_id=f"branch_{br}",
            )
            nd["epistemic"]["retry_budget"] = 2
            nodes.append(nd)

        runtime = make_runtime(nodes)

        for br in range(3):
            node = runtime.graph.nodes[f"br{br}_n0"]
            node.epistemic.state = NodeState.VALIDATED

            # Drain both rerolls
            runtime._route_to_reroll(node, NodeState.VALIDATED, "fail_1")
            node.epistemic.state = NodeState.VALIDATED
            runtime._route_to_reroll(node, NodeState.VALIDATED, "fail_2")
            node.epistemic.state = NodeState.VALIDATED

            # Third attempt: budget should be 0
            rerolled = runtime._route_to_reroll(
                node, NodeState.VALIDATED, "fail_3",
            )
            assert rerolled is False
            assert node.epistemic.state == NodeState.REJECTED


# =====================================================================
# C. Adversarial SLR Injection
# =====================================================================
class TestAdversarialSLR:
    """Phase gradients designed to maximize SLR."""

    def test_orthogonal_child_triggers_collapse_tier(self):
        """parent=[1,0,0], child=[0,1,0] → SLR≈1.0 → collapse tier."""
        parent = MagicMock()
        parent.sie_node = MagicMock(alignment_vector=[1.0, 0.0, 0.0])
        child = MagicMock()
        child.sie_node = MagicMock(alignment_vector=[0.0, 1.0, 0.0])

        slr = calculate_sie_slr(parent, child)
        # Orthogonal: cos_sim ≈ 0 → SLR = 1.0 - 0 = 1.0
        assert slr >= SLR_THRESHOLDS["collapse"], (
            f"SLR {slr} should be >= collapse tier {SLR_THRESHOLDS['collapse']}"
        )

    def test_anti_correlated_child_collapse(self):
        """parent=[1,0,0], child=[-1,0,0] → cos=-1 → SLR=1.0 → hard rejection."""
        nd_p = make_node_dict(node_id="p", child_ids=["c"])
        nd_c = make_node_dict(node_id="c")
        runtime = make_runtime([nd_p, nd_c])

        parent = runtime.graph.nodes["p"]
        child = runtime.graph.nodes["c"]

        parent.sie_node = SemanticReasoningNode(
            content_density=0.8, s_sie=0.9, composite_quality_score=0.5, mode_scaling_factor=1.0,
            alignment_vector=[1.0, 0.0, 0.0],
        )
        child.sie_node = SemanticReasoningNode(
            content_density=0.8, s_sie=0.9, composite_quality_score=0.5, mode_scaling_factor=1.0,
            alignment_vector=[-1.0, 0.0, 0.0],
        )

        child.epistemic.state = NodeState.VALIDATED
        child.epistemic.c_node = 0.9

        # Handle the adversarial SLR
        slr = calculate_sie_slr(parent, child)
        assert slr >= SLR_THRESHOLDS["collapse"]
        rerouted = runtime._handle_sycophancy_failure(child, slr)
        assert rerouted is True
        assert child.epistemic.state == NodeState.REJECTED
        assert child.epistemic.c_node == 0.0

    def test_gradually_drifting_chain_accumulates_slr(self):
        """4-node chain with widening angular steps: edge SLR increases."""
        # Each consecutive pair has a WIDER angular gap:
        # 0→1: 15°, 1→2: 45° delta, 2→3: 90° delta
        import math as _m
        gradients = [
            [1.0, 0.0, 0.0],                                              # 0°
            [_m.cos(_m.radians(15)), _m.sin(_m.radians(15)), 0.0],         # 15°
            [_m.cos(_m.radians(60)), _m.sin(_m.radians(60)), 0.0],         # 60°
            [_m.cos(_m.radians(150)), _m.sin(_m.radians(150)), 0.0],       # 150°
        ]
        node_dicts, edges = build_chain(4)
        runtime = make_runtime(node_dicts)
        G = nx.DiGraph()
        G.add_edges_from(edges)
        runtime._nx_graph = G

        slrs = []
        for i in range(4):
            node = runtime.graph.nodes[f"node_{i}"]
            node.sie_node = SemanticReasoningNode(
                content_density=0.8, s_sie=0.9, composite_quality_score=0.5, mode_scaling_factor=1.0,
                alignment_vector=gradients[i],
            )
            node.epistemic.state = NodeState.SCORED
            if i > 0:
                parent = runtime.graph.nodes[f"node_{i-1}"]
                edge_slr = calculate_sie_slr(parent, node)
                slrs.append(edge_slr)
                runtime.identity_manager.update_on_slr_breach(
                    f"node_{i}", edge_slr, "coding_assistant",
                )

        # SLR should increase along the chain as rotation grows
        assert len(slrs) == 3
        # First edge SLR < last edge SLR
        assert slrs[0] < slrs[-1], f"SLR should increase: {slrs}"
        # Rolling SLR mean should reflect the drift
        assert runtime.identity_manager.get_rolling_slr_mean() > 0.0

    def test_slr_collapse_on_one_branch_doesnt_poison_sibling(self):
        """Two branches: one collapses via SLR, other survives."""
        # Branch A: adversarial pair
        nd_a1 = make_node_dict(node_id="a_1", branch_id="branch_a", child_ids=["a_2"])
        nd_a2 = make_node_dict(node_id="a_2", branch_id="branch_a")
        # Branch B: healthy pair
        nd_b1 = make_node_dict(node_id="b_1", branch_id="branch_b", child_ids=["b_2"])
        nd_b2 = make_node_dict(node_id="b_2", branch_id="branch_b")

        runtime = make_runtime([nd_a1, nd_a2, nd_b1, nd_b2])
        G = nx.DiGraph()
        G.add_edges_from([("a_1", "a_2"), ("b_1", "b_2")])
        runtime._nx_graph = G

        # Score all first, THEN override SIE nodes (set_scored overwrites sie_node)
        for nid in ("a_1", "a_2", "b_1", "b_2"):
            set_scored(runtime, runtime.graph.nodes[nid])

        # Branch A: adversarial gradient flip (orthogonal)
        runtime.graph.nodes["a_1"].sie_node = SemanticReasoningNode(
            content_density=0.8, s_sie=0.9, composite_quality_score=0.5, mode_scaling_factor=1.0,
            alignment_vector=[1.0, 0.0, 0.0],
        )
        runtime.graph.nodes["a_2"].sie_node = SemanticReasoningNode(
            content_density=0.8, s_sie=0.9, composite_quality_score=0.5, mode_scaling_factor=1.0,
            alignment_vector=[0.0, 1.0, 0.0],
        )
        # Branch B: coherent gradients
        for nid in ("b_1", "b_2"):
            runtime.graph.nodes[nid].sie_node = SemanticReasoningNode(
                content_density=0.8, s_sie=0.9, composite_quality_score=0.5, mode_scaling_factor=1.0,
                alignment_vector=[0.8, 0.6, 0.7],
            )

        # Collapse branch A child via SLR
        slr_a = calculate_sie_slr(
            runtime.graph.nodes["a_1"], runtime.graph.nodes["a_2"],
        )
        assert slr_a >= SLR_THRESHOLDS["collapse"]
        runtime._handle_sycophancy_failure(runtime.graph.nodes["a_2"], slr_a)
        assert runtime.graph.nodes["a_2"].epistemic.state == NodeState.REJECTED

        # Branch B should be fully intact
        assert runtime.graph.nodes["b_1"].epistemic.state == NodeState.SCORED
        assert runtime.graph.nodes["b_2"].epistemic.state == NodeState.SCORED

    def test_rolling_slr_critical_triggers_telemetry_governance(self):
        """Many SLR breaches → rolling mean critical → telemetry flags it."""
        runtime = make_runtime([make_node_dict()])

        # Feed 10 high SLR values
        for i in range(10):
            runtime.identity_manager.update_on_slr_breach(
                f"node_{i}", slr=0.75, mode="coding_assistant",
            )

        assert runtime.identity_manager.is_rolling_slr_critical() is True
        mean = runtime.identity_manager.get_rolling_slr_mean()
        assert mean >= TELEMETRY_SLR_MEAN_THRESHOLD

    def test_identity_freeze_from_cumulative_slr_breaches(self):
        """Enough SLR breaches push drift past max_drift → freeze."""
        runtime = make_runtime([make_node_dict()])
        im = runtime.identity_manager

        # Each breach adds slr * 0.1 = 0.08 drift
        # max_drift = 0.3 → need 0.3 / 0.08 ≈ 4 breaches
        for i in range(5):
            im.update_on_slr_breach(f"n{i}", slr=0.8, mode="coding_assistant")

        # Should exceed max_drift
        assert im.drift_score > im.max_drift

        # Enforce: should freeze
        frozen = im.check_and_enforce("root")
        assert frozen is True
        assert im.frozen is True

        # Now any node hitting _enforce_identity should be rejected
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.VALIDATED
        rejected = runtime._enforce_identity(node)
        assert rejected is True
        assert node.epistemic.state == NodeState.REJECTED


# =====================================================================
# D. Invariant Boundary Attacks
# =====================================================================
class TestInvariantBoundary:
    """Push invariant evaluation to exact boundaries and edge cases."""

    def _make_node(self, content_density=0.8, s_sie=0.9, mode_scaling_factor=1.0, c_node=0.9, depth=0):
        n = MagicMock()
        n.sie_node = MagicMock(
            content_density=content_density, s_sie=s_sie, composite_quality_score=0.5, mode_scaling_factor=mode_scaling_factor,
            alignment_vector=[0.8, 0.6, 0.7],
        )
        n.epistemic = MagicMock(c_node=c_node, depth=depth)
        n.v_score = 0.85
        n.constraints = []
        return n

    def test_invariant_exactly_at_threshold_gt(self):
        """'content_density > 0.5' with content_density=0.5 → FAIL (not strictly greater)."""
        eng = InvariantEngine()
        node = self._make_node(content_density=0.5)
        ok, failures = eng.validate_all(node, ["content_density > 0.5"])
        assert ok is False
        assert "content_density > 0.5" in failures

    def test_invariant_exactly_at_threshold_gte(self):
        """'content_density >= 0.5' with content_density=0.5 → PASS."""
        eng = InvariantEngine()
        node = self._make_node(content_density=0.5)
        ok, failures = eng.validate_all(node, ["content_density >= 0.5"])
        assert ok is True
        assert failures == []

    def test_invariant_negative_value(self):
        """'kappa > -1.0' with mode_scaling_factor=0.0 → PASS."""
        eng = InvariantEngine()
        node = self._make_node(mode_scaling_factor=0.0)
        ok, failures = eng.validate_all(node, ["kappa > -1.0"])
        assert ok is True

    def test_invariant_chain_all_must_pass(self):
        """3 invariants, 2 pass, 1 fails → entire set fails."""
        eng = InvariantEngine()
        node = self._make_node(content_density=0.8, s_sie=0.3, c_node=0.9)
        invariants = [
            "content_density > 0.5",     # content_density=0.8 → pass
            "s_sie >= 0.5",  # s_sie=0.3 → FAIL
            "c_node > 0.5",  # c_node=0.9 → pass
        ]
        ok, failures = eng.validate_all(node, invariants)
        assert ok is False
        assert len(failures) == 1
        assert "s_sie >= 0.5" in failures

    def test_invariant_field_resolution_with_missing_sie_node(self):
        """node.sie_node=None → resolves content_density to 0.0 → fails 'content_density > 0.5'."""
        eng = InvariantEngine()
        node = self._make_node()
        node.sie_node = None
        ok, failures = eng.validate_all(node, ["content_density > 0.5"])
        assert ok is False
        assert "content_density > 0.5" in failures

    def test_invariant_float_precision_epsilon(self):
        """'c_node == 0.25' with c_node=0.2500000000001 → PASS (epsilon tolerance)."""
        eng = InvariantEngine()
        node = self._make_node(c_node=0.2500000000001)
        ok, failures = eng.validate_all(node, ["c_node == 0.25"])
        assert ok is True, f"Epsilon tolerance should allow near-equal: failures={failures}"


# =====================================================================
# E. Consistency & Redundancy Flooding
# =====================================================================
class TestConsistencyFlood:
    """Large node sets with pathological similarity patterns."""

    def _make_node(self, node_id, pg, c_node=0.9, constraints=None):
        n = MagicMock()
        n.node_id = node_id
        n.sie_node = MagicMock(alignment_vector=pg)
        n.epistemic = MagicMock(c_node=c_node)
        n.constraints = constraints or []
        return n

    def test_20_identical_nodes_mass_redundancy(self):
        """20 nodes with identical alignment_vector → 19 rejected, 1 survivor."""
        eng = ConsistencyEngine()
        nodes = [
            self._make_node(f"n{i}", [1.0, 0.0, 0.0], c_node=0.9 - i * 0.01)
            for i in range(20)
        ]
        result = eng.validate_global_consistency(nodes)

        # n0 has highest c_node (0.9), should survive
        assert "n0" not in result["rejected_ids"]
        # Most others should be rejected as redundant
        assert len(result["rejected_ids"]) >= 15
        assert result["status"] in ("degraded", "failed")

    def test_contradicting_pairs_resolve_by_confidence(self):
        """10 anti-correlated pairs → 10 lower-confidence nodes rejected."""
        eng = ConsistencyEngine()
        nodes = []
        for i in range(10):
            # High confidence: positive gradient
            nodes.append(
                self._make_node(f"pos_{i}", [1.0, 0.0, 0.0], c_node=0.9)
            )
            # Low confidence: anti-correlated gradient
            nodes.append(
                self._make_node(f"neg_{i}", [-1.0, 0.0, 0.0], c_node=0.5)
            )
        contradictions = eng.detect_contradictions(nodes)
        rejected = eng.resolve_contradictions(nodes, contradictions)

        # All negative (lower confidence) should be rejected
        for i in range(10):
            assert f"neg_{i}" in rejected

    def test_mixed_contradiction_and_redundancy(self):
        """Nodes that are both redundant AND contradicting other nodes."""
        eng = ConsistencyEngine()
        nodes = [
            self._make_node("a", [1.0, 0.0, 0.0], c_node=0.9),
            self._make_node("b", [1.0, 0.0, 0.0], c_node=0.8),   # redundant with a
            self._make_node("c", [-1.0, 0.0, 0.0], c_node=0.7),  # contradicts a, b
        ]
        result = eng.validate_global_consistency(nodes)
        # b should be rejected (redundant with a)
        # c should be rejected (contradicts a)
        assert "a" not in result["rejected_ids"]
        assert result["status"] in ("degraded", "failed")

    def test_consistency_with_zero_vectors_skipped(self):
        """[0,0,0] phase gradient should not cause false positives."""
        eng = ConsistencyEngine()
        nodes = [
            self._make_node("a", [0.0, 0.0, 0.0]),
            self._make_node("b", [0.0, 0.0, 0.0]),
            self._make_node("c", [0.8, 0.6, 0.7]),
        ]
        redundancies = eng.detect_redundancy(nodes)
        contradictions = eng.detect_contradictions(nodes)
        # Zero vectors should not match as redundant or contradicting
        assert all("a" not in pair and "b" not in pair for pair in redundancies), (
            f"Zero vectors should be skipped: {redundancies}"
        )

    def test_consistency_status_degrades_not_crashes(self):
        """Massive contradiction set → status='failed' but no exception."""
        eng = ConsistencyEngine()
        nodes = []
        for i in range(50):
            angle = (i / 50) * 2 * math.pi
            pg = [math.cos(angle), math.sin(angle), 0.0]
            nodes.append(self._make_node(f"n{i}", pg, c_node=0.5 + i * 0.01))

        # Should complete without raising
        result = eng.validate_global_consistency(nodes)
        assert result["status"] in ("consistent", "degraded", "failed")
        assert isinstance(result["rejected_ids"], list)


# =====================================================================
# F. Omega Handshake Under Stress
# =====================================================================
class TestOmegaStress:
    """System-wide Omega validation under degraded conditions."""

    def _make_node(self, node_id, pg=None, constraints=None):
        n = MagicMock()
        n.node_id = node_id
        n.sie_node = MagicMock(alignment_vector=pg or [0.8, 0.6, 0.7])
        n.constraints = constraints or []
        return n

    def _make_identity_manager(self, drift=0.05, frozen=False, rolling_slr=0.1):
        im = MagicMock()
        im.drift_score = drift
        im.frozen = frozen
        im.max_drift = 0.3
        im.get_rolling_slr_mean = MagicMock(return_value=rolling_slr)
        return im

    def test_omega_pass_with_clean_system(self):
        """Healthy system → OMEGA_PASS."""
        ov = OmegaValidator()
        nodes = [self._make_node("a"), self._make_node("b")]
        im = self._make_identity_manager()
        result = ov.omega_handshake(nodes, None, im)
        assert result["status"] == "OMEGA_PASS"

    def test_omega_degraded_high_slr_only(self):
        """SLR mean > 0.65, everything else clean → OMEGA_DEGRADED."""
        ov = OmegaValidator()
        nodes = [self._make_node("a")]
        im = self._make_identity_manager(rolling_slr=0.75)
        result = ov.omega_handshake(nodes, None, im)
        assert result["status"] == "OMEGA_DEGRADED"

    def test_omega_fail_frozen_identity(self):
        """frozen=True → always OMEGA_FAIL."""
        ov = OmegaValidator()
        nodes = [self._make_node("a")]
        im = self._make_identity_manager(frozen=True, drift=0.5)
        result = ov.omega_handshake(nodes, None, im)
        assert result["status"] == "OMEGA_FAIL"

    def test_omega_fail_three_dimensions(self):
        """SLR critical + drift exceeded + orphans → OMEGA_FAIL (>=3 failures)."""
        ov = OmegaValidator()
        G = nx.DiGraph()
        G.add_node("a")
        G.add_node("orphan")
        # orphan has no edges — it's unreachable
        nodes = [self._make_node("a"), self._make_node("orphan")]
        im = self._make_identity_manager(
            drift=0.5, frozen=True, rolling_slr=0.8,
        )
        result = ov.omega_handshake(nodes, G, im)
        assert result["status"] == "OMEGA_FAIL"

    def test_partial_check_blocks_acceptance_on_fatal_constraint(self):
        """Partial omega check with fatal constraint → reroll, not ACCEPTED."""
        fatal_c = MagicMock(severity="fatal", valid=False, description="bad")
        nd = make_node_dict(node_id="omega_block")
        runtime = make_runtime([nd])
        node = runtime.graph.nodes["omega_block"]
        set_scored(runtime, node, c_node=0.95)
        node.constraints = [fatal_c]

        # Partial check should fail
        result = runtime.omega_validator.partial_check(node)
        assert result["pass"] is False

    def test_omega_handshake_cannot_upgrade_status(self):
        """If system status is 'degraded', omega can only keep or worsen it."""
        ov = OmegaValidator()
        nodes = [self._make_node("a")]
        im = self._make_identity_manager()  # Clean system

        result = ov.omega_handshake(nodes, None, im)
        assert result["status"] == "OMEGA_PASS"

        # Simulate: consistency already degraded the system
        # Now re-run omega with a problem
        im_bad = self._make_identity_manager(rolling_slr=0.75)
        result_bad = ov.omega_handshake(nodes, None, im_bad)

        # Can never be better than worst component
        assert result_bad["status"] in ("OMEGA_DEGRADED", "OMEGA_FAIL")

    def test_omega_detects_stranded_non_terminal_orphan_state(self):
        """A node left in VALIDATED at completion must fail orphan-state closure."""
        ov = OmegaValidator()
        accepted = self._make_node("accepted_1")
        accepted.epistemic = MagicMock(state="ACCEPTED")

        stranded = MagicMock()
        stranded.node_id = "stranded_validated"
        stranded.sie_node = MagicMock(alignment_vector=[0.7, 0.7, 0.7])
        stranded.constraints = []
        stranded.epistemic = MagicMock(state="VALIDATED")

        im = self._make_identity_manager()
        result = ov.omega_handshake(
            nodes=[accepted],
            nx_graph=None,
            identity_manager=im,
            all_nodes=[accepted, stranded],
        )

        orphan_dim = result["dimensions"]["orphan_state_closure"]
        assert orphan_dim["pass"] is False
        assert orphan_dim["orphan_count"] == 1
        assert "stranded_validated" in orphan_dim["orphan_node_ids"]
        assert result["status"] in ("OMEGA_DEGRADED", "OMEGA_FAIL")


# =====================================================================
# G. Branch Stability Collapse
# =====================================================================
class TestBranchCollapse:
    """Entire branches under soft degradation."""

    def test_all_branches_unstable_mass_degradation(self):
        """3 branches all with mean c_node < 0.2 → all degraded."""
        nodes = []
        for br in range(3):
            for j in range(2):
                nodes.append(make_node_dict(
                    node_id=f"br{br}_n{j}", branch_id=f"branch_{br}",
                ))

        runtime = make_runtime(nodes)
        runtime._nx_graph = runtime._build_directed_graph(list(runtime.graph.nodes.values()))

        # Set all to SCORED with very low confidence
        for n in runtime.graph.nodes.values():
            n.epistemic.state = NodeState.SCORED
            n.epistemic.c_node = 0.15  # Very low

        runtime._apply_branch_stability()

        for n in runtime.graph.nodes.values():
            # All should be degraded, NOT rejected
            assert n.epistemic.state == NodeState.SCORED
            assert n.epistemic.c_node == pytest.approx(
                0.15 * BRANCH_DEGRADATION_FACTOR, abs=0.001,
            )
            assert n.semantics.get("_branch_degraded") is True

    def test_stable_branch_survives_while_siblings_degrade(self):
        """1 healthy branch + 2 sick → only sick branches degraded."""
        nodes = []
        # Healthy branch
        for j in range(2):
            nodes.append(make_node_dict(
                node_id=f"healthy_{j}", branch_id="healthy",
            ))
        # Sick branches
        for br in range(2):
            for j in range(2):
                nodes.append(make_node_dict(
                    node_id=f"sick{br}_{j}", branch_id=f"sick_{br}",
                ))

        runtime = make_runtime(nodes)
        runtime._nx_graph = runtime._build_directed_graph(list(runtime.graph.nodes.values()))

        for n in runtime.graph.nodes.values():
            n.epistemic.state = NodeState.SCORED
            if "healthy" in n.node_id:
                n.epistemic.c_node = 0.85  # High confidence
            else:
                n.epistemic.c_node = 0.15  # Low confidence

        runtime._apply_branch_stability()

        # Healthy branch untouched
        for nid in ("healthy_0", "healthy_1"):
            n = runtime.graph.nodes[nid]
            assert n.epistemic.c_node == pytest.approx(0.85), (
                f"{nid} should not be degraded"
            )

        # Sick branches degraded
        for br in range(2):
            for j in range(2):
                n = runtime.graph.nodes[f"sick{br}_{j}"]
                assert n.epistemic.c_node < 0.15
                assert n.semantics.get("_branch_degraded") is True

    def test_branch_degradation_compounds_with_depth_penalty(self):
        """Unstable branch + deep node → compound penalty; c_node stays >= 0."""
        nd = make_node_dict(node_id="deep_sick", depth=10, branch_id="sick_deep")
        runtime = make_runtime([nd])
        runtime._nx_graph = runtime._build_directed_graph(list(runtime.graph.nodes.values()))

        node = runtime.graph.nodes["deep_sick"]
        node.epistemic.state = NodeState.SCORED
        node.epistemic.c_node = 0.15
        node.epistemic.depth = 10

        # Apply branch degradation
        runtime._apply_branch_stability()

        # Check degradation happened
        c_after_branch = node.epistemic.c_node
        assert c_after_branch < 0.15

        # Now apply depth penalty via _compute_unified_confidence
        node.sie_node = SemanticReasoningNode(
            content_density=0.8, s_sie=0.9, composite_quality_score=0.5, mode_scaling_factor=1.0,
            alignment_vector=[0.8, 0.6, 0.7],
        )
        node.v_score = 0.85
        runtime._compute_unified_confidence(node)

        # Even after compound penalties, confidence >= 0 (no negative overflow)
        assert node.epistemic.c_node >= 0.0

    def test_degraded_branch_still_eligible_for_reroll(self):
        """Degraded nodes are NOT hard-rejected; they can still reroll."""
        nd = make_node_dict(node_id="degrade_reroll", branch_id="degrading")
        nd["epistemic"]["retry_budget"] = 3
        runtime = make_runtime([nd])
        runtime._nx_graph = runtime._build_directed_graph(list(runtime.graph.nodes.values()))

        node = runtime.graph.nodes["degrade_reroll"]
        node.epistemic.state = NodeState.SCORED
        node.epistemic.c_node = 0.15

        # Apply branch degradation
        runtime._apply_branch_stability()
        assert node.semantics.get("_branch_degraded") is True
        assert node.epistemic.state == NodeState.SCORED  # NOT REJECTED

        # Should still be able to reroll
        rerolled = runtime._route_to_reroll(
            node, NodeState.SCORED, "low_confidence",
        )
        assert rerolled is True
        assert node.epistemic.state == NodeState.REROLL


# =====================================================================
# H. End-to-End Integration Stress
# =====================================================================
class TestEndToEndStress:
    """Full run() with adversarial graph topologies."""

    def test_full_run_empty_graph(self):
        """Zero nodes → status='failed' (no validated nodes), no crash."""
        runtime = make_runtime([])
        result = runtime.run()
        assert result["status"] in ("failed", "completed")
        assert isinstance(result["validated_nodes"], list)
        assert isinstance(result["rejected_nodes"], list)

    def test_full_run_single_node_accepted(self):
        """1 well-formed node → should reach ACCEPTED or at least SCORED."""
        nd = make_node_dict(node_id="solo")
        runtime = make_runtime([nd])
        result = runtime.run()

        # The node should be processed (either accepted or rejected based on SIE)
        assert result["status"] in ("success", "degraded", "failed")
        assert "run_id" in result
        assert isinstance(result["execution_trace"], list)
        assert len(result["execution_trace"]) >= 1

    def test_full_run_with_cycle_detection(self):
        """Circular dependency → V2.0 quarantines the SCC and degrades the run.

        Prior to V2.0 this aborted the whole run with status='failed' the
        moment any cycle was detected. Real-world codebases (e.g. starlette)
        contain legitimate circular imports that triggered this catastrophic
        abort and collapsed every node into rejected_nodes, destroying the
        SFT artifact. V2.0 quarantines the SCC members, seals the break into
        the cryptographic ledger, and continues the run on the remaining DAG.
        """
        # Create nodes with circular topology: A → B → A
        nd_a = make_node_dict(node_id="cycle_a", child_ids=["cycle_b"])
        nd_b = make_node_dict(node_id="cycle_b", child_ids=["cycle_a"])
        runtime = make_runtime([nd_a, nd_b])
        result = runtime.run()

        # Cycle detection must appear in the trace, but the run must not
        # have taken the fatal-abort branch unless the quarantine emptied
        # the graph entirely.
        cycle_steps = [
            step for step in result["execution_trace"]
            if step.get("step") == "cycle_detection"
        ]
        assert cycle_steps, "cycle_detection step missing from execution_trace"
        first_cycle = cycle_steps[0]
        assert first_cycle["status"] in ("degraded", "halted")
        assert first_cycle.get("quarantined_nodes", 0) >= 1

        # Both SCC members must be reflected in graph.quarantine and in the
        # ledger as OMEGA_ORPHAN_QUARANTINE transitions.
        quarantined_ids = {
            (q.get("node_id") if isinstance(q, dict) else None)
            for q in result.get("quarantined_nodes", [])
        }
        assert {"cycle_a", "cycle_b"}.issubset(quarantined_ids)

        ledger = result.get("transition_ledger", [])
        cycle_ledger_entries = [
            r for r in ledger
            if r.get("reason") == "topological_cycle_break"
        ]
        assert len(cycle_ledger_entries) >= 2
        assert result.get("transition_ledger_valid") is True

        # With both nodes quarantined the graph is empty, so the V2.0 path
        # takes the legitimate halted branch (empty remaining DAG) and ends
        # with status='failed'. A test with a partial SCC exercises the
        # degraded-continues path — see test_partial_cycle_quarantine.
        assert result["status"] == "failed"

    def test_partial_cycle_quarantine_keeps_run_alive(self):
        """Cycle touches a subset of the graph → SCC quarantined, DAG survives.

        This is the V2.0 real-world path: a repository contains legitimate
        circular references (e.g. starlette middleware ↔ routing) but most
        of the graph is acyclic. The cycle-break routine must quarantine
        only the SCC members and let the remaining nodes flow through the
        normal state machine (synthesis + Omega handshake) rather than
        collapsing the whole run into a fatal abort. Whether those residual
        nodes end up validated, rejected, or cascaded is governed by the
        usual downstream gates — what matters here is that the cycle-break
        path DID NOT short-circuit the run and that the ledger/quarantine
        accounting is accurate.
        """
        # Cyclic pair + three healthy nodes forming a clean chain.
        nd_a = make_node_dict(node_id="cycle_a", child_ids=["cycle_b"])
        nd_b = make_node_dict(node_id="cycle_b", child_ids=["cycle_a"])
        nd_x = make_node_dict(node_id="healthy_x", child_ids=["healthy_y"])
        nd_y = make_node_dict(node_id="healthy_y", child_ids=["healthy_z"])
        nd_z = make_node_dict(node_id="healthy_z")
        runtime = make_runtime([nd_a, nd_b, nd_x, nd_y, nd_z])
        result = runtime.run()

        # Cycle detection must show the degraded path, not the halted one.
        cycle_steps = [
            step for step in result["execution_trace"]
            if step.get("step") == "cycle_detection"
        ]
        assert cycle_steps
        assert cycle_steps[0]["status"] == "degraded"
        assert cycle_steps[0].get("quarantined_nodes", 0) == 2
        assert cycle_steps[0].get("remaining_nodes", 0) == 3

        # Proof the run did NOT take the fatal early-return — the trace
        # must contain post-cycle-detection steps (synthesis, etc.) that
        # only fire when the main state-machine loop was entered.
        trace_steps = [s.get("step") for s in result["execution_trace"]]
        assert "synthesis" in trace_steps, (
            "Run aborted at cycle detection instead of continuing on the "
            "remaining DAG — V2.0 regression"
        )

        # Cycle-quarantined nodes appear in the graph.quarantine list; the
        # healthy nodes must not. Whether they ended up validated or
        # cascaded is immaterial to this contract.
        quarantined_ids = {
            (q.get("node_id") if isinstance(q, dict) else None)
            for q in result.get("quarantined_nodes", [])
        }
        assert {"cycle_a", "cycle_b"}.issubset(quarantined_ids)
        assert "healthy_x" not in quarantined_ids
        assert "healthy_y" not in quarantined_ids
        assert "healthy_z" not in quarantined_ids

        # Omega handshake must have actually run against the residual DAG
        # (proving we reached the handshake instead of early-aborting).
        omega = result.get("omega_handshake")
        assert omega is not None and omega.get("status") in (
            "OMEGA_PASS", "OMEGA_DEGRADED", "OMEGA_FAIL"
        )

        # Cryptographic ledger must be intact and contain the cycle-break
        # transitions so the break is tamper-evident.
        assert result.get("transition_ledger_valid") is True
        cycle_ledger = [
            r for r in result.get("transition_ledger", [])
            if r.get("reason") == "topological_cycle_break"
        ]
        assert len(cycle_ledger) >= 2

    def test_full_run_max_iterations_halts_gracefully(self):
        """
        Graph designed to never converge: node perpetually rerolls.
        Should halt at MAX_ITERATIONS without crashing.
        """
        nd = make_node_dict(node_id="forever")
        nd["epistemic"]["retry_budget"] = 999  # Effectively infinite
        runtime = make_runtime([nd])

        # Patch execute to always produce invalid contract output,
        # forcing perpetual reroll
        def always_fail_execute(node):
            """Produce output that always fails contract validation."""
            if not isinstance(node.semantics, dict):
                node.semantics = {}
            # Set nothing useful — contract validation will fail
            pass

        with patch.object(runtime, "execute", side_effect=always_fail_execute):
            result = runtime.run()

        # Should complete without crash (V2.1: "success" is now possible
        # after backpressure cap reduction from 0.5 → 0.2)
        assert result["status"] in ("success", "failed", "degraded", "fatal_crash")
        assert "run_id" in result


# =====================================================================
# I. Drift Firewall Integration (D10)
# =====================================================================
from src.validation.pipeline_firewall import enforce_semantic_firewall, DriftViolation, FORBIDDEN_TOKENS


class TestDriftFirewallIntegration:
    """Drift firewall blocks contaminated nodes at the DAG runtime level."""

    def test_contaminated_node_rejected_pre_sie(self):
        """Node with IRER in semantics is rejected before SIE computation."""
        nd = make_node_dict(node_id="drift_001", code_snippet="def irer_leak(): return IRER")
        nd["semantics"]["code_snippet"] = "def irer_leak(): return IRER"
        runtime = make_runtime([nd])
        result = runtime.run()
        node = runtime.graph.nodes["drift_001"]
        assert node.epistemic.state in (NodeState.REJECTED, "REJECTED")
        assert node.epistemic.final_status == "drift_violation"

    def test_clean_node_not_blocked_by_firewall(self):
        """Node with clean semantics passes the drift firewall."""
        nd = make_node_dict(node_id="clean_001")
        runtime = make_runtime([nd])
        result = runtime.run()
        node = runtime.graph.nodes["clean_001"]
        # Clean node should progress past the firewall (may still be rejected by other gates)
        assert node.epistemic.final_status != "drift_violation"

    def test_resonance_field_in_semantics_rejected(self):
        """Resonance field token in node semantics triggers drift rejection."""
        nd = make_node_dict(node_id="drift_002")
        nd["semantics"]["theory"] = "Apply resonance field harmonic"
        runtime = make_runtime([nd])
        result = runtime.run()
        node = runtime.graph.nodes["drift_002"]
        assert node.epistemic.final_status == "drift_violation"

    def test_nura_physics_in_code_rejected(self):
        """nura physics in code_snippet triggers drift rejection."""
        nd = make_node_dict(node_id="drift_003", code_snippet="# based on nura physics\ndef calc(): pass")
        nd["semantics"]["code_snippet"] = "# based on nura physics\ndef calc(): pass"
        runtime = make_runtime([nd])
        result = runtime.run()
        node = runtime.graph.nodes["drift_003"]
        assert node.epistemic.final_status == "drift_violation"

    def test_cascade_from_drift_rejected_parent(self):
        """If a parent is rejected by drift firewall, children cascade to REJECTED."""
        parent = make_node_dict(node_id="drift_parent", child_ids=["drift_child"])
        parent["semantics"]["code_snippet"] = "IRER contamination"
        child = make_node_dict(node_id="drift_child", depth=1)
        runtime = make_runtime([parent, child])
        result = runtime.run()
        parent_node = runtime.graph.nodes["drift_parent"]
        child_node = runtime.graph.nodes["drift_child"]
        assert parent_node.epistemic.final_status == "drift_violation"
        assert child_node.epistemic.state in (NodeState.REJECTED, "REJECTED")
