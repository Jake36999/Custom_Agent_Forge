"""
Priority 3 — Advanced Governance & State Machine Verification
==============================================================

These regression tests lock in the two Priority-3 guarantees that close
the Aletheia Version 2.0 Architectural Roadmap:

    Deliverable 1 — Cryptographic State Transition Ledger
        Every CREATED -> VALIDATED -> SCORED -> TERMINAL mutation is
        sealed into a SHA256 hash chain. The chain is deterministic,
        tamper-evident, and carries (from_state, to_state, trigger,
        seq, prev_hash, this_hash) on every record.

    Deliverable 2 — Omega Orphan Closure (Triad of Failure)
        The OmegaValidator scans the execution graph after the main
        state machine finishes. Any node stranded in a non-terminal
        state is captured as an orphan, quarantined into
        graph.quarantine, mutated to REJECTED, and the run status is
        forcibly downgraded to OMEGA_DEGRADED / OMEGA_FAIL.

These tests are the final checkpoint before V2.0 freeze. If any of
them break, Priority 3 has regressed and the run is no longer
tamper-evident.
"""

import hashlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import networkx as nx
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipeline.dag_runtime import (
    DAGRuntime,
    InMemoryEpistemicGraph,
    NodeState,
    TransitionTrigger,
)
from src.pipeline.omega_validator import OmegaValidator


# ---------------------------------------------------------------------------
# Shared skill factory (reuses the minimal ACCEPTED-ready payload shape)
# ---------------------------------------------------------------------------

def _make_skill(node_id: str, depth: int = 0, branch_id: str = "root") -> dict:
    return {
        "node_id": node_id,
        "name": node_id,
        "file": "test.py",
        "code_snippet": (
            "def foo(x: int) -> int:\n"
            '    """Computes increment."""\n'
            "    return x + 1\n"
        ),
        "imports": [],
        "operator_type": "function",
        "source_type": "ast_code",
        "skill_type": "execution",
        "semantics": {
            "code_snippet": (
                "def foo(x: int) -> int:\n"
                '    """Computes increment."""\n'
                "    return x + 1\n"
            ),
            "name": "foo",
        },
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


def _make_runtime(nodes, mode: str = "coding_assistant") -> DAGRuntime:
    graph = InMemoryEpistemicGraph(nodes)
    acs = MagicMock(spec=[])
    return DAGRuntime(graph, acs, mode=mode)


# ===========================================================================
# Deliverable 1 — Cryptographic State Transition Ledger
# ===========================================================================

class TestCryptographicLedger:
    """Guarantees around _log_transition and the SHA256 hash chain."""

    def test_ledger_starts_from_genesis_with_zero_seq(self):
        rt = _make_runtime([])
        assert rt._transition_seq == 0
        assert rt._last_transition_hash == "GENESIS"
        assert rt.get_transition_ledger() == []

    def test_single_transition_records_full_schema(self):
        rt = _make_runtime([])
        rec = rt._log_transition(
            "node_x", NodeState.CREATED, NodeState.VALIDATED,
            trigger=TransitionTrigger.CONTRACT_PASSED,
        )
        # All six provenance fields must be present
        for field in ("seq", "node_id", "from_state", "to_state",
                      "trigger", "prev_hash", "this_hash"):
            assert field in rec, f"ledger record missing {field}"
        assert rec["seq"] == 1
        assert rec["prev_hash"] == "GENESIS"
        assert rec["from_state"] == NodeState.CREATED
        assert rec["to_state"] == NodeState.VALIDATED
        assert rec["trigger"] == TransitionTrigger.CONTRACT_PASSED
        # SHA256 hex digest is 64 characters
        assert len(rec["this_hash"]) == 64
        int(rec["this_hash"], 16)  # must be valid hex

    def test_sequence_increments_monotonically(self):
        rt = _make_runtime([])
        rt._log_transition("a", NodeState.CREATED, NodeState.VALIDATED,
                           trigger=TransitionTrigger.CONTRACT_PASSED)
        rt._log_transition("a", NodeState.VALIDATED, NodeState.SCORED,
                           trigger=TransitionTrigger.C_FINAL_SCORED)
        rt._log_transition("a", NodeState.SCORED, NodeState.ACCEPTED,
                           trigger=TransitionTrigger.C_FINAL_ACCEPTED)
        seqs = [r["seq"] for r in rt.get_transition_ledger()]
        assert seqs == [1, 2, 3]

    def test_hash_chain_links_prev_to_current(self):
        rt = _make_runtime([])
        rt._log_transition("a", NodeState.CREATED, NodeState.VALIDATED,
                           trigger=TransitionTrigger.CONTRACT_PASSED)
        rt._log_transition("a", NodeState.VALIDATED, NodeState.SCORED,
                           trigger=TransitionTrigger.C_FINAL_SCORED)
        ledger = rt.get_transition_ledger()
        assert ledger[0]["prev_hash"] == "GENESIS"
        assert ledger[1]["prev_hash"] == ledger[0]["this_hash"]

    def test_hash_chain_is_deterministic(self):
        """Identical inputs must produce identical hash chains across runs."""
        rt_a = _make_runtime([])
        rt_b = _make_runtime([])
        for rt in (rt_a, rt_b):
            rt._log_transition("n1", NodeState.CREATED, NodeState.VALIDATED,
                               trigger=TransitionTrigger.CONTRACT_PASSED)
            rt._log_transition("n1", NodeState.VALIDATED, NodeState.SCORED,
                               trigger=TransitionTrigger.C_FINAL_SCORED)
        assert rt_a._last_transition_hash == rt_b._last_transition_hash
        # Per-record hashes must also match
        for a, b in zip(rt_a.get_transition_ledger(), rt_b.get_transition_ledger()):
            assert a["this_hash"] == b["this_hash"]

    def test_verify_transition_ledger_passes_on_clean_chain(self):
        rt = _make_runtime([])
        rt._log_transition("a", NodeState.CREATED, NodeState.VALIDATED,
                           trigger=TransitionTrigger.CONTRACT_PASSED)
        rt._log_transition("a", NodeState.VALIDATED, NodeState.SCORED,
                           trigger=TransitionTrigger.C_FINAL_SCORED)
        rt._log_transition("a", NodeState.SCORED, NodeState.ACCEPTED,
                           trigger=TransitionTrigger.C_FINAL_ACCEPTED)
        assert rt.verify_transition_ledger() is True

    def test_verify_transition_ledger_detects_tamper(self):
        """Mutating a ledger record must fail the hash chain verification."""
        rt = _make_runtime([])
        rt._log_transition("a", NodeState.CREATED, NodeState.VALIDATED,
                           trigger=TransitionTrigger.CONTRACT_PASSED)
        rt._log_transition("a", NodeState.VALIDATED, NodeState.SCORED,
                           trigger=TransitionTrigger.C_FINAL_SCORED)
        # Forge a record: flip the trigger on the first entry
        rt._transition_ledger[0]["trigger"] = "FORGED_TRIGGER"
        assert rt.verify_transition_ledger() is False

    def test_verify_transition_ledger_detects_reorder(self):
        """Reordering ledger records must fail verification."""
        rt = _make_runtime([])
        rt._log_transition("a", NodeState.CREATED, NodeState.VALIDATED,
                           trigger=TransitionTrigger.CONTRACT_PASSED)
        rt._log_transition("b", NodeState.CREATED, NodeState.VALIDATED,
                           trigger=TransitionTrigger.CONTRACT_PASSED)
        rt._transition_ledger[0], rt._transition_ledger[1] = (
            rt._transition_ledger[1], rt._transition_ledger[0]
        )
        assert rt.verify_transition_ledger() is False

    def test_transition_trigger_enum_normalization(self):
        """Free-form reason strings must fall back to the enum taxonomy."""
        # Known token maps directly
        assert TransitionTrigger.normalize("contract_passed") == "contract_passed"
        # Prefix match
        assert TransitionTrigger.normalize("contract_passed: extra detail") == "contract_passed"
        # Unknown token preserved verbatim (diagnostic fidelity)
        unknown = TransitionTrigger.normalize("some_random_string")
        assert unknown == "some_random_string"

    def test_reason_backward_compat_without_explicit_trigger(self):
        """Call sites that pass a reason string (no trigger) must still work."""
        rt = _make_runtime([])
        rec = rt._log_transition(
            "n", NodeState.CREATED, NodeState.VALIDATED,
            reason="contract_passed: invariants ok",
        )
        # Trigger resolved from the reason prefix
        assert rec["trigger"] == TransitionTrigger.CONTRACT_PASSED
        assert rt.verify_transition_ledger() is True


# ===========================================================================
# Deliverable 1b — End-to-end ledger coverage through full run()
# ===========================================================================

class TestLedgerCoverageThroughRun:
    """A full DAG run must leave a verifiable end-to-end ledger."""

    def test_run_emits_transition_ledger_in_result(self):
        nd = _make_skill("e2e_ledger_001")
        rt = _make_runtime([nd])
        result = rt.run()
        assert "transition_ledger" in result
        assert "transition_ledger_valid" in result
        assert result["transition_ledger_valid"] is True
        assert isinstance(result["transition_ledger"], list)
        assert len(result["transition_ledger"]) >= 1

    def test_run_ledger_starts_with_bootstrap_transition(self):
        """Legacy-state nodes must emit a bootstrap ledger entry as their
        first transition. We explicitly construct a legacy-state node so
        the bootstrap branch fires."""
        nd = _make_skill("boot_001")
        nd["epistemic"]["state"] = "unresolved"  # LEGACY state
        rt = _make_runtime([nd])
        rt.run()
        ledger = rt.get_transition_ledger()
        assert len(ledger) >= 1
        # First ledger record must be the bootstrap normalization
        assert ledger[0]["node_id"] == "boot_001"
        assert ledger[0]["to_state"] == NodeState.CREATED
        assert ledger[0]["trigger"] == TransitionTrigger.BOOTSTRAP

    def test_run_ledger_terminates_every_active_node(self):
        """After run() finishes, no node should still be in an ACTIVE state."""
        skills = [_make_skill(f"final_{i}") for i in range(3)]
        rt = _make_runtime(skills)
        rt.run()
        for node in rt.graph.nodes.values():
            assert node.epistemic.state in NodeState.TERMINAL, (
                f"{node.node_id} ended in non-terminal state {node.epistemic.state}"
            )


# ===========================================================================
# Deliverable 2 — Omega Orphan Closure (Triad of Failure)
# ===========================================================================

class TestOmegaOrphanClosure:
    """Direct unit tests for OmegaValidator.validate_orphan_state_closure."""

    def test_empty_node_list_passes(self):
        ov = OmegaValidator()
        res = ov.validate_orphan_state_closure([])
        assert res["pass"] is True
        assert res["orphan_count"] == 0

    def test_all_terminal_nodes_passes(self):
        ov = OmegaValidator()

        class _Ep:
            def __init__(self, state): self.state = state

        class _N:
            def __init__(self, nid, state):
                self.node_id = nid
                self.epistemic = _Ep(state)

        nodes = [_N("a", "ACCEPTED"), _N("b", "REJECTED")]
        res = ov.validate_orphan_state_closure(nodes)
        assert res["pass"] is True
        assert res["orphan_count"] == 0
        assert res["orphan_node_ids"] == []

    def test_stranded_validated_node_is_orphan(self):
        ov = OmegaValidator()

        class _Ep:
            def __init__(self, state): self.state = state

        class _N:
            def __init__(self, nid, state):
                self.node_id = nid
                self.epistemic = _Ep(state)

        nodes = [
            _N("good_01", "ACCEPTED"),
            _N("stranded_01", "VALIDATED"),  # ← orphan: stuck in non-terminal
            _N("stranded_02", "SCORED"),     # ← orphan: stuck in non-terminal
            _N("good_02", "REJECTED"),
        ]
        res = ov.validate_orphan_state_closure(nodes)
        assert res["pass"] is False
        assert res["orphan_count"] == 2
        assert set(res["orphan_node_ids"]) == {"stranded_01", "stranded_02"}
        assert res["orphan_states"]["stranded_01"] == "VALIDATED"
        assert res["orphan_states"]["stranded_02"] == "SCORED"


class TestOmegaHandshakeDegradation:
    """omega_handshake() must route orphan detections into status downgrades."""

    def test_omega_handshake_flags_orphans_via_all_nodes(self):
        ov = OmegaValidator()

        class _Ep:
            def __init__(self, state): self.state = state

        class _N:
            def __init__(self, nid, state):
                self.node_id = nid
                self.epistemic = _Ep(state)
                self.constraints = None

        validated = [_N("good_01", "ACCEPTED")]
        all_nodes = validated + [_N("stranded_01", "VALIDATED")]
        result = ov.omega_handshake(
            nodes=validated,
            nx_graph=None,
            identity_manager=None,
            telemetry=None,
            all_nodes=all_nodes,
        )
        # Orphan dimension must be present and must have failed
        d6 = result["dimensions"]["orphan_state_closure"]
        assert d6["pass"] is False
        assert "stranded_01" in d6["orphan_node_ids"]
        # Status can only be downgraded (DEGRADED or FAIL), never PASS
        assert result["status"] in ("OMEGA_DEGRADED", "OMEGA_FAIL")


# ===========================================================================
# Deliverable 3 — Pathological Stranded-Node Regression (end-to-end)
# ===========================================================================

class TestOmegaStrandedNodeIntegration:
    """Full-loop test that proves the runtime catches a deliberately
    stranded non-terminal node and quarantines it via the cryptographic
    ledger before finalization."""

    def _make_runtime_with_stranded_orphan(self):
        """
        Construct a DAG where one node never transitions past VALIDATED.

        Approach: run the DAG normally with a single valid node, then
        before the main state machine exits, smuggle an additional node
        into the graph dict in a stranded VALIDATED state. The Omega
        handshake at the end of run() must catch it.
        """
        real_skill = _make_skill("legit_node_001")
        rt = _make_runtime([real_skill])

        # Patch the main loop by pre-seeding a stranded node into the
        # graph AFTER add_nodes but BEFORE run(). The node is in the
        # graph store but manually pinned to VALIDATED so the loop will
        # not touch it (because we also mark it ACTIVE=False via a
        # direct state manipulation post-loop).
        stranded_skill = _make_skill("stranded_orphan_001")
        rt.graph.add_nodes([stranded_skill])
        # Force the stranded node straight into VALIDATED, then short-
        # circuit the active check by removing its reroll budget. The
        # runtime's state-machine loop will still iterate over it once
        # (because VALIDATED is in ACTIVE), so we also patch out
        # _transition_to_scored for this specific node_id so it never
        # advances beyond VALIDATED.
        return rt

    def test_orphan_detected_and_quarantined_by_omega(self, monkeypatch):
        rt = self._make_runtime_with_stranded_orphan()

        # Intercept _transition_to_scored: return False for the orphan
        # so it is stranded, but let legitimate nodes pass through.
        original = rt._transition_to_scored

        def stranded_gate(node):
            if node.node_id == "stranded_orphan_001":
                # Refuse to advance beyond VALIDATED, leaving the node
                # in a non-terminal state so the Omega sweep must catch
                # it. We also set retry_budget=0 so the reroll path
                # cannot resurrect it.
                node.epistemic.retry_budget = 0
                # Lock state back to VALIDATED in case something else
                # mutated it.
                node.epistemic.state = NodeState.VALIDATED
                return False
            return original(node)

        monkeypatch.setattr(rt, "_transition_to_scored", stranded_gate)

        # Also stub out the reroll path so VALIDATED→anything doesn't
        # happen for the orphan.
        original_route = rt._route_to_reroll

        def no_reroll_for_orphan(node, prev, reason, violation_details=None):
            if node.node_id == "stranded_orphan_001":
                # Pin it in place — this simulates a hard-crash mid-
                # transition where the node is orphaned.
                node.epistemic.state = NodeState.VALIDATED
                return False
            return original_route(node, prev, reason, violation_details)

        monkeypatch.setattr(rt, "_route_to_reroll", no_reroll_for_orphan)

        result = rt.run()

        # --- Assertions ---
        omega = result.get("omega_handshake")
        assert omega is not None, "Omega handshake must run on completion"
        d6 = omega["dimensions"].get("orphan_state_closure", {})
        # The orphan must have been detected
        assert "stranded_orphan_001" in d6.get("orphan_node_ids", []), (
            "OmegaValidator must detect the stranded VALIDATED node as an orphan"
        )
        # Status must be downgraded to FAIL or DEGRADED
        assert omega["status"] in ("OMEGA_FAIL", "OMEGA_DEGRADED")
        assert result["status"] in ("failed", "degraded")

        # The orphan must now be in REJECTED (quarantine mutated it)
        orphan_node = rt.graph.nodes["stranded_orphan_001"]
        assert orphan_node.epistemic.state == NodeState.REJECTED, (
            f"orphan must be mutated to REJECTED, got {orphan_node.epistemic.state}"
        )
        assert orphan_node.epistemic.final_status == "omega_orphan_quarantined"

        # The orphan must appear in graph.quarantine with reason
        q_ids = [
            q.get("node_id") for q in rt.graph.quarantine
            if isinstance(q, dict)
            and q.get("quarantine_reason") == "omega_orphan_state"
        ]
        assert "stranded_orphan_001" in q_ids, (
            "orphan must be appended to graph.quarantine with quarantine_reason"
        )

        # Cryptographic ledger must contain the orphan-quarantine transition
        ledger = rt.get_transition_ledger()
        orphan_records = [
            r for r in ledger
            if r["node_id"] == "stranded_orphan_001"
            and r["trigger"] == TransitionTrigger.OMEGA_ORPHAN_QUARANTINE
        ]
        assert len(orphan_records) == 1, (
            "ledger must contain exactly one OMEGA_ORPHAN_QUARANTINE record for the orphan"
        )
        assert orphan_records[0]["to_state"] == NodeState.REJECTED
        # And the chain must still verify end-to-end
        assert rt.verify_transition_ledger() is True

    def test_clean_run_does_not_trigger_orphan_closure(self):
        """Baseline: a well-formed run must NOT emit orphan-quarantine records."""
        nd = _make_skill("clean_run_001")
        rt = _make_runtime([nd])
        rt.run()
        ledger = rt.get_transition_ledger()
        orphan_records = [
            r for r in ledger
            if r["trigger"] == TransitionTrigger.OMEGA_ORPHAN_QUARANTINE
        ]
        assert orphan_records == []
        assert rt.verify_transition_ledger() is True
