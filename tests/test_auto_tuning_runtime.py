import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

# Ensure project root is importable for absolute src imports.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import src.pipeline.dag_runtime as dag_runtime_module
from src.pipeline.dag_runtime import DAGRuntime, InMemoryEpistemicGraph, NodeState
from src.pipeline.identity_manager import IdentityManager


class DummyACS:
    pass


def _make_runtime() -> DAGRuntime:
    malformed_payload = {
        "node_id": "auto-tune-node",
        "name": "bad-node",
        "file": "x.py",
        "code_snippet": "print('x')",
        "imports": [],
    }
    graph = InMemoryEpistemicGraph([malformed_payload])
    return DAGRuntime(graph=graph, acs=DummyACS(), mode="advocate")


def test_auto_tuning_defaults_preserve_baseline_behavior() -> None:
    runtime = _make_runtime()

    assert runtime.edge_coherence_tuning_threshold == dag_runtime_module.EDGE_COHERENCE_THRESHOLD
    assert runtime.telemetry_check_cadence == 1
    assert runtime.recovery_tick_cadence == 1
    assert runtime._should_run_telemetry_check(1) is True
    assert runtime._should_run_recovery_tick(1) is True


def test_identity_manager_recovery_lambda_override_bounds_and_effect() -> None:
    mgr = IdentityManager(max_drift=1.0)

    with pytest.raises(ValueError):
        mgr.set_recovery_lambda(0.2)

    mgr.drift_score = 1.0
    default_after = mgr.apply_recovery_tick()

    mgr.drift_score = 1.0
    mgr.set_recovery_lambda(0.1)
    override_after = mgr.apply_recovery_tick()

    assert mgr.get_recovery_lambda() == 0.1
    assert override_after < default_after


def test_validate_parent_chain_uses_runtime_coherence_threshold(monkeypatch) -> None:
    runtime = _make_runtime()

    parent = SimpleNamespace(
        node_id="parent",
        epistemic=SimpleNamespace(state=NodeState.ACCEPTED, c_node=0.9),
        semantics={},
        sie_node=SimpleNamespace(alignment_vector=[1.0, 0.0, 0.0]),
    )
    child = SimpleNamespace(
        node_id="child",
        epistemic=SimpleNamespace(state=NodeState.CREATED, c_node=0.0),
        semantics={},
        sie_node=SimpleNamespace(alignment_vector=[1.0, 0.0, 0.0]),
    )

    graph = nx.DiGraph()
    graph.add_edge("parent", "child")
    runtime._nx_graph = graph
    runtime._get_nodes_safely = lambda: [parent, child]

    monkeypatch.setattr(runtime, "_compute_sie_edge_coherence", lambda p, c: 0.25)
    monkeypatch.setattr(dag_runtime_module, "calculate_sie_slr", lambda p, c: 0.0)

    runtime.edge_coherence_tuning_threshold = 0.3
    assert runtime._validate_parent_chain(child) is False

    runtime.edge_coherence_tuning_threshold = 0.2
    assert runtime._validate_parent_chain(child) is True


def test_recovery_cadence_gating() -> None:
    runtime = _make_runtime()
    runtime.recovery_tick_cadence = 3

    assert runtime._should_run_recovery_tick(1) is False
    assert runtime._should_run_recovery_tick(2) is False
    assert runtime._should_run_recovery_tick(3) is True


def test_telemetry_check_cadence_gating() -> None:
    runtime = _make_runtime()
    runtime.telemetry_check_cadence = 2

    assert runtime._should_run_telemetry_check(1) is False
    assert runtime._should_run_telemetry_check(2) is True


def test_auto_tuning_tightens_threshold_and_reduces_cadence_on_high_slr() -> None:
    runtime = _make_runtime()
    runtime.auto_tuning_cooldown_checks = 0
    runtime.recovery_tick_cadence = 3
    runtime.identity_manager.slr_history = [0.8, 0.8, 0.75]
    runtime.identity_manager.drift_score = 0.25
    runtime._policy_check_count = 1

    runtime._apply_runtime_auto_tuning(non_rejected_ratio=0.4)

    assert runtime.edge_coherence_tuning_threshold > dag_runtime_module.EDGE_COHERENCE_THRESHOLD
    assert runtime.recovery_tick_cadence == 2
    assert runtime._tuning_history["coherence_threshold"]
    assert runtime._tuning_history["recovery_tick_cadence"]


def test_auto_tuning_loosens_threshold_and_relaxes_cadence_on_stable_signal() -> None:
    runtime = _make_runtime()
    runtime.auto_tuning_cooldown_checks = 0
    runtime.edge_coherence_tuning_threshold = 0.34
    runtime.recovery_tick_cadence = 2
    runtime.identity_manager.slr_history = [0.2, 0.25, 0.18]
    runtime.identity_manager.drift_score = 0.02
    runtime._policy_check_count = 1

    runtime._apply_runtime_auto_tuning(non_rejected_ratio=0.95)

    assert runtime.edge_coherence_tuning_threshold == 0.32
    assert runtime.recovery_tick_cadence == 3


def test_auto_tuning_determinism_for_same_state() -> None:
    runtime_a = _make_runtime()
    runtime_b = _make_runtime()

    for runtime in (runtime_a, runtime_b):
        runtime.auto_tuning_cooldown_checks = 0
        runtime.edge_coherence_tuning_threshold = 0.34
        runtime.recovery_tick_cadence = 2
        runtime.identity_manager.slr_history = [0.2, 0.25, 0.18]
        runtime.identity_manager.drift_score = 0.02
        runtime._policy_check_count = 1

    runtime_a._apply_runtime_auto_tuning(non_rejected_ratio=0.95)
    runtime_b._apply_runtime_auto_tuning(non_rejected_ratio=0.95)

    assert runtime_a.edge_coherence_tuning_threshold == runtime_b.edge_coherence_tuning_threshold
    assert runtime_a.recovery_tick_cadence == runtime_b.recovery_tick_cadence
    assert runtime_a._tuning_history == runtime_b._tuning_history
