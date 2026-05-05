import sys
from pathlib import Path

# Ensure project root is importable for absolute src imports.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.models import SemanticReasoningNode
from src.pipeline.dag_runtime import DAGRuntime, InMemoryEpistemicGraph, NodeState


class DummyACS:
    pass


def _make_node_payload(node_id: str, branch_id: str, c_node: float, valid_flag: bool) -> dict:
    return {
        "node_id": node_id,
        "name": node_id,
        "file": "test.py",
        "code_snippet": "def f(x):\n    return x",
        "imports": [],
        "operator_type": "function",
        "source_type": "ast_code",
        "skill_type": "execution",
        "semantics": {"code_snippet": "def f(x):\n    return x", "name": node_id},
        "teaching_layer": {
            "skill_identity": {"name": node_id},
            "method_metadata": {"name": node_id, "language": "python"},
            "reasoning_vectors": {
                "intent": "compute",
                "strategy": "direct",
                "constraints": ["deterministic"],
                "execution_pattern": ["functional"],
                "failure_modes": ["type_error"],
            },
            "implementation_template": {"code": "pass"},
        },
        "epistemic": {
            "state": NodeState.ACCEPTED,
            "c_node": c_node,
            "retry_budget": 6,
            "depth": 0,
            "branch_id": branch_id,
        },
        "validation_pass": True,
        "constraints": {
            "initial_constraints": [
                {
                    "type": "semantic",
                    "tags": ["x_truth"],
                    "valid": valid_flag,
                    "severity": "error",
                    "description": "parallel branch claim",
                }
            ]
        },
    }


def test_runtime_consistency_quarantines_parallel_contradiction_and_routes_trace() -> None:
    nodes = [
        _make_node_payload("branch_a", "branch_a", 0.95, True),
        _make_node_payload("branch_b", "branch_b", 0.88, False),
    ]

    runtime = DAGRuntime(graph=InMemoryEpistemicGraph(nodes), acs=DummyACS(), mode="advocate")

    # Force explicit semantic contradiction across parallel branches.
    runtime.graph.nodes["branch_a"].sie_node = SemanticReasoningNode(
        content_density=0.8,
        s_sie=0.9,
        composite_quality_score=0.5,
        mode_scaling_factor=1.0,
        alignment_vector=[1.0, 0.0, 0.0],
    )
    runtime.graph.nodes["branch_b"].sie_node = SemanticReasoningNode(
        content_density=0.8,
        s_sie=0.9,
        composite_quality_score=0.5,
        mode_scaling_factor=1.0,
        alignment_vector=[-1.0, 0.0, 0.0],
    )

    result = runtime.run()

    consistency_result = result.get("consistency_result") or {}
    contradictions = consistency_result.get("contradictions") or []
    rejected_ids = set(consistency_result.get("rejected_ids") or [])

    assert contradictions, "Expected explicit contradiction in parallel branches"
    assert "branch_b" in rejected_ids

    rejected_by_id = {
        n["node_id"]: n for n in (result.get("rejected_nodes") or []) if isinstance(n, dict)
    }
    assert "branch_b" in rejected_by_id

    rejected_semantics = rejected_by_id["branch_b"].get("semantics", {})
    assert rejected_by_id["branch_b"].get("epistemic", {}).get("final_status") == "consistency_violation"
    assert rejected_semantics.get("execution_eligible") is False

    traces = result.get("rejected_traces") or []
    consistency_traces = [
        t for t in traces
        if isinstance(t, dict)
        and t.get("node_id") == "branch_b"
        and t.get("reason") == "consistency_violation"
    ]
    assert consistency_traces, "Consistency violation must be routed to rejected_traces"
    assert isinstance(consistency_traces[0].get("consistency", {}).get("contradictions", []), list)
