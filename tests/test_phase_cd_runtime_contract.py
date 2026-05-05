import sys
from pathlib import Path
import pytest

# Ensure project root is importable for absolute src imports.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipeline.dag_runtime import InMemoryEpistemicGraph, DAGRuntime


class DummyACS:
    """Minimal ACS stub for runtime construction in containment tests."""



def test_semantic_node_builder_derives_domain_agnostic_invariants() -> None:
    pytest.importorskip("spacy")
    from src.pipeline.semantic_compiler import SemanticNodeBuilder

    frames = [
        {
            "predicate": {"lemma": "apply"},
            "arguments": [{"role": "agent"}, {"role": "theme"}],
            "confidence": {"predicate": 0.9, "arguments": 0.8},
        },
        {
            "predicate": {"lemma": "evaluate"},
            "arguments": [{"role": "agent"}],
            "confidence": {"predicate": 0.8, "arguments": 0.7},
        },
    ]
    discourse = {"label": "method", "confidence": 0.82}
    graph = {
        "nodes": [{"id": 1}, {"id": 2}, {"id": 3}],
        "edges": [{"source": 1, "target": 2}, {"source": 2, "target": 3}],
    }

    node = SemanticNodeBuilder(
        text="Apply and evaluate constrained transformations.",
        frames=frames,
        discourse=discourse,
        graph=graph,
        section="body",
        source_file="sample.txt",
        index=0,
        pipeline_type="procedural",
        pipeline_role="transform",
        intent="execution",
        lens="ExecutorLens",
        verifiability="high",
        mode="advocate",
        config={"frame_min_arg_confidence": 0.5},
        context_window=[],
        previous_node=None,
    ).build()

    assert "invariant_signals" in node
    signals = node["invariant_signals"]
    assert "channels" in signals
    assert "derived" in signals
    derived = signals["derived"]
    assert 0.0 <= derived["content_density"] <= 1.0
    assert 0.0 <= derived["content_density"] <= 1.0  # replaces resonance_density
    assert len(derived["alignment_vector"]) == 3
    assert derived["composite_quality_score"] >= 0.0
    assert isinstance(node["execution_eligible"], bool)



def test_dag_runtime_contains_projected_schema_mismatch_without_crash() -> None:
    malformed_payload = {
        "node_id": "bad-node-1",
        "name": "bad-node",
        "file": "x.py",
        "code_snippet": "print('x')",
        "imports": [],
        # Missing mandatory ProjectedNode fields like operator_type/teaching_layer/epistemic
    }

    graph = InMemoryEpistemicGraph([malformed_payload])
    runtime = DAGRuntime(graph=graph, acs=DummyACS(), mode="advocate")
    result = runtime.run()

    assert result["status"] in ("failed", "degraded", "success")
    assert result["execution_trace"]
    assert result["rejected_nodes"]

    rejected = result["rejected_nodes"][0]
    assert rejected.get("semantics", {}).get("execution_eligible") is False
    assert rejected.get("epistemic", {}).get("final_status") == "projected_schema_mismatch"

    reasons = [t.get("reason") for t in result.get("rejected_traces", [])]
    assert "projected_node_validation_error" in reasons
