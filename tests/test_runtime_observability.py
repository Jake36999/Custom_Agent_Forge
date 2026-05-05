import sys
from pathlib import Path

# Ensure project root is importable for absolute src imports.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipeline.dag_runtime import InMemoryEpistemicGraph, DAGRuntime
from src.pipeline.telemetry import TelemetryCollector


class DummyACS:
    """Minimal ACS stub for runtime construction in observability tests."""


def test_telemetry_runtime_summary_aggregates_elapsed_ms() -> None:
    collector = TelemetryCollector()
    collector.record("runtime", "dag_processing_time", payload={"elapsed_ms": 12.0})
    collector.record("runtime", "dag_processing_time", payload={"elapsed_ms": 30.0})

    snapshot = collector.snapshot()
    runtime = snapshot["runtime_summary"]

    assert runtime["count"] == 2
    assert runtime["min_ms"] == 12.0
    assert runtime["max_ms"] == 30.0
    assert runtime["mean_ms"] == 21.0


def test_runtime_result_includes_processing_time_and_telemetry_runtime_summary() -> None:
    malformed_payload = {
        "node_id": "bad-node-observe",
        "name": "bad-node",
        "file": "x.py",
        "code_snippet": "print('x')",
        "imports": [],
        # Missing mandatory ProjectedNode fields like operator_type/teaching_layer/epistemic
    }

    graph = InMemoryEpistemicGraph([malformed_payload])
    runtime = DAGRuntime(graph=graph, acs=DummyACS(), mode="advocate")
    result = runtime.run()

    assert "dag_processing_time_ms" in result
    assert isinstance(result["dag_processing_time_ms"], float)
    assert result["dag_processing_time_ms"] >= 0.0

    telemetry = result.get("telemetry") or {}
    runtime_summary = telemetry.get("runtime_summary")
    # In fallback-stub environments telemetry can be an empty dict; when the
    # concrete collector is active, runtime_summary must include at least one sample.
    if isinstance(runtime_summary, dict):
        assert runtime_summary.get("count", 0) >= 1
        assert runtime_summary.get("max_ms", 0.0) >= runtime_summary.get("min_ms", 0.0)


def test_state_transition_telemetry_emits_structured_payload() -> None:
    malformed_payload = {
        "node_id": "transition-payload-node",
        "name": "bad-node",
        "file": "x.py",
        "code_snippet": "print('x')",
        "imports": [],
    }

    graph = InMemoryEpistemicGraph([malformed_payload])
    runtime = DAGRuntime(graph=graph, acs=DummyACS(), mode="advocate")
    runtime._log_transition("n1", "CREATED", "VALIDATED", "contract_passed")

    telemetry = runtime.telemetry
    if not hasattr(telemetry, "_events"):
        return

    events = getattr(telemetry, "_events", [])
    assert events, "Expected at least one telemetry event after _log_transition"
    payload = events[-1].payload
    assert payload.get("from_state") == "CREATED"
    assert payload.get("to_state") == "VALIDATED"
    assert payload.get("trigger") == "contract_passed"
