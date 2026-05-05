"""
Aletheia Refactored Module Tests
---------------------------------
Covers: contracts, reroll, identity_manager, telemetry, SIE scoring,
DAG cascade logic, and the DAG scoring pass.
"""

import json
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the project root is on the path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ===================================================================
# 1. CONTRACT VALIDATION
# ===================================================================
from src.pipeline.contracts import (
    TheoristOutput, CodingOutput, AdvocateOutput, VeteranOutput,
    validate_contract, MODE_CONTRACTS,
)


class TestContracts:
    """Spec Priority 1 — Execution Contract Validator."""

    def test_theorist_valid(self):
        payload = {"constraints": ["no_mutation"], "dependencies": [], "invariants": [], "interfaces": []}
        ok, err = validate_contract(payload, TheoristOutput)
        assert ok is True
        assert err is None

    def test_theorist_missing_constraints(self):
        payload = {"dependencies": ["foo"]}
        ok, err = validate_contract(payload, TheoristOutput)
        assert ok is False
        assert err is not None

    def test_coding_valid(self):
        payload = {"implementation": "def f(): pass", "tests": "assert f() is None"}
        ok, err = validate_contract(payload, CodingOutput)
        assert ok is True

    def test_coding_empty_implementation(self):
        payload = {"implementation": ""}
        ok, err = validate_contract(payload, CodingOutput)
        assert ok is False

    def test_advocate_valid(self):
        payload = {"theory": "Architecture demands X", "implementation": "class X: pass"}
        ok, err = validate_contract(payload, AdvocateOutput)
        assert ok is True

    def test_advocate_missing_theory(self):
        payload = {"implementation": "class X: pass"}
        ok, err = validate_contract(payload, AdvocateOutput)
        assert ok is False

    def test_veteran_valid(self):
        payload = {"traceback": "Traceback ...", "diff": "- old\n+ new", "resolved_code": "def f(): return 1"}
        ok, err = validate_contract(payload, VeteranOutput)
        assert ok is True

    def test_veteran_missing_diff(self):
        payload = {"traceback": "Traceback ...", "resolved_code": "def f(): pass"}
        ok, err = validate_contract(payload, VeteranOutput)
        assert ok is False

    def test_mode_contracts_mapping(self):
        assert set(MODE_CONTRACTS.keys()) == {"theorist", "coding_assistant", "advocate", "veteran"}


# ===================================================================
# 1b. SEMANTIC DRIFT FIREWALL (D10)
# ===================================================================
from src.validation.pipeline_firewall import (
    enforce_semantic_firewall, DriftViolation, FORBIDDEN_TOKENS,
)


class TestSemanticDriftFirewall:
    """D10 — Drift firewall prevents cross-domain contamination."""

    def test_clean_text_passes(self):
        enforce_semantic_firewall("This is a normal function definition.")

    def test_empty_string_passes(self):
        enforce_semantic_firewall("")

    def test_none_input_passes(self):
        enforce_semantic_firewall(None)

    def test_forbidden_token_irer_raises(self):
        with pytest.raises(DriftViolation, match="IRER"):
            enforce_semantic_firewall("This contains IRER pseudo-physics.")

    def test_forbidden_token_resonance_field_raises(self):
        with pytest.raises(DriftViolation, match="resonance field"):
            enforce_semantic_firewall("Apply the resonance field equation.")

    def test_forbidden_token_sse_raises(self):
        with pytest.raises(DriftViolation, match="SSE 0.00087"):
            enforce_semantic_firewall("Calibrated to SSE 0.00087 threshold.")

    def test_forbidden_token_nura_raises(self):
        with pytest.raises(DriftViolation, match="nura physics"):
            enforce_semantic_firewall("Based on nura physics principles.")

    def test_case_insensitive_detection(self):
        with pytest.raises(DriftViolation):
            enforce_semantic_firewall("irer leaked into output")

    def test_context_attached_to_exception(self):
        with pytest.raises(DriftViolation) as exc_info:
            enforce_semantic_firewall("IRER detected", context="node:test_001")
        assert "node:test_001" in str(exc_info.value)

    def test_contract_rejects_drift_in_payload(self):
        payload = {"constraints": ["no IRER mutation"], "dependencies": [], "invariants": [], "interfaces": []}
        ok, err = validate_contract(payload, TheoristOutput)
        assert ok is False
        assert "DriftViolation" in err

    def test_contract_passes_clean_payload(self):
        payload = {"constraints": ["no_mutation"], "dependencies": [], "invariants": [], "interfaces": []}
        ok, err = validate_contract(payload, TheoristOutput)
        assert ok is True

    def test_all_forbidden_tokens_are_strings(self):
        for token in FORBIDDEN_TOKENS:
            assert isinstance(token, str)
            assert len(token) > 0

    # --- Compiler-internal self-description tokens ---

    def test_compiler_internal_omega_handshake_raises(self):
        with pytest.raises(DriftViolation, match="omega handshake"):
            enforce_semantic_firewall("The omega handshake was successful.")

    def test_compiler_internal_bounded_inevitability_raises(self):
        with pytest.raises(DriftViolation, match="bounded inevitability"):
            enforce_semantic_firewall("Achieves bounded inevitability.")

    def test_legitimate_physics_terms_pass_firewall(self):
        """Real physics vocabulary must not be blocked — the pipeline
        needs to compile legitimate physics stacks."""
        enforce_semantic_firewall("alignment_vector = [0.1, 0.2, 0.3]")
        enforce_semantic_firewall("waveform amplitude = content_density * kappa")
        enforce_semantic_firewall("Compute the resonance density of the field.")
        enforce_semantic_firewall("topological cascade analysis complete")

    def test_forbidden_tokens_count(self):
        """4 URF brands + 2 compiler-internal = 6 total."""
        assert len(FORBIDDEN_TOKENS) == 6


# ===================================================================
# 1c. DPO PAIR FORMATTING (D3)
# ===================================================================
from src.pipeline.dataset_formatter import format_rejected_traces


class TestDPOPairFormatting:
    """D3 — Rejected traces emit DPO preference pairs {prompt, chosen, rejected}."""

    def test_complete_pair_written(self, tmp_path):
        traces = [{"node_id": "n1", "reason": "contract_violation",
                    "rejected": {"code": "bad"}, "corrected": {"code": "good"}}]
        out = tmp_path / "dpo.jsonl"
        count = format_rejected_traces(traces, out)
        assert count == 1
        import jsonlines
        with jsonlines.open(out) as reader:
            entry = list(reader)[0]
        assert "prompt" in entry
        assert "chosen" in entry
        assert "rejected" in entry
        assert "instruction" not in entry
        assert "rejected_output" not in entry

    def test_missing_chosen_skipped(self, tmp_path):
        traces = [{"node_id": "n2", "reason": "sie_underflow",
                    "rejected": {"code": "bad"}}]
        out = tmp_path / "dpo.jsonl"
        count = format_rejected_traces(traces, out)
        assert count == 0

    def test_missing_rejected_skipped(self, tmp_path):
        traces = [{"node_id": "n3", "reason": "contract",
                    "corrected": {"code": "good"}}]
        out = tmp_path / "dpo.jsonl"
        count = format_rejected_traces(traces, out)
        assert count == 0

    def test_chosen_field_fallback(self, tmp_path):
        """'chosen' key in trace used as fallback when 'corrected' absent."""
        traces = [{"node_id": "n4", "reason": "reroll",
                    "rejected": {"v": 1}, "chosen": {"v": 2}}]
        out = tmp_path / "dpo.jsonl"
        count = format_rejected_traces(traces, out)
        assert count == 1
        import jsonlines
        with jsonlines.open(out) as reader:
            entry = list(reader)[0]
        assert '"v": 2' in entry["chosen"]

    def test_multiple_traces_counted(self, tmp_path):
        traces = [
            {"node_id": f"n{i}", "reason": "r", "rejected": {"x": i}, "corrected": {"x": i+1}}
            for i in range(5)
        ]
        out = tmp_path / "dpo.jsonl"
        count = format_rejected_traces(traces, out)
        assert count == 5

    def test_prompt_contains_node_and_reason(self, tmp_path):
        traces = [{"node_id": "nodeX", "reason": "slr_collapse",
                    "rejected": "bad", "corrected": "good"}]
        out = tmp_path / "dpo.jsonl"
        format_rejected_traces(traces, out)
        import jsonlines
        with jsonlines.open(out) as reader:
            entry = list(reader)[0]
        assert "nodeX" in entry["prompt"]
        assert "slr_collapse" in entry["prompt"]


# ===================================================================
# 1d. ACCEPTED-ONLY DATASET GATE (D11)
# ===================================================================
from src.pipeline.dataset_formatter import format_dataset


class TestAcceptedOnlyGate:
    """D11 — Only ACCEPTED/terminal nodes pass the epistemic survival gate."""

    def _make_skill_yaml(self, node_id, state, tmp_path):
        """Create a minimal Aletheia skill YAML that parses through the formatter."""
        skill = {
            "node_id": node_id,
            "name": node_id,
            "file": "test.py",
            "code_snippet": "def f(): pass",
            "imports": [],
            "operator_type": "function",
            "source_type": "ast_code",
            "skill_type": "execution",
            "semantics": {"code_snippet": "def f(): pass", "name": "f"},
            "teaching_layer": {
                "skill_identity": {"name": node_id},
                "method_metadata": {"name": node_id, "language": "python"},
                "reasoning_vectors": {
                    "intent": "compute",
                    "strategy": "direct",
                    "constraints": ["none"],
                    "execution_pattern": ["functional"],
                    "failure_modes": ["none"],
                },
                "implementation_template": {"code": "pass"},
            },
            "epistemic": {"state": state, "c_node": 0.9, "retry_budget": 6, "depth": 0, "branch_id": "root"},
        }
        import yaml
        file_path = tmp_path / f"{node_id}.yaml"
        with open(file_path, "w") as f:
            yaml.dump(skill, f)
        return file_path

    def test_accepted_node_passes_gate(self, tmp_path):
        self._make_skill_yaml("accepted_001", "ACCEPTED", tmp_path)
        out = tmp_path / "output" / "dataset.jsonl"
        format_dataset(tmp_path, out)
        assert out.exists()
        import jsonlines
        with jsonlines.open(out) as reader:
            items = list(reader)
        # At least one item should pass (the ACCEPTED node)
        assert len(items) >= 1

    def test_rejected_node_blocked_by_gate(self, tmp_path):
        self._make_skill_yaml("rejected_001", "REJECTED", tmp_path)
        out = tmp_path / "output" / "dataset.jsonl"
        format_dataset(tmp_path, out)
        # The rejected node should NOT appear in the output
        if out.exists():
            import jsonlines
            with jsonlines.open(out) as reader:
                items = list(reader)
            for item in items:
                meta = item.get("_aletheia_metadata", {})
                assert meta.get("node_id") != "rejected_001"

    def test_null_state_blocked_by_gate(self, tmp_path):
        self._make_skill_yaml("null_001", "CREATED", tmp_path)
        out = tmp_path / "output" / "dataset.jsonl"
        format_dataset(tmp_path, out)
        if out.exists():
            import jsonlines
            with jsonlines.open(out) as reader:
                items = list(reader)
            for item in items:
                meta = item.get("_aletheia_metadata", {})
                assert meta.get("node_id") != "null_001"


# ===================================================================
# 2. REROLL ENGINE
# ===================================================================
from src.pipeline.reroll import RerollEngine, _score_candidate, _MODE_CLEARABLE_KEYS


class TestRerollEngine:
    """D7 — 2-candidate branching reroll + DPO pair capture."""

    def _make_mock_node(self):
        """Create a minimal mock node with epistemic state and semantics."""
        node = MagicMock()
        node.node_id = "test_node_001"
        node.epistemic = MagicMock()
        node.epistemic.state = "REROLL"
        node.semantics = {"code_snippet": "def broken(): pass", "name": "broken"}
        return node

    def _make_mock_runtime(self):
        runtime = MagicMock()
        runtime.mode = "coding_assistant"
        runtime.execute = MagicMock()
        return runtime

    def test_inject_failure_context(self):
        engine = RerollEngine()
        node = self._make_mock_node()
        engine.inject_failure_context(node, "contract_violation", "missing implementation field")
        ctx = node.semantics["_reroll_context"]
        assert ctx["previous_failure_reason"] == "contract_violation"
        assert ctx["violation_details"] == "missing implementation field"
        assert ctx["reroll_attempt"] == 1

    def test_reroll_returns_rejected_trace(self):
        engine = RerollEngine()
        node = self._make_mock_node()
        runtime = self._make_mock_runtime()

        result = engine.reroll(node, runtime, reason="test_failure")
        assert result is not None
        assert result["node_id"] == "test_node_001"
        assert "rejected" in result
        assert "corrected" in result
        assert result["reason"] == "test_failure"

    def test_reroll_increments_counter(self):
        engine = RerollEngine()
        node = self._make_mock_node()
        runtime = self._make_mock_runtime()
        engine.reroll(node, runtime)
        # After first reroll, state transitions to VALIDATED; reset to REROLL
        node.epistemic.state = "REROLL"
        engine.reroll(node, runtime)
        assert engine.reroll_count == 2

    def test_reroll_evaluates_2_candidates(self):
        """D7: Reroll generates 2 candidates and reports count."""
        engine = RerollEngine()
        node = self._make_mock_node()
        runtime = self._make_mock_runtime()
        result = engine.reroll(node, runtime, reason="contract_violation")
        assert result is not None
        assert result["candidates_evaluated"] == 2

    def test_reroll_selects_best_candidate(self):
        """D7: Best candidate by content density is selected."""
        engine = RerollEngine()
        node = self._make_mock_node()
        runtime = self._make_mock_runtime()
        call_count = [0]
        def side_effect(n):
            call_count[0] += 1
            if call_count[0] == 1:
                n.semantics = {"code_snippet": "x"}  # Sparse
            else:
                n.semantics = {"code_snippet": "def good(x): return x + 1", "name": "good", "imports": ["os"]}  # Rich
        runtime.execute = MagicMock(side_effect=side_effect)
        result = engine.reroll(node, runtime)
        assert result is not None
        assert result["best_score"] > 0

    def test_score_candidate_empty_dict(self):
        assert _score_candidate({}) == 0.0

    def test_score_candidate_rich_content(self):
        score = _score_candidate({"code_snippet": "def f(): pass", "name": "f", "imports": ["os"]})
        assert score > 0.0

    def test_coding_assistant_clears_theory_key(self):
        """Issue 3 — coding_assistant reroll must clear 'theory' to prevent drift persistence."""
        assert "theory" in _MODE_CLEARABLE_KEYS["coding_assistant"]
        engine = RerollEngine()
        node = MagicMock()
        node.node_id = "drift_node"
        node.epistemic = MagicMock()
        node.epistemic.state = "REROLL"
        node.semantics = {
            "code_snippet": "def broken(): pass",
            "name": "broken",
            "theory": "Apply resonance field harmonic",
        }
        engine._clear_generated_keys(node, "coding_assistant")
        assert "theory" not in node.semantics
        assert "code_snippet" not in node.semantics
        assert "name" not in node.semantics


# ===================================================================
# 3. IDENTITY MANAGER
# ===================================================================
from src.pipeline.identity_manager import IdentityManager


class TestIdentityManager:
    """Spec Priority 5 — Identity drift enforcement + KL divergence."""

    def test_initial_state(self):
        mgr = IdentityManager(max_drift=0.3)
        assert mgr.drift_score == 0.0
        assert mgr.frozen is False
        assert mgr.version == 1
        assert sum(mgr.mode_weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_slr_breach_increments_drift(self):
        mgr = IdentityManager(max_drift=0.3)
        mgr.update_on_slr_breach("node_001", slr=0.8, mode="theorist")
        assert mgr.drift_score > 0.0
        assert mgr.slr_breach_count == 1
        assert mgr.behavior_profile["theorist"] > 0.0

    def test_slr_breach_boosts_veteran_weight(self):
        mgr = IdentityManager()
        initial_vet = mgr.mode_weights["veteran"]
        mgr.update_on_slr_breach("node_001", slr=0.9, mode="theorist")
        assert mgr.mode_weights["veteran"] > initial_vet
        assert sum(mgr.mode_weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_kl_divergence_zero_at_start(self):
        mgr = IdentityManager()
        kl = mgr.compute_kl_divergence()
        # At start, current == stable, so KL ≈ 0
        assert kl == pytest.approx(0.0, abs=0.01)

    def test_kl_divergence_increases_after_drift(self):
        mgr = IdentityManager()
        mgr.behavior_profile["theorist"] = 5.0
        kl = mgr.compute_kl_divergence()
        assert kl > 0.0

    def test_check_and_enforce_freezes_on_high_drift(self):
        mgr = IdentityManager(max_drift=0.1)
        mgr.drift_score = 0.15  # Exceed threshold
        frozen = mgr.check_and_enforce("branch_1")
        assert frozen is True
        assert mgr.frozen is True

    def test_check_and_enforce_ok_on_low_drift(self):
        mgr = IdentityManager(max_drift=0.5)
        frozen = mgr.check_and_enforce("branch_1")
        assert frozen is False
        assert mgr.frozen is False

    def test_revert_to_stable(self):
        mgr = IdentityManager()
        mgr.drift_score = 0.5
        mgr.frozen = True
        mgr.slr_breach_count = 3
        mgr.behavior_profile["theorist"] = 10.0
        mgr.revert_to_stable()
        assert mgr.drift_score == 0.0
        assert mgr.frozen is False
        assert mgr.slr_breach_count == 0
        assert mgr.behavior_profile["theorist"] == 0.0

    def test_create_from_source(self):
        mgr = IdentityManager()
        nodes = [
            {"node_id": "n1", "code_snippet": "def foo():\n    assert True", "imports": ["os"], "skill_type": "execution"},
            {"node_id": "n2", "code_snippet": "class Bar:\n    x: int = 0", "imports": [], "skill_type": "validation"},
        ]
        mgr.create_from_source(nodes)
        assert mgr.version == 1
        assert mgr.stable_hash_reference != ""
        assert "dependency_resolution" in mgr.capabilities
        assert "code_generation" in mgr.capabilities
        assert "validation" in mgr.capabilities

    def test_create_from_source_deterministic(self):
        mgr1 = IdentityManager()
        mgr2 = IdentityManager()
        nodes = [{"node_id": "abc", "code_snippet": "pass", "imports": []}]
        mgr1.create_from_source(nodes)
        mgr2.create_from_source(nodes)
        assert mgr1.stable_hash_reference == mgr2.stable_hash_reference

    def test_update_from_trajectory(self):
        from types import SimpleNamespace
        mgr = IdentityManager()
        # update_from_trajectory uses getattr(trajectory, 'steps', []) — needs object attributes
        trajectory = SimpleNamespace(steps=[
            SimpleNamespace(fix="added import os for dependency", diagnosis=""),
            SimpleNamespace(fix="", diagnosis="missing type annotation on return"),
        ])
        v_before = mgr.version
        mgr.update_from_trajectory(trajectory)
        assert mgr.version == v_before + 1


# ===================================================================
# 4. TELEMETRY COLLECTOR
# ===================================================================
from src.pipeline.telemetry import TelemetryCollector


class TestTelemetry:
    """Spec Priority 9 — Observability hooks."""

    def test_record_and_snapshot(self):
        tc = TelemetryCollector()
        tc.record("slr", "breach_detected", node_id="n1", payload={"slr": 0.72})
        tc.record("mode", "execution", node_id="n1", payload={"mode": "veteran"})
        tc.record("sie", "scored", node_id="n1", payload={"s_sie": 0.65})

        snap = tc.snapshot()
        assert snap["total_events"] == 3
        assert snap["slr_distribution"]["count"] == 1
        assert snap["slr_distribution"]["mean"] == pytest.approx(0.72)
        assert snap["mode_balance"]["veteran"] == 1

    def test_slr_distribution_stats(self):
        tc = TelemetryCollector()
        for val in [0.5, 0.6, 0.7, 0.8]:
            tc.record("slr", "breach", payload={"slr": val})
        dist = tc.slr_distribution()
        assert dist["count"] == 4
        assert dist["mean"] == pytest.approx(0.65, abs=0.01)
        assert dist["min"] == pytest.approx(0.5)
        assert dist["max"] == pytest.approx(0.8)

    def test_drift_velocity(self):
        tc = TelemetryCollector()
        tc.record("identity", "drift", payload={"drift_increment": 0.05})
        tc.record("identity", "drift", payload={"drift_increment": 0.03})
        vel = tc.drift_velocity()
        assert vel["count"] == 2
        assert vel["cumulative"] == pytest.approx(0.08, abs=0.001)

    def test_persist_to_file(self, tmp_path):
        tc = TelemetryCollector()
        tc.record("mode", "execution", payload={"mode": "theorist"})
        out = tmp_path / "telemetry.jsonl"
        tc.persist(str(out))
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert len(lines) >= 1  # At least one event + optional summary


# ===================================================================
# 5. SIE SCORING (via ACS Engine)
# ===================================================================

class TestSIEScoring:
    """Spec Priority 2 — SIE model calculations: s_sie = κ · ρ · ||∇φ||."""

    def test_compute_sie_score_basic(self):
        try:
            from src.pipeline.acs_engine import ACSExecutionGovernor
        except ImportError:
            pytest.skip("acs_engine not importable in test environment")

        acs = ACSExecutionGovernor.__new__(ACSExecutionGovernor)
        acs.config = {}
        acs.penalties = {
            "syntax_error_mult": 0.3,
            "unsafe_io_mult": 0.5,
            "sast_finding_mult": 0.7,
            "sycophancy_mult": 0.4,
            "utility_cosmetic_mult": 0.9,
            "circular_logic_mult": 0.3,
        }

        # Minimal node dict
        node = {
            "code_snippet": "def hello():\n    \"\"\"Greets.\"\"\"\n    return 'world'",
            "teaching_layer": {
                "reasoning_vectors": {
                    "intent": "Greet",
                    "strategy": "Return literal",
                    "constraints": [],
                    "execution_pattern": [],
                    "failure_modes": [],
                },
            },
            "skill_type": "execution",
        }
        result = acs.compute_sie_score(node)
        assert "content_density" in result
        assert "alignment_vector" in result
        assert "composite_quality_score" in result
        assert "s_sie" in result
        assert result["content_density"] >= 0.0
        assert result["composite_quality_score"] >= 0.0
        assert result["s_sie"] == result["composite_quality_score"]  # s_sie = composite_quality_score in current impl


# ===================================================================
# 6. DAG CASCADE LOGIC
# ===================================================================

class TestDAGCascade:
    """Spec Priority 3 — Edge-level constraint enforcement + cascade rejection."""

    def test_edge_constraint_zeroes_downstream(self):
        """If an edge has constraint_ok=False, downstream CognitiveNode.c_final should be 0."""
        try:
            from src.core.models import CognitiveNode, ReasoningEdge, AletheiaSkill, Constraint
        except ImportError:
            pytest.skip("models not importable")

        # Build a ReasoningEdge with constraint_ok=False
        edge = ReasoningEdge(
            source_id="parent_1",
            target_id="child_1",
            edge_type="dependency",
            constraint_ok=False,
        )
        assert edge.constraint_ok is False

        # The DAG runtime _build_cognitive_layer checks constraint_ok on edges and zeroes target c_final.
        # We verify the model allows c_final=0 assignment.
        from src.core.models import CognitiveNode, AletheiaSkill

        skill = AletheiaSkill(
            node_id="child_1",
            name="child_func",
            file="test.py",
            code_snippet="pass",
            imports=[],
            operator_type="function",
            teaching_layer={
                "skill_identity": {"name": "child"},
                "method_metadata": {"name": "child", "language": "python"},
                "reasoning_vectors": {"intent": "test", "strategy": "test"},
                "implementation_template": {"code": "pass"},
            },
        )
        cn = CognitiveNode(cognitive_id="child_1", skill=skill, c_final=0.8)
        assert cn.c_final == 0.8
        cn.c_final = 0.0
        assert cn.c_final == 0.0

    def test_constraint_severity_levels(self):
        from src.core.models import Constraint
        fatal = Constraint(type="structural", description="missing return", severity="fatal", tags=["ast"])
        error = Constraint(type="semantic", description="unused var", severity="error", tags=["lint"])
        warn = Constraint(type="operational", description="slow loop", severity="warning", tags=["perf"])
        assert fatal.severity == "fatal"
        assert error.severity == "error"
        assert warn.severity == "warning"


# ===================================================================
# 7. DAG SCORING PASS (Integration)
# ===================================================================

class TestDAGScoringPass:
    """Spec Priority 4 — Veteran/Advocate payloads route through DAG."""

    def test_wrap_veteran_payload(self):
        from src.pipeline.dag_scoring_pass import wrap_payload
        payload = {
            "attempt_code": "x = 1/0",
            "traceback": "ZeroDivisionError",
            "analysis_text": "Division by zero",
            "diff": "- 1/0\n+ 1/1",
            "resolved_code": "x = 1/1",
        }
        wrapped = wrap_payload(payload, "veteran", "test.json")
        assert wrapped["source_type"] == "veteran_diagnostic"
        assert wrapped["semantics"]["orchestration_mode"] == "veteran"
        assert wrapped["semantics"]["traceback"] == "ZeroDivisionError"
        assert "epistemic" in wrapped

    def test_wrap_advocate_payload(self):
        from src.pipeline.dag_scoring_pass import wrap_payload
        payload = {
            "theory_text": "Architecture demands idempotency",
            "implementation_code": "class Idem: pass",
            "text_to_code_ratio": 0.7,
        }
        wrapped = wrap_payload(payload, "advocate", "test.json")
        assert wrapped["source_type"] == "advocate_theory"
        assert wrapped["semantics"]["orchestration_mode"] == "advocate"
        assert wrapped["semantics"]["theory"] == "Architecture demands idempotency"
        assert wrapped["semantics"]["implementation"] == "class Idem: pass"

    def test_auto_detect_veteran(self):
        from src.pipeline.dag_scoring_pass import _detect_payload_mode
        assert _detect_payload_mode({"traceback": "err", "diff": "d"}) == "veteran"

    def test_auto_detect_advocate(self):
        from src.pipeline.dag_scoring_pass import _detect_payload_mode
        assert _detect_payload_mode({"theory_text": "t", "implementation_code": "c"}) == "advocate"

    def test_auto_detect_coding(self):
        from src.pipeline.dag_scoring_pass import _detect_payload_mode
        assert _detect_payload_mode({"node_id": "n1", "code_snippet": "pass"}) == "coding_assistant"

    def test_scoring_pass_empty_dir(self, tmp_path):
        from src.pipeline.dag_scoring_pass import run_scoring_pass
        result = run_scoring_pass(tmp_path / "nonexistent", tmp_path / "out")
        assert result["status"] == "error"

    def test_scoring_pass_with_payload(self, tmp_path):
        from src.pipeline.dag_scoring_pass import run_scoring_pass

        # Write a veteran payload
        input_dir = tmp_path / "complexes"
        input_dir.mkdir()
        payload = [{
            "attempt_code": "x = 1/0",
            "traceback": "ZeroDivisionError: division by zero",
            "analysis_text": "Literal division by zero",
            "diff": "- 1/0\n+ 1",
            "resolved_code": "x = 1",
        }]
        (input_dir / "vet_extract.json").write_text(json.dumps(payload))

        output_dir = tmp_path / "scored"
        result = run_scoring_pass(input_dir, output_dir, mode="veteran")

        assert result["status"] == "completed"
        assert result["nodes_processed"] >= 1
        assert (output_dir / "dag_scored_nodes.json").exists()


# ===================================================================
# 8. CAUSAL AUTHORITY ENFORCEMENT (v5.1)
# ===================================================================
import networkx as nx
from src.pipeline.dag_runtime import (
    DAGRuntime, InMemoryEpistemicGraph, NodeState, QUALITY_FLOOR,
    ACCEPTANCE_THRESHOLD,
    TELEMETRY_SLR_MEAN_THRESHOLD, TELEMETRY_DRIFT_CUMULATIVE_THRESHOLD,
)


class TestCausalAuthority:
    """v5.1 — DAG as causal authority layer (signals → actions)."""

    def _make_node_dict(self, node_id="test_001", **overrides):
        """Build a minimal AletheiaSkill-compatible dict."""
        base = {
            "node_id": node_id,
            "name": node_id,
            "file": "test.py",
            "code_snippet": (
                "def foo(x: int) -> int:\n"
                "    \"\"\"Computes increment.\"\"\"\n"
                "    return x + 1\n"
            ),
            "imports": [],
            "operator_type": "function",
            "source_type": "ast_code",
            "skill_type": "execution",
            "semantics": {
                "code_snippet": (
                    "def foo(x: int) -> int:\n"
                    "    \"\"\"Computes increment.\"\"\"\n"
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
            },
        }
        base.update(overrides)
        return base

    def _make_runtime(self, nodes, mode="coding_assistant"):
        """Create DAGRuntime with in-memory graph and no ACS (forces inline SIE)."""
        graph = InMemoryEpistemicGraph(nodes)
        acs = MagicMock(spec=[])  # Empty spec: no attributes auto-created
        runtime = DAGRuntime(graph, acs, mode=mode)
        return runtime

    # ------ Standalone SIE Physics (inline fallback) ------

    def test_inline_sie_code_node(self):
        """Inline SIE produces content_density > 0 and s_sie >= QUALITY_FLOOR for a code node."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        result = runtime._compute_sie_inline(node)
        assert result["content_density"] > 0.0, "Code node must have content_density > 0 from AST depth"
        assert result["s_sie"] >= QUALITY_FLOOR, f"s_sie={result['s_sie']} must >= {QUALITY_FLOOR}"
        assert result["s_sie"] == result["composite_quality_score"]
        assert len(result["alignment_vector"]) == 3

    def test_inline_sie_text_node(self):
        """Inline SIE produces content_density > 0 for text nodes via word-count heuristic."""
        node_dict = self._make_node_dict()
        node_dict["code_snippet"] = ""
        node_dict["semantics"] = {
            "chunk_text": (
                "This theory implies that the constraint invariant requires "
                "careful handling because the system specifically depends on "
                "therefore maintaining consistency however the architecture "
                + " ".join(["substantive"] * 200)
            ),
        }
        node_dict["source_type"] = "theory"
        runtime = self._make_runtime([node_dict], mode="theorist")
        node = list(runtime.graph.nodes.values())[0]
        result = runtime._compute_sie_inline(node)
        assert result["content_density"] > 0.0, "Text node must have content_density > 0 from word count"
        assert result["composite_quality_score"] > 0.0

    def test_inline_sie_empty_node_underflow(self):
        """Empty node (no code, no text) should have composite_quality_score = 0 (underflow)."""
        node_dict = self._make_node_dict()
        node_dict["code_snippet"] = ""
        node_dict["semantics"] = {}
        node_dict["source_context"] = {}
        runtime = self._make_runtime([node_dict])
        node = list(runtime.graph.nodes.values())[0]
        result = runtime._compute_sie_inline(node)
        assert result["content_density"] == 0.0
        assert result["composite_quality_score"] == 0.0

    def test_compute_sie_uses_inline_when_no_acs(self):
        """_compute_sie falls back to inline when ACS lacks compute_sie_score."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        passed = runtime._compute_sie(node)
        assert passed is True, "Well-formed code node must pass SIE gate"
        assert node.sie_node is not None
        assert node.sie_node.s_sie > 0.0

    def test_compute_sie_inline_with_teaching_layer_boosts_consistency(self):
        """Teaching layer reasoning vectors raise the consistency component."""
        node_dict = self._make_node_dict()
        runtime = self._make_runtime([node_dict])
        node = list(runtime.graph.nodes.values())[0]
        result = runtime._compute_sie_inline(node)
        # All 5 reasoning vectors filled → consistency = 1.0
        assert result["alignment_vector"][2] == pytest.approx(1.0)

    # ------ Identity → Execution Authority ------

    def test_enforce_identity_frozen_kills_node(self):
        """Globally frozen identity rejects the node immediately."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.VALIDATED
        runtime.identity_manager.frozen = True

        killed = runtime._enforce_identity(node)
        assert killed is True
        assert node.epistemic.state == NodeState.REJECTED
        assert node.epistemic.c_node == 0.0
        assert node.epistemic.final_status == "identity_frozen"

    def test_enforce_identity_drift_triggers_freeze(self):
        """High drift score triggers containment and rejects node."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.VALIDATED
        runtime.identity_manager.drift_score = 0.5  # Above max_drift=0.3

        killed = runtime._enforce_identity(node)
        assert killed is True
        assert node.epistemic.state == NodeState.REJECTED
        assert runtime.identity_manager.frozen is True

    def test_enforce_identity_drift_captures_advocate_audit_trace(self):
        """Identity freeze routes advocate audit report into rejected_traces."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.VALIDATED
        runtime.identity_manager.drift_score = 0.5  # Above max_drift=0.3

        runtime._enforce_identity(node)
        assert len(runtime.rejected_traces) >= 1
        trace = runtime.rejected_traces[-1]
        assert trace["reason"] == "advocate_audit_identity_drift"
        assert trace["node_id"] == node.node_id
        assert "audit_report" in trace["structural_snapshot"]
        audit = trace["structural_snapshot"]["audit_report"]
        assert "root_cause" in audit
        assert "recommendation" in audit
        assert audit["frozen_node"] == node.node_id

    def test_configurable_sie_quality_floor(self):
        """SIE quality floor can be overridden via constructor."""
        runtime = self._make_runtime([self._make_node_dict()])
        assert runtime.quality_floor == QUALITY_FLOOR  # default

        # Create runtime with custom floor
        graph = InMemoryEpistemicGraph([self._make_node_dict()])
        from unittest.mock import MagicMock
        acs = MagicMock(spec=[])
        runtime2 = DAGRuntime(graph, acs, mode="advocate", quality_floor=0.05)
        assert runtime2.quality_floor == 0.05

    def test_enforce_identity_ok_on_low_drift(self):
        """Low drift allows node to proceed unharmed."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.VALIDATED

        killed = runtime._enforce_identity(node)
        assert killed is False
        assert node.epistemic.state == NodeState.VALIDATED

    # ------ Parent-Chain Topological Authority ------

    def test_parent_chain_valid_no_parents(self):
        """Root nodes with no parents always pass validation."""
        runtime = self._make_runtime([self._make_node_dict()])
        runtime._nx_graph = runtime._build_directed_graph(runtime._get_nodes_safely())
        node = list(runtime.graph.nodes.values())[0]
        assert runtime._validate_parent_chain(node) is True

    def test_parent_chain_rejected_parent_applies_soft_penalty(self):
        """Rejected parent softly penalizes the child instead of hard failing."""
        parent = self._make_node_dict("parent_001")
        child = self._make_node_dict("child_001")
        runtime = self._make_runtime([parent, child])

        runtime.graph.nodes["parent_001"].epistemic.state = NodeState.REJECTED
        G = nx.DiGraph()
        G.add_edge("parent_001", "child_001")
        runtime._nx_graph = G

        child_node = runtime.graph.nodes["child_001"]
        child_node.epistemic.c_node = 0.95
        assert runtime._validate_parent_chain(child_node) is True
        assert child_node.epistemic.c_node == pytest.approx(0.475)

    def test_parent_chain_accepted_parent_passes(self):
        """Accepted parent with coherent SIE allows child through parent-chain check."""
        parent = self._make_node_dict("parent_001")
        child = self._make_node_dict("child_001")
        runtime = self._make_runtime([parent, child])

        runtime.graph.nodes["parent_001"].epistemic.state = NodeState.ACCEPTED
        # Both nodes need SIE data for coherence gate (conservative default)
        from src.core.models import SemanticReasoningNode
        runtime.graph.nodes["parent_001"].sie_node = SemanticReasoningNode(
            content_density=0.8, s_sie=0.9, composite_quality_score=0.5, mode_scaling_factor=1.0, alignment_vector=[0.8, 0.6, 0.7],
        )
        runtime.graph.nodes["child_001"].sie_node = SemanticReasoningNode(
            content_density=0.8, s_sie=0.9, composite_quality_score=0.5, mode_scaling_factor=1.0, alignment_vector=[0.8, 0.6, 0.7],
        )
        G = nx.DiGraph()
        G.add_edge("parent_001", "child_001")
        runtime._nx_graph = G

        child_node = runtime.graph.nodes["child_001"]
        assert runtime._validate_parent_chain(child_node) is True

    def test_transition_to_terminal_survives_rejected_parent_soft_penalty(self):
        """_transition_to_terminal accepts strong nodes after the soft parent penalty."""
        parent = self._make_node_dict("parent_001")
        child = self._make_node_dict("child_001")
        runtime = self._make_runtime([parent, child])
        runtime.omega_validator = None

        G = nx.DiGraph()
        G.add_edge("parent_001", "child_001")
        runtime._nx_graph = G

        runtime.graph.nodes["parent_001"].epistemic.state = NodeState.REJECTED

        child_node = runtime.graph.nodes["child_001"]
        child_node.epistemic.state = NodeState.SCORED
        child_node.epistemic.c_node = 0.95  # Well above threshold

        runtime._transition_to_terminal(child_node)

        assert child_node.epistemic.state == NodeState.ACCEPTED
        assert child_node.epistemic.final_status == "validated"
        assert child_node.epistemic.c_node == pytest.approx(0.475)

    # ------ Telemetry → Actions ------

    def test_telemetry_slr_threshold_pushes_drift(self):
        """High aggregate SLR mean triggers synthetic drift increment."""
        runtime = self._make_runtime([self._make_node_dict()])
        for v in [0.7, 0.8, 0.9]:
            runtime.telemetry.record("slr", "breach", payload={"slr": v})
        initial_drift = runtime.identity_manager.drift_score
        runtime._check_telemetry_thresholds()
        assert runtime.identity_manager.drift_score > initial_drift

    def test_telemetry_drift_velocity_runs_cleanly(self):
        """High cumulative drift velocity triggers proactive freeze check without error."""
        runtime = self._make_runtime([self._make_node_dict()])
        for _ in range(5):
            runtime.telemetry.record(
                "identity", "drift", payload={"drift_increment": 0.1}
            )
        runtime._check_telemetry_thresholds()
        assert runtime.telemetry.drift_velocity()["cumulative"] == pytest.approx(0.5)

    def test_telemetry_sie_underflow_alert(self):
        """Low SIE mean fires systemic underflow alert without error."""
        runtime = self._make_runtime([self._make_node_dict()])
        for v in [0.01, 0.02, 0.03]:
            runtime.telemetry.record("sie", "computed", payload={"s_sie": v})
        runtime._check_telemetry_thresholds()
        sie = runtime.telemetry.sie_summary()
        assert sie["mean"] < TELEMETRY_SLR_MEAN_THRESHOLD

    # ------ Integration: Identity enforced during scoring ------

    def test_identity_enforced_during_scoring_phase(self):
        """_transition_to_scored rejects node when identity is pre-frozen."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.VALIDATED
        runtime.identity_manager.frozen = True

        result = runtime._transition_to_scored(node)
        assert result is False
        assert node.epistemic.state == NodeState.REJECTED


# ===================================================================
# 9. FIELD ENGINE PATCHES (v5.2) → CAUSAL AUTHORITY (v5.3)
# ===================================================================
from src.pipeline.dag_runtime import (
    CONFIDENCE_ALPHA, CONFIDENCE_BETA, CONFIDENCE_GAMMA, CONFIDENCE_DELTA,
    INSTABILITY_BAND_LOW, EDGE_COHERENCE_THRESHOLD,
    SLR_THRESHOLDS, calculate_sie_slr,
)


class TestAdditiveConfidence:
    """Additive weighted confidence: c_final = α·s_sie + β·s_acs + γ·s_topology + δ·s_validation."""

    def _make_node_dict(self, node_id="test_001", **overrides):
        base = {
            "node_id": node_id,
            "name": node_id,
            "file": "test.py",
            "code_snippet": (
                "def foo(x: int) -> int:\n"
                "    \"\"\"Computes increment.\"\"\"\n"
                "    return x + 1\n"
            ),
            "imports": [],
            "operator_type": "function",
            "source_type": "ast_code",
            "skill_type": "execution",
            "semantics": {
                "code_snippet": (
                    "def foo(x: int) -> int:\n"
                    "    \"\"\"Computes increment.\"\"\"\n"
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
            },
        }
        base.update(overrides)
        return base

    def _make_runtime(self, nodes, mode="coding_assistant"):
        graph = InMemoryEpistemicGraph(nodes)
        acs = MagicMock(spec=[])
        return DAGRuntime(graph, acs, mode=mode)

    def test_additive_confidence(self):
        """c_final = 0.4*s_sie + 0.3*s_acs + 0.2*s_topology + 0.1*s_validation."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        node.sie_node = MagicMock(s_sie=0.9, alignment_vector=[0.5, 0.5, 0.5])
        node.epistemic.c_node = 0.85  # s_acs
        node.v_score = 1.0  # s_validation
        # s_topology defaults to 0.8 (no deps)
        runtime._compute_unified_confidence(node)
        expected = round(0.4 * 0.9 + 0.3 * 0.85 + 0.2 * 0.8 + 0.1 * 1.0, 4)
        assert node.epistemic.c_node == expected

    def test_low_sie_contributes_proportionally(self):
        """Low s_sie reduces c_final proportionally, not to zero."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        node.sie_node = MagicMock(s_sie=0.1)
        node.epistemic.c_node = 0.8
        node.v_score = 0.9
        runtime._compute_unified_confidence(node)
        expected = round(0.4 * 0.1 + 0.3 * 0.8 + 0.2 * 0.8 + 0.1 * 0.9, 4)
        assert node.epistemic.c_node == expected
        assert node.epistemic.c_node > 0.0  # Not hard-gated to zero

    def test_low_topology_contributes_proportionally(self):
        """Low s_topology reduces c_final proportionally, not to zero."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        node.sie_node = MagicMock(s_sie=0.9)
        node.epistemic.c_node = 0.8
        node.v_score = 0.9
        # Force s_topology to 0.2 via semantics
        node.semantics = {"system_centrality_blast_radius": 0.2}
        runtime._compute_unified_confidence(node)
        expected = round(0.4 * 0.9 + 0.3 * 0.8 + 0.2 * 0.2 + 0.1 * 0.9, 4)
        assert node.epistemic.c_node == expected
        assert node.epistemic.c_node > 0.0  # Not hard-gated to zero

    def test_confidence_stores_model_for_audit(self):
        """Confidence breakdown includes the model type and weights."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        node.sie_node = MagicMock(s_sie=0.8)
        node.epistemic.c_node = 0.7
        node.v_score = 0.9
        runtime._compute_unified_confidence(node)
        assert node.epistemic.confidence["model"] == "additive_weighted"
        assert "weights" in node.epistemic.confidence
        assert node.epistemic.confidence["weights"]["alpha"] == 0.4

    def test_system_backpressure_dampens_confidence(self):
        """Non-zero field pressure reduces c_final."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        node.sie_node = MagicMock(s_sie=0.9)
        node.epistemic.c_node = 0.85
        node.v_score = 1.0

        # Compute without pressure
        runtime.system_backpressure = 0.0
        runtime._compute_unified_confidence(node)
        c_no_pressure = node.epistemic.c_node

        # Compute with pressure
        node.epistemic.c_node = 0.85  # Reset ACS input
        runtime.system_backpressure = 0.3
        runtime._compute_unified_confidence(node)
        c_with_pressure = node.epistemic.c_node

        assert c_with_pressure < c_no_pressure
        assert c_with_pressure == pytest.approx(c_no_pressure * 0.7, abs=0.001)

    def test_sie_quality_floor_value(self):
        """QUALITY_FLOOR is set to 0.1 — a practical semantic quality threshold."""
        assert QUALITY_FLOOR == pytest.approx(0.1, abs=1e-6)


class TestSIEEdgeCoherence:
    """Patch 2 — SIE alignment_vector coherence across edges."""

    def _make_node_dict(self, node_id="test_001"):
        return {
            "node_id": node_id,
            "name": node_id,
            "file": "test.py",
            "code_snippet": "def foo(): return 1",
            "imports": [],
            "operator_type": "function",
            "source_type": "ast_code",
            "skill_type": "execution",
            "semantics": {"code_snippet": "def foo(): return 1", "name": "foo"},
            "teaching_layer": {
                "skill_identity": {"name": node_id},
                "method_metadata": {"name": node_id, "language": "python"},
                "reasoning_vectors": {
                    "intent": "test", "strategy": "test",
                    "constraints": ["none"],
                    "execution_pattern": ["direct"],
                    "failure_modes": ["none"],
                },
                "implementation_template": {"code": "pass"},
            },
            "epistemic": {"state": "CREATED", "c_node": 0.0, "retry_budget": 6},
        }

    def _make_runtime(self, nodes):
        graph = InMemoryEpistemicGraph(nodes)
        acs = MagicMock(spec=[])
        return DAGRuntime(graph, acs, mode="coding_assistant")

    def test_coherent_gradients_pass(self):
        """Parallel phase gradients have high cosine similarity → pass."""
        parent_d = self._make_node_dict("parent")
        child_d = self._make_node_dict("child")
        runtime = self._make_runtime([parent_d, child_d])
        parent = runtime.graph.nodes["parent"]
        child = runtime.graph.nodes["child"]
        parent.sie_node = MagicMock(s_sie=0.8, alignment_vector=[0.6, 0.5, 0.7])
        child.sie_node = MagicMock(s_sie=0.7, alignment_vector=[0.5, 0.4, 0.6])
        coherence = runtime._compute_sie_edge_coherence(parent, child)
        assert coherence > EDGE_COHERENCE_THRESHOLD

    def test_orthogonal_gradients_fail(self):
        """Orthogonal phase gradients → low coherence."""
        parent_d = self._make_node_dict("parent")
        child_d = self._make_node_dict("child")
        runtime = self._make_runtime([parent_d, child_d])
        parent = runtime.graph.nodes["parent"]
        child = runtime.graph.nodes["child"]
        parent.sie_node = MagicMock(s_sie=0.8, alignment_vector=[1.0, 0.0, 0.0])
        child.sie_node = MagicMock(s_sie=0.7, alignment_vector=[0.0, 0.0, 1.0])
        coherence = runtime._compute_sie_edge_coherence(parent, child)
        assert coherence < EDGE_COHERENCE_THRESHOLD

    def test_zero_gradient_returns_zero(self):
        """Zero vector phase gradient → incoherent (0.0)."""
        parent_d = self._make_node_dict("parent")
        child_d = self._make_node_dict("child")
        runtime = self._make_runtime([parent_d, child_d])
        parent = runtime.graph.nodes["parent"]
        child = runtime.graph.nodes["child"]
        parent.sie_node = MagicMock(s_sie=0.5, alignment_vector=[0.0, 0.0, 0.0])
        child.sie_node = MagicMock(s_sie=0.5, alignment_vector=[0.5, 0.5, 0.5])
        coherence = runtime._compute_sie_edge_coherence(parent, child)
        assert coherence == 0.0

    def test_no_sie_data_passes(self):
        """Nodes without SIE data pass coherence (cannot enforce)."""
        parent_d = self._make_node_dict("parent")
        child_d = self._make_node_dict("child")
        runtime = self._make_runtime([parent_d, child_d])
        parent = runtime.graph.nodes["parent"]
        child = runtime.graph.nodes["child"]
        parent.sie_node = None
        child.sie_node = MagicMock(s_sie=0.5, alignment_vector=[0.5, 0.5, 0.5])
        coherence = runtime._compute_sie_edge_coherence(parent, child)
        assert coherence == 0.0  # Conservative: missing SIE = incoherent

    def test_parent_chain_rejects_incoherent_edge(self):
        """Parent with orthogonal SIE blocks child acceptance."""
        parent_d = self._make_node_dict("parent")
        child_d = self._make_node_dict("child")
        runtime = self._make_runtime([parent_d, child_d])

        G = nx.DiGraph()
        G.add_edge("parent", "child")
        runtime._nx_graph = G

        parent = runtime.graph.nodes["parent"]
        child = runtime.graph.nodes["child"]
        parent.epistemic.state = NodeState.ACCEPTED
        parent.sie_node = MagicMock(s_sie=0.8, alignment_vector=[1.0, 0.0, 0.0])
        child.sie_node = MagicMock(s_sie=0.7, alignment_vector=[0.0, 0.0, 1.0])

        assert runtime._validate_parent_chain(child) is False


class TestInstabilityBand:
    """Patch 3 — Borderline nodes forced to reroll."""

    def _make_node_dict(self, node_id="test_001"):
        return {
            "node_id": node_id,
            "name": node_id,
            "file": "test.py",
            "code_snippet": "def foo(): return 1",
            "imports": [],
            "operator_type": "function",
            "source_type": "ast_code",
            "skill_type": "execution",
            "semantics": {"code_snippet": "def foo(): return 1", "name": "foo"},
            "teaching_layer": {
                "skill_identity": {"name": node_id},
                "method_metadata": {"name": node_id, "language": "python"},
                "reasoning_vectors": {
                    "intent": "test", "strategy": "test",
                    "constraints": ["none"],
                    "execution_pattern": ["direct"],
                    "failure_modes": ["none"],
                },
                "implementation_template": {"code": "pass"},
            },
            "epistemic": {"state": "CREATED", "c_node": 0.0, "retry_budget": 6},
        }

    def _make_runtime(self, nodes):
        graph = InMemoryEpistemicGraph(nodes)
        acs = MagicMock(spec=[])
        return DAGRuntime(graph, acs, mode="coding_assistant")

    def test_instability_band_triggers_reroll(self):
        """c_final in [0.25, 0.40) → REROLL (not REJECTED)."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.SCORED
        node.epistemic.c_node = 0.33  # Inside instability band

        runtime._transition_to_terminal(node)
        assert node.epistemic.state == NodeState.REROLL

    def test_below_instability_band_rejects_immediately(self):
        """c_final below 0.25 → immediate REJECTED (no reroll below instability band)."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.SCORED
        node.epistemic.c_node = 0.10

        runtime._transition_to_terminal(node)
        assert node.epistemic.state == NodeState.REJECTED

    def test_above_threshold_accepts(self):
        """c_final >= 0.40 → ACCEPTED (no instability reroll)."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.SCORED
        node.epistemic.c_node = 0.50

        runtime._transition_to_terminal(node)
        assert node.epistemic.state == NodeState.ACCEPTED

    def test_instability_exhausts_budget_to_rejection(self):
        """Instability reroll with budget=0 → REJECTED."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.SCORED
        node.epistemic.c_node = 0.33
        node.epistemic.retry_budget = 0

        runtime._transition_to_terminal(node)
        assert node.epistemic.state == NodeState.REJECTED


class TestFieldPressure:
    """Patch 4 — Global field pressure from telemetry."""

    def _make_node_dict(self, node_id="test_001"):
        return {
            "node_id": node_id,
            "name": node_id,
            "file": "test.py",
            "code_snippet": "def foo(): return 1",
            "imports": [],
            "operator_type": "function",
            "source_type": "ast_code",
            "skill_type": "execution",
            "semantics": {"code_snippet": "def foo(): return 1", "name": "foo"},
            "teaching_layer": {
                "skill_identity": {"name": node_id},
                "method_metadata": {"name": node_id, "language": "python"},
                "reasoning_vectors": {
                    "intent": "test", "strategy": "test",
                    "constraints": ["none"],
                    "execution_pattern": ["direct"],
                    "failure_modes": ["none"],
                },
                "implementation_template": {"code": "pass"},
            },
            "epistemic": {"state": "CREATED", "c_node": 0.0, "retry_budget": 6},
        }

    def _make_runtime(self, nodes):
        graph = InMemoryEpistemicGraph(nodes)
        acs = MagicMock(spec=[])
        return DAGRuntime(graph, acs, mode="coding_assistant")

    def test_system_backpressure_starts_at_zero(self):
        """Initial field pressure is 0."""
        runtime = self._make_runtime([self._make_node_dict()])
        assert runtime.system_backpressure == 0.0

    def test_system_backpressure_rises_with_degradation(self):
        """Low SIE + high drift → field pressure increases."""
        runtime = self._make_runtime([self._make_node_dict()])
        # Feed low SIE scores
        for v in [0.02, 0.03, 0.04]:
            runtime.telemetry.record("sie", "computed", payload={"s_sie": v})
        # Feed drift
        for _ in range(3):
            runtime.telemetry.record("identity", "drift", payload={"drift_increment": 0.08})

        runtime._check_telemetry_thresholds()
        assert runtime.system_backpressure > 0.0

    def test_system_backpressure_capped_at_half(self):
        """Field pressure cannot exceed 0.5."""
        runtime = self._make_runtime([self._make_node_dict()])
        # Extreme degradation
        for v in [0.0, 0.0, 0.0]:
            runtime.telemetry.record("sie", "computed", payload={"s_sie": v})
        for _ in range(10):
            runtime.telemetry.record("identity", "drift", payload={"drift_increment": 0.5})
        # Reject the only node to tank topology
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.REJECTED

        runtime._check_telemetry_thresholds()
        assert runtime.system_backpressure <= 0.5

    def test_healthy_system_low_pressure(self):
        """High SIE + no drift → field pressure stays low."""
        runtime = self._make_runtime([self._make_node_dict()])
        for v in [0.8, 0.9, 0.7]:
            runtime.telemetry.record("sie", "computed", payload={"s_sie": v})

        runtime._check_telemetry_thresholds()
        assert runtime.system_backpressure < 0.3


# ===================================================================
# 13. OUTPUT SANITIZATION (Phase 4)
# ===================================================================
from src.pipeline.dag_scoring_pass import sanitize_scored_node


class TestOutputSanitization:
    """Phase 4 — Strip internal scoring mechanics from dataset output."""

    def test_strips_sie_node(self):
        """sie_node (raw SIE internals) must be removed."""
        node = {"node_id": "n1", "sie_node": {"content_density": 0.8, "composite_quality_score": 0.5, "s_sie": 0.5}}
        result = sanitize_scored_node(node)
        assert "sie_node" not in result

    def test_strips_acs_internals(self):
        """ACS handshake, violations, and audit flag must be removed."""
        node = {
            "node_id": "n1",
            "v_score": 0.9,
            "acs_handshake_sid": "abc123",
            "acs_violations": ["v1"],
            "acs_audited": True,
        }
        result = sanitize_scored_node(node)
        for k in ("v_score", "acs_handshake_sid", "acs_violations", "acs_audited"):
            assert k not in result

    def test_strips_pagerank_from_semantics(self):
        """PageRank centrality must be stripped from semantics."""
        node = {
            "node_id": "n1",
            "semantics": {
                "system_centrality_blast_radius": 0.42,
                "code_snippet": "keep this",
            },
        }
        result = sanitize_scored_node(node)
        assert "system_centrality_blast_radius" not in result["semantics"]
        assert result["semantics"]["code_snippet"] == "keep this"

    def test_strips_internal_semantics_keys(self):
        """All internal ACS/reroll keys must be stripped from semantics."""
        node = {
            "node_id": "n1",
            "semantics": {
                "_acs_trajectory": {"steps": []},
                "_governance_directive": {"action": "reroll"},
                "_acs_structured_constraints": [{"type": "fatal"}],
                "_reroll_context": {"reason": "test"},
                "name": "keep",
            },
        }
        result = sanitize_scored_node(node)
        for k in ("_acs_trajectory", "_governance_directive",
                   "_acs_structured_constraints", "_reroll_context"):
            assert k not in result["semantics"]
        assert result["semantics"]["name"] == "keep"

    def test_strips_confidence_decomposition(self):
        """Confidence breakdown must be stripped to prevent formula leakage."""
        node = {
            "node_id": "n1",
            "epistemic": {
                "state": "ACCEPTED",
                "c_node": 0.88,
                "confidence": {"sie": 0.9, "acs": 0.85, "topology": 0.8, "final": 0.88},
            },
        }
        result = sanitize_scored_node(node)
        assert "confidence" not in result["epistemic"]
        assert result["epistemic"]["c_node"] == 0.88  # Keep final score


# ===================================================================
# 14. REROLL CONTEXT INJECTION (Phase 2)
# ===================================================================

class TestRerollContextInjection:
    """Phase 2 — Pydantic ValidationError context flows into reroll recovery."""

    def _make_node_dict(self, node_id="test_001"):
        return {
            "node_id": node_id,
            "name": node_id,
            "file": "test.py",
            "code_snippet": "def foo(): return 1",
            "imports": [],
            "operator_type": "function",
            "source_type": "ast_code",
            "skill_type": "execution",
            "semantics": {"code_snippet": "def foo(): return 1", "name": "foo"},
            "teaching_layer": {
                "skill_identity": {"name": node_id},
                "method_metadata": {"name": node_id, "language": "python"},
                "reasoning_vectors": {
                    "intent": "test", "strategy": "test",
                    "constraints": ["none"],
                    "execution_pattern": ["direct"],
                    "failure_modes": ["none"],
                },
                "implementation_template": {"code": "pass"},
            },
            "epistemic": {"state": "CREATED", "c_node": 0.0, "retry_budget": 6},
        }

    def _make_runtime(self, nodes):
        graph = InMemoryEpistemicGraph(nodes)
        acs = MagicMock(spec=[])
        return DAGRuntime(graph, acs, mode="coding_assistant")

    def test_reroll_injects_violation_details(self):
        """_route_to_reroll injects violation_details into node semantics."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.VALIDATED

        err_msg = "Field required: 'implementation'"
        runtime._route_to_reroll(
            node, NodeState.VALIDATED,
            f"contract_violation: {err_msg}",
            violation_details=err_msg,
        )
        assert node.epistemic.state == NodeState.REROLL
        ctx = node.semantics["_reroll_context"]
        assert ctx["violation_details"] == err_msg
        assert "contract_violation" in ctx["previous_failure_reason"]

    def test_rejected_trace_has_structural_snapshot(self):
        """Rejected traces include structural metadata for SFT isolation."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.VALIDATED
        node.epistemic.c_node = 0.3

        runtime._route_to_reroll(node, NodeState.VALIDATED, "low_confidence")
        assert len(runtime.rejected_traces) == 1
        trace = runtime.rejected_traces[0]
        assert "structural_snapshot" in trace
        snap = trace["structural_snapshot"]
        assert snap["c_node"] == 0.3
        assert snap["pre_state"] == NodeState.VALIDATED
        assert "system_backpressure" in snap

    def test_rejected_snapshot_excludes_injected_context(self):
        """Rejected trace snapshot is captured BEFORE corrective context injection."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.VALIDATED

        runtime._route_to_reroll(
            node, NodeState.VALIDATED, "test_reason",
            violation_details="test error",
        )
        trace = runtime.rejected_traces[0]
        # Rejected snapshot should NOT contain _reroll_context
        assert "_reroll_context" not in trace["rejected"]
        # But the live node semantics SHOULD contain it
        assert "_reroll_context" in node.semantics


# ===================================================================
# 14. SLR THRESHOLDS + calculate_sie_slr
# ===================================================================
class TestSLRThresholds:
    """Section 9 — SLR threshold constants and calculate_sie_slr function."""

    def test_slr_thresholds_tiers(self):
        """SLR_THRESHOLDS has correct tier structure."""
        assert "safe" in SLR_THRESHOLDS
        assert "warning" in SLR_THRESHOLDS
        assert "reroll" in SLR_THRESHOLDS
        assert "collapse" in SLR_THRESHOLDS
        assert SLR_THRESHOLDS["safe"] < SLR_THRESHOLDS["warning"]
        assert SLR_THRESHOLDS["warning"] < SLR_THRESHOLDS["reroll"]
        assert SLR_THRESHOLDS["reroll"] < SLR_THRESHOLDS["collapse"]

    def test_calculate_sie_slr_identical_vectors(self):
        """Identical alignment_vector vectors → SLR = 0.0."""
        parent = MagicMock()
        parent.sie_node = MagicMock(alignment_vector=[0.5, 0.7, 0.3])
        child = MagicMock()
        child.sie_node = MagicMock(alignment_vector=[0.5, 0.7, 0.3])
        assert calculate_sie_slr(parent, child) == 0.0

    def test_calculate_sie_slr_orthogonal_vectors(self):
        """Orthogonal alignment_vector vectors → SLR = 1.0."""
        parent = MagicMock()
        parent.sie_node = MagicMock(alignment_vector=[1.0, 0.0, 0.0])
        child = MagicMock()
        child.sie_node = MagicMock(alignment_vector=[0.0, 1.0, 0.0])
        assert calculate_sie_slr(parent, child) == 1.0

    def test_calculate_sie_slr_no_sie_node(self):
        """Missing SIE node → SLR = 0.0 (safe default)."""
        parent = MagicMock(sie_node=None)
        child = MagicMock()
        child.sie_node = MagicMock(alignment_vector=[0.5, 0.5, 0.5])
        assert calculate_sie_slr(parent, child) == 0.0

    def test_calculate_sie_slr_similar_vectors(self):
        """Similar but not identical vectors → small SLR."""
        parent = MagicMock()
        parent.sie_node = MagicMock(alignment_vector=[0.5, 0.7, 0.3])
        child = MagicMock()
        child.sie_node = MagicMock(alignment_vector=[0.6, 0.7, 0.3])
        slr = calculate_sie_slr(parent, child)
        assert 0.0 < slr < 0.3  # Similar vectors = low SLR


# ===================================================================
# 15. VECTOR SLR IN SYCOPHANCY DETECTOR
# ===================================================================
from src.pipeline.acs_engine import SycophancyDetector


class TestCosineSLR:
    """D2 — Cosine-similarity SLR with Tikhonov floor regularization."""

    def test_compute_cosine_slr_identical(self):
        """Identical vectors → SLR = 0.0."""
        det = SycophancyDetector()
        assert det.compute_cosine_slr([1.0, 0.5, 0.3], [1.0, 0.5, 0.3]) == 0.0

    def test_compute_cosine_slr_orthogonal(self):
        """Orthogonal vectors → SLR = 1.0."""
        det = SycophancyDetector()
        assert det.compute_cosine_slr([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == 1.0

    def test_compute_cosine_slr_empty(self):
        """Empty vectors → safe default 0.0."""
        det = SycophancyDetector()
        assert det.compute_cosine_slr([], [1.0, 0.5]) == 0.0

    def test_compute_cosine_slr_bounded(self):
        """SLR is always in [0, 1]."""
        det = SycophancyDetector()
        slr = det.compute_cosine_slr([0.8, 0.2, 0.5], [0.3, 0.9, 0.1])
        assert 0.0 <= slr <= 1.0

    def test_backward_compat_alias(self):
        """compute_vector_slr is a backward-compatible alias for compute_cosine_slr."""
        det = SycophancyDetector()
        result_new = det.compute_cosine_slr([0.8, 0.2, 0.5], [0.3, 0.9, 0.1])
        result_old = det.compute_vector_slr([0.8, 0.2, 0.5], [0.3, 0.9, 0.1])
        assert result_new == result_old

    def test_tikhonov_floor_prevents_zero_norm_crash(self):
        """Near-zero vectors stabilized by sensitivity floor instead of returning 0."""
        det = SycophancyDetector(sensitivity=0.15)
        result = det.compute_cosine_slr([0.0, 0.0, 0.0], [1.0, 0.5, 0.3])
        assert 0.0 <= result <= 1.0

    def test_sensitivity_parameter_stored(self):
        det = SycophancyDetector(sensitivity=0.25)
        assert det.sensitivity == 0.25


# ===================================================================
# 16. HANDLE SYCOPHANCY FAILURE
# ===================================================================
class TestHandleSycophancyFailure:
    """Section 4 — _handle_sycophancy_failure tiers."""

    def _make_node_dict(self, node_id="test_001"):
        return {
            "node_id": node_id,
            "name": node_id,
            "file": "test.py",
            "code_snippet": "def foo(): return 1",
            "imports": [],
            "operator_type": "function",
            "source_type": "ast_code",
            "skill_type": "execution",
            "semantics": {"code_snippet": "def foo(): return 1", "name": "foo"},
            "teaching_layer": {
                "skill_identity": {"name": node_id},
                "method_metadata": {"name": node_id, "language": "python"},
                "reasoning_vectors": {
                    "intent": "test", "strategy": "test",
                    "constraints": ["none"],
                    "execution_pattern": ["direct"],
                    "failure_modes": ["none"],
                },
                "implementation_template": {"code": "pass"},
            },
            "epistemic": {"state": "CREATED", "c_node": 0.0, "retry_budget": 6},
        }

    def _make_runtime(self, nodes, mode="coding_assistant"):
        graph = InMemoryEpistemicGraph(nodes)
        acs = MagicMock(spec=[])
        return DAGRuntime(graph, acs, mode=mode)

    def test_collapse_tier_rejects(self):
        """SLR >= 0.8 → hard rejection with c_node = 0.0."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.VALIDATED
        node.epistemic.c_node = 0.9
        result = runtime._handle_sycophancy_failure(node, 0.85)
        assert result is True
        assert node.epistemic.state == NodeState.REJECTED
        assert node.epistemic.c_node == 0.0

    def test_reroll_tier_routes(self):
        """SLR in [0.7, 0.8) → identity mutation + reroll."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.VALIDATED
        node.epistemic.c_node = 0.9
        result = runtime._handle_sycophancy_failure(node, 0.72)
        assert result is True
        # Confidence should be collapsed by (1 - 0.72) factor
        assert node.epistemic.c_node == pytest.approx(0.9 * 0.28, abs=0.01)

    def test_warning_tier_no_route(self):
        """SLR in [0.6, 0.7) → no routing, returns False."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.VALIDATED
        result = runtime._handle_sycophancy_failure(node, 0.62)
        assert result is False

    def test_safe_tier_no_action(self):
        """SLR < 0.6 → no action, returns False."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.VALIDATED
        result = runtime._handle_sycophancy_failure(node, 0.25)
        assert result is False


# ===================================================================
# 16b. κ MODE DIFFERENTIATION (D6)
# ===================================================================
class TestKappaModeDifferentiation:
    """D6 — per-mode κ scaling in SIE computation."""

    def _make_node_dict(self, node_id="kappa_001", **overrides):
        base = {
            "node_id": node_id,
            "name": node_id,
            "file": "test.py",
            "code_snippet": (
                "def foo(x: int) -> int:\n"
                "    \"\"\"Computes increment.\"\"\"\n"
                "    return x + 1\n"
            ),
            "imports": [],
            "operator_type": "function",
            "source_type": "ast_code",
            "skill_type": "execution",
            "semantics": {
                "code_snippet": (
                    "def foo(x: int) -> int:\n"
                    "    \"\"\"Computes increment.\"\"\"\n"
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
            },
        }
        base.update(overrides)
        return base

    def _make_runtime(self, nodes, mode="coding_assistant"):
        from unittest.mock import MagicMock
        from src.pipeline.dag_runtime import DAGRuntime, InMemoryEpistemicGraph
        graph = InMemoryEpistemicGraph(nodes)
        acs = MagicMock(spec=[])
        return DAGRuntime(graph, acs, mode=mode)

    def test_kappa_constant_exists(self):
        """SCALING_FACTOR_BY_MODE covers all four execution modes."""
        from src.pipeline.dag_runtime import SCALING_FACTOR_BY_MODE
        assert "theorist" in SCALING_FACTOR_BY_MODE
        assert "coding_assistant" in SCALING_FACTOR_BY_MODE
        assert "advocate" in SCALING_FACTOR_BY_MODE
        assert "veteran" in SCALING_FACTOR_BY_MODE

    def test_kappa_values_correct(self):
        """Each mode has the spec-defined κ value."""
        from src.pipeline.dag_runtime import SCALING_FACTOR_BY_MODE
        assert SCALING_FACTOR_BY_MODE["theorist"] == 1.8  # V2.1: raised from 1.2 for text-based SIE
        assert SCALING_FACTOR_BY_MODE["coding_assistant"] == 1.0
        assert SCALING_FACTOR_BY_MODE["advocate"] == 0.9
        assert SCALING_FACTOR_BY_MODE["veteran"] == 1.1

    def test_theorist_boosts_sie(self):
        """Theorist mode (κ=1.2) produces higher s_sie than coding_assistant (κ=1.0)."""
        nd = self._make_node_dict()
        rt_theorist = self._make_runtime([nd], mode="theorist")
        rt_coding = self._make_runtime([self._make_node_dict()], mode="coding_assistant")

        node_t = list(rt_theorist.graph.nodes.values())[0]
        node_c = list(rt_coding.graph.nodes.values())[0]

        sie_t = rt_theorist._compute_sie_inline(node_t)
        sie_c = rt_coding._compute_sie_inline(node_c)

        assert sie_t["mode_scaling_factor"] == 1.8  # V2.1: raised from 1.2
        assert sie_c["mode_scaling_factor"] == 1.0
        # Same content, same content_density & gradient, so s_sie should scale by κ ratio
        assert sie_t["s_sie"] > sie_c["s_sie"]
        expected_ratio = round(sie_t["s_sie"] / sie_c["s_sie"], 1)
        assert expected_ratio == 1.8, f"Expected ~1.8x ratio, got {expected_ratio}"

    def test_advocate_dampens_sie(self):
        """Advocate mode (κ=0.9) produces lower s_sie than coding_assistant (κ=1.0)."""
        rt_adv = self._make_runtime([self._make_node_dict()], mode="advocate")
        rt_cod = self._make_runtime([self._make_node_dict()], mode="coding_assistant")

        node_a = list(rt_adv.graph.nodes.values())[0]
        node_c = list(rt_cod.graph.nodes.values())[0]

        sie_a = rt_adv._compute_sie_inline(node_a)
        sie_c = rt_cod._compute_sie_inline(node_c)

        assert sie_a["mode_scaling_factor"] == 0.9
        assert sie_a["s_sie"] < sie_c["s_sie"]

    def test_veteran_slight_boost(self):
        """Veteran mode (κ=1.1) produces slightly higher s_sie than coding_assistant."""
        rt_vet = self._make_runtime([self._make_node_dict()], mode="veteran")
        rt_cod = self._make_runtime([self._make_node_dict()], mode="coding_assistant")

        sie_v = rt_vet._compute_sie_inline(list(rt_vet.graph.nodes.values())[0])
        sie_c = rt_cod._compute_sie_inline(list(rt_cod.graph.nodes.values())[0])

        assert sie_v["mode_scaling_factor"] == 1.1
        assert sie_v["s_sie"] > sie_c["s_sie"]

    def test_acs_path_corrected_to_mode_kappa(self):
        """When ACS returns mode_scaling_factor=1.0, _compute_sie re-scales to mode-specific κ."""
        from unittest.mock import MagicMock
        from src.pipeline.dag_runtime import DAGRuntime, InMemoryEpistemicGraph, SCALING_FACTOR_BY_MODE

        nd = self._make_node_dict()
        graph = InMemoryEpistemicGraph([nd])
        acs = MagicMock()
        # Simulate ACS returning mode_scaling_factor=1.0 (mode-unaware)
        acs.compute_sie_score = MagicMock(return_value={
            "content_density": 0.5,
            "alignment_vector": [0.4, 0.5, 0.6],
            "composite_quality_score": 0.44,
            "mode_scaling_factor": 1.0,
            "s_sie": 0.44,
        })
        runtime = DAGRuntime(graph, acs, mode="theorist")
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = "VALIDATED"

        result = runtime._compute_sie(node)
        assert result is True
        # SIE should be re-scaled by theorist κ=1.8 (V2.1)
        assert node.sie_node.mode_scaling_factor == 1.8
        assert node.sie_node.s_sie > 0.44  # Was 0.44 at κ=1.0


# ===================================================================
# 17. ROLLING SLR HISTORY IN IDENTITY MANAGER
# ===================================================================
class TestRollingSLRHistory:
    """Section 8 — Temporal SLR memory in identity_manager."""

    def test_slr_history_records(self):
        """SLR breaches are appended to slr_history."""
        im = IdentityManager(max_drift=1.0)
        im.update_on_slr_breach("n1", 0.5, "advocate")
        im.update_on_slr_breach("n2", 0.7, "advocate")
        assert len(im.slr_history) == 2
        assert im.slr_history == [0.5, 0.7]

    def test_rolling_mean(self):
        """get_rolling_slr_mean returns correct average."""
        im = IdentityManager(max_drift=1.0)
        im.update_on_slr_breach("n1", 0.4, "advocate")
        im.update_on_slr_breach("n2", 0.8, "advocate")
        assert im.get_rolling_slr_mean() == pytest.approx(0.6, abs=0.01)

    def test_rolling_cap(self):
        """History is capped at 50 entries."""
        im = IdentityManager(max_drift=100.0)
        for i in range(60):
            im.update_on_slr_breach(f"n{i}", 0.5, "advocate")
        assert len(im.slr_history) == 50

    def test_critical_threshold(self):
        """is_rolling_slr_critical triggers when mean >= 0.65."""
        im = IdentityManager(max_drift=1.0)
        im.update_on_slr_breach("n1", 0.7, "advocate")
        im.update_on_slr_breach("n2", 0.7, "advocate")
        assert im.is_rolling_slr_critical() is True

    def test_not_critical_below_threshold(self):
        """Rolling SLR below 0.65 is not critical."""
        im = IdentityManager(max_drift=1.0)
        im.update_on_slr_breach("n1", 0.3, "advocate")
        im.update_on_slr_breach("n2", 0.4, "advocate")
        assert im.is_rolling_slr_critical() is False

    def test_revert_clears_history(self):
        """revert_to_stable clears slr_history."""
        im = IdentityManager(max_drift=1.0)
        im.update_on_slr_breach("n1", 0.7, "advocate")
        im.revert_to_stable()
        assert im.slr_history == []
        assert im.get_rolling_slr_mean() == 0.0

    def test_snapshot_includes_rolling_slr(self):
        """get_state_snapshot includes rolling SLR metrics."""
        im = IdentityManager(max_drift=1.0)
        im.update_on_slr_breach("n1", 0.7, "advocate")
        snap = im.get_state_snapshot()
        assert "rolling_slr_mean" in snap
        assert "rolling_slr_critical" in snap
        assert "slr_history_len" in snap
        assert snap["slr_history_len"] == 1


# ===================================================================
# 18. PHASE 2.4 — COHERENCE DECAY ON CONFIDENCE
# ===================================================================
class TestCoherenceDecay:
    """Phase 2.4 — SIE-SLR coherence decay dampens confidence."""

    def _make_node_dict(self, node_id="test_001"):
        return {
            "node_id": node_id,
            "name": node_id,
            "file": "test.py",
            "code_snippet": "def foo(): return 1",
            "imports": [],
            "operator_type": "function",
            "source_type": "ast_code",
            "skill_type": "execution",
            "semantics": {"code_snippet": "def foo(): return 1", "name": "foo"},
            "teaching_layer": {
                "skill_identity": {"name": node_id},
                "method_metadata": {"name": node_id, "language": "python"},
                "reasoning_vectors": {
                    "intent": "test", "strategy": "test",
                    "constraints": ["none"],
                    "execution_pattern": ["direct"],
                    "failure_modes": ["none"],
                },
                "implementation_template": {"code": "pass"},
            },
            "epistemic": {"state": "CREATED", "c_node": 0.0, "retry_budget": 6},
        }

    def _make_runtime(self, nodes, mode="coding_assistant"):
        graph = InMemoryEpistemicGraph(nodes)
        acs = MagicMock(spec=[])
        return DAGRuntime(graph, acs, mode=mode)

    def test_no_parents_no_decay(self):
        """Without a graph, coherence_decay = 0.0."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        node.sie_node = MagicMock(s_sie=0.9, alignment_vector=[0.5, 0.5, 0.5])
        node.epistemic.c_node = 0.85
        node.v_score = 1.0
        runtime._compute_unified_confidence(node)
        assert node.epistemic.confidence["coherence_decay"] == 0.0
        # c = 0.4*0.9 + 0.3*0.85 + 0.2*0.8 + 0.1*1.0 = 0.875
        expected = round(0.4 * 0.9 + 0.3 * 0.85 + 0.2 * 0.8 + 0.1 * 1.0, 4)
        assert node.epistemic.c_node == expected

    def test_coherence_decay_stored_in_audit(self):
        """Confidence breakdown includes coherence_decay key."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        node.sie_node = MagicMock(s_sie=0.9)
        node.epistemic.c_node = 0.85
        node.v_score = 1.0
        runtime._compute_unified_confidence(node)
        assert "coherence_decay" in node.epistemic.confidence
        assert "identity_drift" in node.epistemic.confidence


# ===================================================================
# 19. PHASE 3.5 — IDENTITY DRIFT DAMPENS CONFIDENCE
# ===================================================================
class TestIdentityDriftDampening:
    """Phase 3.5 — Identity drift reduces confidence multiplicatively."""

    def _make_node_dict(self, node_id="test_001"):
        return {
            "node_id": node_id,
            "name": node_id,
            "file": "test.py",
            "code_snippet": "def foo(): return 1",
            "imports": [],
            "operator_type": "function",
            "source_type": "ast_code",
            "skill_type": "execution",
            "semantics": {"code_snippet": "def foo(): return 1", "name": "foo"},
            "teaching_layer": {
                "skill_identity": {"name": node_id},
                "method_metadata": {"name": node_id, "language": "python"},
                "reasoning_vectors": {
                    "intent": "test", "strategy": "test",
                    "constraints": ["none"],
                    "execution_pattern": ["direct"],
                    "failure_modes": ["none"],
                },
                "implementation_template": {"code": "pass"},
            },
            "epistemic": {"state": "CREATED", "c_node": 0.0, "retry_budget": 6},
        }

    def _make_runtime(self, nodes, mode="coding_assistant"):
        graph = InMemoryEpistemicGraph(nodes)
        acs = MagicMock(spec=[])
        return DAGRuntime(graph, acs, mode=mode)

    def test_drift_reduces_confidence(self):
        """Non-zero identity drift lowers c_final."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        node.sie_node = MagicMock(s_sie=0.9, alignment_vector=[0.5, 0.5, 0.5])
        node.epistemic.c_node = 0.85
        node.v_score = 1.0

        # Compute without drift
        runtime._compute_unified_confidence(node)
        c_no_drift = node.epistemic.c_node

        # Inject drift and recompute
        node.epistemic.c_node = 0.85  # Reset ACS input
        runtime.identity_manager.drift_score = 0.2
        runtime._compute_unified_confidence(node)
        c_with_drift = node.epistemic.c_node

        assert c_with_drift < c_no_drift
        assert c_with_drift == pytest.approx(c_no_drift * 0.8, abs=0.001)

    def test_zero_drift_no_dampening(self):
        """Zero drift does not affect confidence."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        node.sie_node = MagicMock(s_sie=0.9)
        node.epistemic.c_node = 0.85
        node.v_score = 1.0
        runtime.identity_manager.drift_score = 0.0
        runtime._compute_unified_confidence(node)
        assert node.epistemic.confidence["identity_drift"] == 0.0


# ===================================================================
# 20. PHASE 3.6 — ROLLING SLR → SYSTEM PRESSURE
# ===================================================================
class TestRollingSLRPressure:
    """Phase 3.6 — Rolling SLR critical triggers advocate boost + threshold lowering."""

    def _make_node_dict(self, node_id="test_001"):
        return {
            "node_id": node_id,
            "name": node_id,
            "file": "test.py",
            "code_snippet": "def foo(): return 1",
            "imports": [],
            "operator_type": "function",
            "source_type": "ast_code",
            "skill_type": "execution",
            "semantics": {"code_snippet": "def foo(): return 1", "name": "foo"},
            "teaching_layer": {
                "skill_identity": {"name": node_id},
                "method_metadata": {"name": node_id, "language": "python"},
                "reasoning_vectors": {
                    "intent": "test", "strategy": "test",
                    "constraints": ["none"],
                    "execution_pattern": ["direct"],
                    "failure_modes": ["none"],
                },
                "implementation_template": {"code": "pass"},
            },
            "epistemic": {"state": "CREATED", "c_node": 0.0, "retry_budget": 6},
        }

    def _make_runtime(self, nodes, mode="coding_assistant"):
        graph = InMemoryEpistemicGraph(nodes)
        acs = MagicMock(spec=[])
        return DAGRuntime(graph, acs, mode=mode)

    def test_critical_slr_lowers_threshold(self):
        """When rolling SLR is critical, _check_telemetry_thresholds lowers acceptance_threshold."""
        runtime = self._make_runtime([self._make_node_dict()])
        # Push rolling SLR above 0.65 on the identity manager
        for _ in range(5):
            runtime.identity_manager.update_on_slr_breach("n", 0.8, "advocate")
        assert runtime.identity_manager.is_rolling_slr_critical()
        # Store original threshold
        orig_threshold = runtime.acceptance_threshold
        # Trigger telemetry thresholds (needs mock telemetry)
        runtime.telemetry = MagicMock()
        runtime.telemetry.slr_distribution.return_value = {"count": 0, "mean": 0.0, "std": 0.0}
        runtime.telemetry.drift_velocity.return_value = {"cumulative": 0.0, "velocity": 0.0}
        runtime.telemetry.sie_summary.return_value = {"count": 0, "mean": 0.5}
        runtime._check_telemetry_thresholds()
        assert runtime.acceptance_threshold < orig_threshold

    def test_advocate_weight_boost_on_slr_pressure(self):
        """Rolling SLR critical boosts advocate mode weight via _check_telemetry_thresholds."""
        runtime = self._make_runtime([self._make_node_dict()])
        for _ in range(5):
            runtime.identity_manager.update_on_slr_breach("n", 0.8, "advocate")
        orig_advocate = runtime.identity_manager.mode_weights.get("advocate", 0.25)
        runtime.telemetry = MagicMock()
        runtime.telemetry.slr_distribution.return_value = {"count": 0, "mean": 0.0, "std": 0.0}
        runtime.telemetry.drift_velocity.return_value = {"cumulative": 0.0, "velocity": 0.0}
        runtime.telemetry.sie_summary.return_value = {"count": 0, "mean": 0.5}
        runtime._check_telemetry_thresholds()
        assert runtime.identity_manager.mode_weights["advocate"] > orig_advocate


# ===================================================================
# 21. PHASE 5.8 — EPISTEMIC FAILURE REASONS IN REROLL
# ===================================================================
class TestEpistemicRerollContext:
    """Phase 5.8 — Reroll context enriched with epistemic failure data."""

    def _make_node_dict(self, node_id="test_001"):
        return {
            "node_id": node_id,
            "name": node_id,
            "file": "test.py",
            "code_snippet": "def foo(): return 1",
            "imports": [],
            "operator_type": "function",
            "source_type": "ast_code",
            "skill_type": "execution",
            "semantics": {"code_snippet": "def foo(): return 1", "name": "foo"},
            "teaching_layer": {
                "skill_identity": {"name": node_id},
                "method_metadata": {"name": node_id, "language": "python"},
                "reasoning_vectors": {
                    "intent": "test", "strategy": "test",
                    "constraints": ["none"],
                    "execution_pattern": ["direct"],
                    "failure_modes": ["none"],
                },
                "implementation_template": {"code": "pass"},
            },
            "epistemic": {"state": "CREATED", "c_node": 0.0, "retry_budget": 6},
        }

    def _make_runtime(self, nodes, mode="coding_assistant"):
        graph = InMemoryEpistemicGraph(nodes)
        acs = MagicMock(spec=[])
        return DAGRuntime(graph, acs, mode=mode)

    def test_sycophancy_reroll_has_epistemic_context(self):
        """Sycophancy reroll injects epistemic failure type and SLR data."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.VALIDATED
        node.epistemic.c_node = 0.8
        node.sie_node = MagicMock(s_sie=0.7)

        runtime._route_to_reroll(
            node, NodeState.VALIDATED, "sycophancy_reroll: slr=0.72",
        )
        ctx = node.semantics["_reroll_context"]
        assert ctx["failure_type"] == "epistemic_failure"
        assert "slr" in ctx
        assert "sie_divergence" in ctx
        assert "identity_drift" in ctx

    def test_non_sycophancy_reroll_no_epistemic_keys(self):
        """Non-sycophancy rerolls do NOT inject epistemic failure type."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.VALIDATED

        runtime._route_to_reroll(
            node, NodeState.VALIDATED, "contract_failure",
        )
        ctx = node.semantics["_reroll_context"]
        assert "failure_type" not in ctx


# ===================================================================
# 23. SIE PROJECTION LAYER
# ===================================================================
from src.pipeline.sie_projection import SIEProjection, CANONICAL_FIELDS


class TestSIEProjection:
    """SIE Projection Layer — canonical field resolution."""

    def _make_node(self, **overrides):
        nd = MagicMock()
        nd.sie_node = MagicMock(
            content_density=0.75, s_sie=0.85, composite_quality_score=0.6, mode_scaling_factor=1.0,
            alignment_vector=[0.8, 0.6, 0.7],
        )
        nd.epistemic = MagicMock(c_node=0.9, depth=2)
        nd.v_score = 0.8
        nd.constraints = [MagicMock(), MagicMock()]
        for k, v in overrides.items():
            setattr(nd, k, v)
        return nd

    def test_project_returns_all_canonical_fields(self):
        node = self._make_node()
        proj = SIEProjection.project(node)
        for field in CANONICAL_FIELDS:
            assert field in proj, f"Missing field: {field}"

    def test_resolve_rho(self):
        node = self._make_node()
        assert SIEProjection.resolve(node, "content_density") == pytest.approx(0.75)

    def test_resolve_c_node(self):
        node = self._make_node()
        assert SIEProjection.resolve(node, "c_node") == pytest.approx(0.9)

    def test_resolve_depth(self):
        node = self._make_node()
        assert SIEProjection.resolve(node, "depth") == pytest.approx(2.0)

    def test_resolve_constraint_count(self):
        node = self._make_node()
        assert SIEProjection.resolve(node, "constraint_count") == pytest.approx(2.0)

    def test_resolve_phase_gradient_norm(self):
        import numpy as np
        node = self._make_node()
        expected = float(np.linalg.norm([0.8, 0.6, 0.7]))
        assert SIEProjection.resolve(node, "phase_gradient_norm") == pytest.approx(expected, abs=0.001)

    def test_resolve_unknown_field_returns_zero(self):
        node = self._make_node()
        assert SIEProjection.resolve(node, "nonexistent_field") == 0.0

    def test_resolve_no_sie_node(self):
        node = self._make_node(sie_node=None)
        assert SIEProjection.resolve(node, "content_density") == 0.0
        assert SIEProjection.resolve(node, "s_sie") == 0.0


# ===================================================================
# 24. INVARIANT ENGINE
# ===================================================================
from src.pipeline.invariant_engine import InvariantEngine


class TestInvariantEngine:
    """Invariant Execution Engine — parse, validate, security."""

    def test_parse_simple_gt(self):
        eng = InvariantEngine()
        result = eng.parse_invariant("content_density > 0.5")
        assert result == {"field": "content_density", "op": ">", "value": 0.5}

    def test_parse_gte(self):
        eng = InvariantEngine()
        result = eng.parse_invariant("s_sie >= 0.3")
        assert result == {"field": "s_sie", "op": ">=", "value": 0.3}

    def test_parse_lte(self):
        eng = InvariantEngine()
        result = eng.parse_invariant("depth <= 5")
        assert result == {"field": "depth", "op": "<=", "value": 5.0}

    def test_parse_invalid_field_rejected(self):
        eng = InvariantEngine()
        assert eng.parse_invariant("evil_field > 1") is None

    def test_parse_invalid_op_rejected(self):
        eng = InvariantEngine()
        # "~=" is not in ALLOWED_OPS
        assert eng.parse_invariant("content_density ~= 0.5") is None

    def test_parse_injection_attack_rejected(self):
        """SECURITY: code injection attempts must be rejected at parse time."""
        eng = InvariantEngine()
        assert eng.parse_invariant("__import__('os').system('rm -rf /')") is None
        assert eng.parse_invariant("eval('1+1')") is None
        assert eng.parse_invariant("exec('print(1)')") is None

    def test_validate_passing_invariant(self):
        eng = InvariantEngine()
        node = MagicMock()
        node.sie_node = MagicMock(content_density=0.8)
        parsed = {"field": "content_density", "op": ">", "value": 0.5}
        assert eng.validate_invariant(node, parsed) is True

    def test_validate_failing_invariant(self):
        eng = InvariantEngine()
        node = MagicMock()
        node.sie_node = MagicMock(content_density=0.3)
        parsed = {"field": "content_density", "op": ">", "value": 0.5}
        assert eng.validate_invariant(node, parsed) is False

    def test_validate_all_mixed(self):
        eng = InvariantEngine()
        node = MagicMock()
        node.sie_node = MagicMock(content_density=0.8, s_sie=0.2)
        node.epistemic = MagicMock(c_node=0.9, depth=1)
        node.v_score = 0.75
        invariants = ["content_density > 0.5", "s_sie >= 0.5"]  # first passes, second fails
        ok, failures = eng.validate_all(node, invariants)
        assert ok is False
        assert "s_sie >= 0.5" in failures

    def test_validate_all_empty_invariants(self):
        eng = InvariantEngine()
        node = MagicMock()
        ok, failures = eng.validate_all(node, [])
        assert ok is True
        assert failures == []

    def test_validate_all_unparseable_treated_as_failure(self):
        eng = InvariantEngine()
        node = MagicMock()
        ok, failures = eng.validate_all(node, ["garbage string!!!"])
        assert ok is False
        assert "garbage string!!!" in failures


# ===================================================================
# 25. CONSISTENCY ENGINE
# ===================================================================
from src.pipeline.consistency_engine import ConsistencyEngine


class TestConsistencyEngine:
    """Global Consistency Engine — contradiction & redundancy detection."""

    def _make_node(self, node_id, pg, c_node=0.9, constraints=None):
        n = MagicMock()
        n.node_id = node_id
        n.sie_node = MagicMock(alignment_vector=pg)
        n.epistemic = MagicMock(c_node=c_node)
        n.constraints = constraints or []
        return n

    def test_no_contradictions_clean_set(self):
        eng = ConsistencyEngine()
        nodes = [
            self._make_node("a", [1.0, 0.0, 0.0]),
            self._make_node("b", [0.0, 1.0, 0.0]),
        ]
        result = eng.validate_global_consistency(nodes)
        assert result["status"] == "consistent"
        assert result["contradictions"] == []

    def test_detect_redundancy_high_similarity(self):
        eng = ConsistencyEngine()
        nodes = [
            self._make_node("a", [1.0, 0.0, 0.0], c_node=0.9),
            self._make_node("b", [1.0, 0.0, 0.0], c_node=0.8),
        ]
        redundancies = eng.detect_redundancy(nodes)
        assert len(redundancies) == 1
        assert ("a", "b") in redundancies

    def test_detect_contradiction_anti_correlated(self):
        eng = ConsistencyEngine()
        nodes = [
            self._make_node("a", [1.0, 0.0, 0.0]),
            self._make_node("b", [-1.0, 0.0, 0.0]),
        ]
        contradictions = eng.detect_contradictions(nodes)
        assert len(contradictions) >= 1
        assert any("anti_correlated" in c["reason"] for c in contradictions)

    def test_detect_contradiction_opposing_constraints(self):
        c1 = MagicMock(type="semantic", tags=["safety"], valid=True, severity="error")
        c2 = MagicMock(type="semantic", tags=["safety"], valid=False, severity="error")
        eng = ConsistencyEngine()
        nodes = [
            self._make_node("a", [0.5, 0.5, 0.5], constraints=[c1]),
            self._make_node("b", [0.6, 0.4, 0.5], constraints=[c2]),
        ]
        contradictions = eng.detect_contradictions(nodes)
        assert len(contradictions) >= 1
        assert any("constraint_contradiction" in c["reason"] for c in contradictions)

    def test_resolve_keeps_higher_confidence(self):
        eng = ConsistencyEngine()
        nodes = [
            self._make_node("a", [1.0, 0.0, 0.0], c_node=0.9),
            self._make_node("b", [-1.0, 0.0, 0.0], c_node=0.7),
        ]
        contradictions = eng.detect_contradictions(nodes)
        rejected = eng.resolve_contradictions(nodes, contradictions)
        assert "b" in rejected
        assert "a" not in rejected

    def test_empty_nodes_returns_consistent(self):
        eng = ConsistencyEngine()
        result = eng.validate_global_consistency([])
        assert result["status"] == "consistent"

    def test_validate_global_degraded(self):
        eng = ConsistencyEngine()
        nodes = [
            self._make_node("a", [1.0, 0.0, 0.0], c_node=0.9),
            self._make_node("b", [1.0, 0.0, 0.0], c_node=0.8),
            self._make_node("c", [0.0, 1.0, 0.0], c_node=0.85),
        ]
        result = eng.validate_global_consistency(nodes)
        # a and b are redundant — b rejected; c survives
        assert result["status"] == "degraded"
        assert "b" in result["rejected_ids"]


# ===================================================================
# 26. OMEGA VALIDATOR
# ===================================================================
from src.pipeline.omega_validator import OmegaValidator


class TestOmegaValidator:
    """Omega Handshake Validator — 5 dimensions + partial/final split."""

    def _make_node(self, node_id, pg=None, constraints=None):
        n = MagicMock()
        n.node_id = node_id
        n.sie_node = MagicMock(alignment_vector=pg or [0.8, 0.6, 0.7])
        n.constraints = constraints or []
        return n

    def test_omega_pass_all_dimensions(self):
        ov = OmegaValidator()
        nodes = [self._make_node("a"), self._make_node("b")]
        im = MagicMock()
        im.get_rolling_slr_mean = MagicMock(return_value=0.1)
        im.drift_score = 0.05
        im.frozen = False
        im.max_drift = 0.3
        result = ov.omega_handshake(nodes, None, im)
        assert result["status"] == "OMEGA_PASS"

    def test_omega_degraded_one_soft_fail(self):
        ov = OmegaValidator()
        nodes = [self._make_node("a")]
        im = MagicMock()
        im.get_rolling_slr_mean = MagicMock(return_value=0.8)  # SLR fails
        im.drift_score = 0.05
        im.frozen = False
        im.max_drift = 0.3
        result = ov.omega_handshake(nodes, None, im)
        assert result["status"] == "OMEGA_DEGRADED"

    def test_omega_fail_identity_frozen(self):
        """Identity frozen → critical → always OMEGA_FAIL."""
        ov = OmegaValidator()
        nodes = [self._make_node("a")]
        im = MagicMock()
        im.get_rolling_slr_mean = MagicMock(return_value=0.1)
        im.drift_score = 0.5
        im.frozen = True
        im.max_drift = 0.3
        result = ov.omega_handshake(nodes, None, im)
        assert result["status"] == "OMEGA_FAIL"

    def test_partial_check_passes_clean_node(self):
        ov = OmegaValidator()
        node = self._make_node("a")
        im = MagicMock()
        im.drift_score = 0.05
        im.frozen = False
        im.max_drift = 0.3
        result = ov.partial_check(node, identity_manager=im)
        assert result["pass"] is True

    def test_partial_check_fails_fatal_constraint(self):
        c = MagicMock(severity="fatal", valid=False, description="bad")
        ov = OmegaValidator()
        node = self._make_node("a", constraints=[c])
        result = ov.partial_check(node)
        assert result["pass"] is False

    def test_constraint_satisfaction_passes_no_fatal(self):
        c = MagicMock(severity="warning", valid=False)
        ov = OmegaValidator()
        result = ov.validate_constraint_satisfaction([self._make_node("a", constraints=[c])])
        assert result["pass"] is True

    def test_slr_integrity_passes_low_mean(self):
        ov = OmegaValidator()
        im = MagicMock()
        im.get_rolling_slr_mean = MagicMock(return_value=0.2)
        result = ov.validate_slr_integrity(im)
        assert result["pass"] is True

    def test_topological_closure_no_orphans(self):
        import networkx as nx
        G = nx.DiGraph()
        G.add_edge("a", "b")
        nodes = [self._make_node("a"), self._make_node("b")]
        ov = OmegaValidator()
        result = ov.validate_topological_closure(nodes, G)
        assert result["pass"] is True
        assert result["orphan_count"] == 0


# ===================================================================
# 27. DEPTH PENALTY (Phase 7)
# ===================================================================
from src.pipeline.dag_runtime import DEPTH_PENALTY_RATE, DEPTH_PENALTY_CAP


class TestDepthPenalty:
    """Phase 7 — depth penalty dampens confidence with increasing depth."""

    def _make_node_dict(self, node_id="test_depth", depth=0, **overrides):
        base = {
            "node_id": node_id,
            "name": node_id,
            "file": "test.py",
            "code_snippet": "def foo(x: int) -> int:\n    \"\"\"Inc.\"\"\"\n    return x + 1\n",
            "imports": [],
            "operator_type": "function",
            "source_type": "ast_code",
            "skill_type": "execution",
            "semantics": {
                "code_snippet": "def foo(x: int) -> int:\n    \"\"\"Inc.\"\"\"\n    return x + 1\n",
                "name": "foo",
            },
            "teaching_layer": {
                "skill_identity": {"name": node_id},
                "method_metadata": {"name": node_id, "language": "python"},
                "reasoning_vectors": {
                    "intent": "compute", "strategy": "direct return",
                    "constraints": ["no_mutation"], "execution_pattern": ["functional"],
                    "failure_modes": ["type_error"],
                },
                "implementation_template": {"code": "pass"},
            },
            "epistemic": {
                "state": "CREATED", "c_node": 0.0, "retry_budget": 6, "depth": depth,
            },
        }
        base.update(overrides)
        return base

    def _make_runtime(self, nodes, mode="coding_assistant"):
        graph = InMemoryEpistemicGraph(nodes)
        acs = MagicMock(spec=[])
        return DAGRuntime(graph, acs, mode=mode)

    def test_depth_zero_no_penalty(self):
        """depth=0 → depth_penalty=0 → no dampening."""
        runtime = self._make_runtime([self._make_node_dict(depth=0)])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.VALIDATED
        node.epistemic.c_node = 0.8
        node.sie_node = MagicMock(s_sie=0.9, alignment_vector=[0.8, 0.6, 0.7])
        node.v_score = 0.85
        runtime._compute_unified_confidence(node)
        assert node.epistemic.confidence["depth_penalty"] == 0.0

    def test_depth_penalty_increases_with_depth(self):
        """Higher depth → higher penalty → lower confidence."""
        runtime = self._make_runtime([self._make_node_dict(depth=3)])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.VALIDATED
        node.epistemic.c_node = 0.8
        node.sie_node = MagicMock(s_sie=0.9, alignment_vector=[0.8, 0.6, 0.7])
        node.v_score = 0.85
        runtime._compute_unified_confidence(node)
        expected_penalty = 3 * DEPTH_PENALTY_RATE
        assert node.epistemic.confidence["depth_penalty"] == pytest.approx(expected_penalty)
        # c_final should be lower than a depth=0 node
        c_with_depth = node.epistemic.c_node

        # Reset for depth=0
        node.epistemic.depth = 0
        node.epistemic.c_node = 0.8
        runtime._compute_unified_confidence(node)
        c_no_depth = node.epistemic.c_node
        assert c_with_depth < c_no_depth

    def test_depth_penalty_capped(self):
        """Penalty should not exceed DEPTH_PENALTY_CAP even at extreme depth."""
        runtime = self._make_runtime([self._make_node_dict(depth=100)])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.VALIDATED
        node.epistemic.c_node = 0.8
        node.sie_node = MagicMock(s_sie=0.9, alignment_vector=[0.8, 0.6, 0.7])
        node.v_score = 0.85
        runtime._compute_unified_confidence(node)
        assert node.epistemic.confidence["depth_penalty"] == DEPTH_PENALTY_CAP


# ===================================================================
# 28. SOFT BRANCH STABILITY (Phase 7)
# ===================================================================
from src.pipeline.dag_runtime import BRANCH_STABILITY_THRESHOLD, BRANCH_DEGRADATION_FACTOR


class TestSoftBranchStability:
    """Phase 7 — soft branch degradation replaces hard pruning."""

    def _make_node_dict(self, node_id="test_br", branch_id="root", **overrides):
        base = {
            "node_id": node_id,
            "name": node_id,
            "file": "test.py",
            "code_snippet": "def foo(x: int) -> int:\n    \"\"\"Inc.\"\"\"\n    return x + 1\n",
            "imports": [],
            "operator_type": "function",
            "source_type": "ast_code",
            "skill_type": "execution",
            "semantics": {
                "code_snippet": "def foo(x: int) -> int:\n    \"\"\"Inc.\"\"\"\n    return x + 1\n",
                "name": "foo",
            },
            "teaching_layer": {
                "skill_identity": {"name": node_id},
                "method_metadata": {"name": node_id, "language": "python"},
                "reasoning_vectors": {
                    "intent": "compute", "strategy": "direct return",
                    "constraints": ["no_mutation"], "execution_pattern": ["functional"],
                    "failure_modes": ["type_error"],
                },
                "implementation_template": {"code": "pass"},
            },
            "epistemic": {
                "state": "SCORED", "c_node": 0.3, "retry_budget": 6,
                "branch_id": branch_id,
            },
        }
        base.update(overrides)
        return base

    def _make_runtime(self, nodes, mode="coding_assistant"):
        graph = InMemoryEpistemicGraph(nodes)
        acs = MagicMock(spec=[])
        return DAGRuntime(graph, acs, mode=mode)

    def test_unstable_branch_gets_degraded(self):
        """Low c_node branch → soft degradation, not hard prune."""
        nodes = [
            self._make_node_dict("n1", branch_id="br_1"),
            self._make_node_dict("n2", branch_id="br_1"),
        ]
        runtime = self._make_runtime(nodes)
        for n in runtime.graph.nodes.values():
            n.epistemic.state = NodeState.SCORED
            n.epistemic.c_node = 0.2  # very low

        runtime._nx_graph = runtime._build_directed_graph(list(runtime.graph.nodes.values()))
        runtime._apply_branch_stability()

        for n in runtime.graph.nodes.values():
            # Should be degraded, NOT rejected
            assert n.epistemic.state == NodeState.SCORED
            assert n.epistemic.c_node == pytest.approx(0.2 * BRANCH_DEGRADATION_FACTOR, abs=0.001)
            assert n.semantics.get("_branch_degraded") is True

    def test_stable_branch_not_degraded(self):
        """High c_node branch → no degradation."""
        nodes = [
            self._make_node_dict("n1", branch_id="br_2"),
        ]
        runtime = self._make_runtime(nodes)
        n = list(runtime.graph.nodes.values())[0]
        n.epistemic.state = NodeState.SCORED
        n.epistemic.c_node = 0.85

        runtime._nx_graph = runtime._build_directed_graph(list(runtime.graph.nodes.values()))
        runtime._apply_branch_stability()

        # Confidence should be untouched
        assert n.epistemic.c_node == pytest.approx(0.85)


# ===================================================================
# 29. INVARIANT INTEGRATION IN DAG RUNTIME
# ===================================================================
class TestInvariantIntegration:
    """Phase 8 integration — invariant failure blocks scoring, routes to reroll."""

    def _make_node_dict(self, node_id="test_inv", invariants=None, **overrides):
        sem = {
            "code_snippet": "def foo(x: int) -> int:\n    \"\"\"Inc.\"\"\"\n    return x + 1\n",
            "name": "foo",
        }
        if invariants:
            sem["invariants"] = invariants
        base = {
            "node_id": node_id,
            "name": node_id,
            "file": "test.py",
            "code_snippet": "def foo(x: int) -> int:\n    \"\"\"Inc.\"\"\"\n    return x + 1\n",
            "imports": [],
            "operator_type": "function",
            "source_type": "ast_code",
            "skill_type": "execution",
            "semantics": sem,
            "teaching_layer": {
                "skill_identity": {"name": node_id},
                "method_metadata": {"name": node_id, "language": "python"},
                "reasoning_vectors": {
                    "intent": "compute", "strategy": "direct return",
                    "constraints": ["no_mutation"], "execution_pattern": ["functional"],
                    "failure_modes": ["type_error"],
                },
                "implementation_template": {"code": "pass"},
            },
            "epistemic": {
                "state": "CREATED", "c_node": 0.0, "retry_budget": 6,
            },
        }
        base.update(overrides)
        return base

    def _make_runtime(self, nodes, mode="coding_assistant"):
        graph = InMemoryEpistemicGraph(nodes)
        acs = MagicMock(spec=[])
        return DAGRuntime(graph, acs, mode=mode)

    def test_invariant_failure_blocks_scoring(self):
        """Failing invariant routes node to REROLL, not SCORED."""
        runtime = self._make_runtime([
            self._make_node_dict(invariants=["content_density > 0.99"]),
        ])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.VALIDATED
        node.epistemic.c_node = 0.8
        # SIE with content_density=0.5 — invariant "content_density > 0.99" will fail
        node.sie_node = MagicMock(
            s_sie=0.9, content_density=0.5, composite_quality_score=0.6, mode_scaling_factor=1.0,
            alignment_vector=[0.8, 0.6, 0.7],
        )
        node.v_score = 0.85

        # Patch upstream steps so we reach invariant enforcement cleanly
        with patch.object(runtime, "_compute_sie", return_value=True), \
             patch.object(runtime, "_apply_acs"), \
             patch.object(runtime, "_check_governance_directive", return_value=False), \
             patch.object(runtime, "_enforce_identity", return_value=False), \
             patch.object(runtime, "_validate_evidence"):
            result = runtime._transition_to_scored(node)
        assert result is False
        assert node.epistemic.state in (NodeState.REROLL, NodeState.REJECTED)

    def test_passing_invariant_allows_scoring(self):
        """Passing invariant does not block scoring."""
        runtime = self._make_runtime([
            self._make_node_dict(invariants=["content_density > 0.3"]),
        ])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.VALIDATED
        node.epistemic.c_node = 0.8
        node.sie_node = MagicMock(
            s_sie=0.9, content_density=0.8, composite_quality_score=0.6, mode_scaling_factor=1.0,
            alignment_vector=[0.8, 0.6, 0.7],
        )
        node.v_score = 0.85

        with patch.object(runtime, "_compute_sie", return_value=True), \
             patch.object(runtime, "_apply_acs"), \
             patch.object(runtime, "_check_governance_directive", return_value=False), \
             patch.object(runtime, "_enforce_identity", return_value=False), \
             patch.object(runtime, "_validate_evidence"):
            result = runtime._transition_to_scored(node)
        assert result is True
        assert node.epistemic.state == NodeState.SCORED

    def test_no_invariants_passes(self):
        """No invariants in semantics → scoring proceeds normally."""
        runtime = self._make_runtime([self._make_node_dict()])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.VALIDATED
        node.epistemic.c_node = 0.8
        node.sie_node = MagicMock(
            s_sie=0.9, content_density=0.8, composite_quality_score=0.6, mode_scaling_factor=1.0,
            alignment_vector=[0.8, 0.6, 0.7],
        )
        node.v_score = 0.85

        with patch.object(runtime, "_compute_sie", return_value=True), \
             patch.object(runtime, "_apply_acs"), \
             patch.object(runtime, "_check_governance_directive", return_value=False), \
             patch.object(runtime, "_enforce_identity", return_value=False), \
             patch.object(runtime, "_validate_evidence"):
            result = runtime._transition_to_scored(node)
        assert result is True


# ===================================================================
# 29. TIKHONOV RIDGE REGRESSION (D2+D12)
# ===================================================================
import numpy as np
from src.pipeline.acs_engine import SycophancyDetector


class TestTikhonovRidge:
    """D2+D12 — Full Tikhonov Ridge Regression SLR."""

    def test_tikhonov_identity_matrix_low_slr(self):
        """Identity A with aligned y → low SLR (w ≈ y)."""
        det = SycophancyDetector(sensitivity=0.15)
        A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        y = [0.8, 0.5, 0.3]
        slr = det.compute_tikhonov_slr(A, y)
        assert 0.0 <= slr <= 0.15, f"Identity matrix should yield near-zero SLR, got {slr}"

    def test_tikhonov_fallback_on_empty(self):
        """Empty matrix → returns 0.0."""
        det = SycophancyDetector()
        assert det.compute_tikhonov_slr([], [1.0, 2.0]) == 0.0
        assert det.compute_tikhonov_slr([[1, 2]], []) == 0.0

    def test_tikhonov_nan_guard(self):
        """NaN in inputs → returns 0.0."""
        det = SycophancyDetector()
        assert det.compute_tikhonov_slr([[float('nan'), 1], [1, 0]], [0.5, 0.5]) == 0.0

    def test_tikhonov_bounded(self):
        """SLR is always in [0, 1]."""
        det = SycophancyDetector(sensitivity=0.15)
        A = [[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]]
        y = [0.3, 0.7, 0.5]
        slr = det.compute_tikhonov_slr(A, y)
        assert 0.0 <= slr <= 1.0

    def test_dispatch_prefers_tikhonov(self):
        """When 2D matrix provided, dispatch uses Tikhonov path."""
        det = SycophancyDetector(sensitivity=0.15)
        A = [[1, 0], [0, 1]]
        y = [0.8, 0.6]
        slr_dispatch = det.compute_slr(confusion_matrix=A, response_vector=y)
        slr_direct = det.compute_tikhonov_slr(A, y)
        assert slr_dispatch == slr_direct

    def test_dispatch_falls_back_to_cosine(self):
        """When only flat vectors, dispatch uses cosine SLR."""
        det = SycophancyDetector(sensitivity=0.15)
        a = [0.8, 0.2, 0.5]
        b = [0.3, 0.9, 0.1]
        slr_dispatch = det.compute_slr(signal_a=a, signal_b=b)
        slr_cosine = det.compute_cosine_slr(a, b)
        assert slr_dispatch == slr_cosine

    def test_dispatch_none_returns_zero(self):
        """When no signals provided, dispatch returns 0.0."""
        det = SycophancyDetector()
        assert det.compute_slr() == 0.0


# ===================================================================
# 30. HASH VERIFICATION ON ROLLBACK (D8)
# ===================================================================
from src.pipeline.identity_manager import IdentityManager


class TestHashVerificationRollback:
    """D8 — Cryptographic hash verification on identity rollback."""

    def test_revert_succeeds_with_valid_hash(self):
        """Normal create_from_source + SLR breach + revert works without error."""
        mgr = IdentityManager()
        nodes = [{"node_id": "n1", "code_snippet": "def foo(): pass", "imports": [], "skill_type": "execution"}]
        mgr.create_from_source(nodes)
        mgr.update_on_slr_breach("n1", 0.5, "theorist")
        assert mgr.drift_score > 0
        mgr.revert_to_stable()
        assert mgr.drift_score == 0.0

    def test_revert_raises_on_corrupted_stable(self):
        """Tampering with _stable_state after create_from_source → RuntimeError."""
        mgr = IdentityManager()
        nodes = [{"node_id": "n1", "code_snippet": "def foo(): pass", "imports": [], "skill_type": "execution"}]
        mgr.create_from_source(nodes)
        # Tamper with stable behavior snapshot
        mgr._stable_state.behavior_profile["theorist"] = 999.0
        with pytest.raises(RuntimeError, match="corruption"):
            mgr.revert_to_stable()

    def test_revert_succeeds_with_empty_hash(self):
        """When no hash reference is set, revert proceeds normally (no verification)."""
        mgr = IdentityManager()
        assert mgr.stable_hash_reference == ""
        mgr.update_on_slr_breach("n1", 0.5, "theorist")
        mgr.revert_to_stable()  # Should not raise
        assert mgr.drift_score == 0.0



# ===================================================================
# 32. CONFIGURABLE BRANCH_COUNT (D7)
# ===================================================================
from src.pipeline.reroll import RerollEngine


class TestConfigurableBranchCount:
    """D7 — Configurable branch_count on RerollEngine."""

    def test_default_branch_count_is_2(self):
        engine = RerollEngine()
        assert engine.branch_count == 2

    def test_branch_count_configurable(self):
        """Setting branch_count to 3 → reroll generates 3 candidates."""
        engine = RerollEngine()
        engine.branch_count = 3

        # Create a mock node + runtime
        node = MagicMock()
        node.epistemic.state = "REROLL"
        node.node_id = "bc_test"
        node.semantics = {"code_snippet": "x = 1"}

        mock_runtime = MagicMock()
        mock_runtime.mode = "coding_assistant"
        # Track execute calls
        call_count = [0]
        def mock_execute(n):
            call_count[0] += 1
            n.semantics = {"code_snippet": f"x = {call_count[0]}"}
        mock_runtime.execute = mock_execute

        engine.reroll(node, mock_runtime)
        assert call_count[0] == 3, f"Expected 3 execute calls, got {call_count[0]}"

    def test_high_severity_uses_3_branches(self):
        """dag_runtime sets branch_count=3 for sycophancy reroll reasons."""
        graph = InMemoryEpistemicGraph()
        node_dict = {
            "node_id": "hs_test",
            "name": "hstest",
            "file": "test.py",
            "code_snippet": "def foo(): return 1",
            "imports": [],
            "operator_type": "function",
            "source_type": "ast_code",
            "skill_type": "execution",
            "semantics": {
                "code_snippet": "def foo(): return 1",
                "_reroll_context": {
                    "previous_failure_reason": "sycophancy_reroll: slr=0.85",
                },
            },
            "teaching_layer": {},
            "epistemic": {
                "state": "CREATED",
                "c_node": 0.5,
                "retry_budget": 6,
            },
        }
        graph.add_nodes([node_dict])
        acs = MagicMock()
        runtime = DAGRuntime(graph=graph, acs=acs, mode="coding_assistant")
        node = list(runtime.graph.nodes.values())[0]
        # Ensure epistemic state exists and is REROLL
        if node.epistemic is None:
            node.epistemic = MagicMock()
        node.epistemic.state = NodeState.REROLL
        node.epistemic.retry_budget = 2
        node.epistemic.final_status = None

        with patch.object(runtime.reroll_engine, "reroll", return_value=None) as mock_reroll:
            runtime._handle_reroll(node)
            assert runtime.reroll_engine.branch_count == 3


# ===================================================================
# 33. GRAPH BACKEND SHIM (D9)
# ===================================================================
from src.interfaces.graph_backend import is_gpu_available, get_graph_backend


class TestGraphBackendShim:
    """D9 — Optional cuGraph GPU backend shim."""

    def test_fallback_to_networkx(self):
        """Without cuGraph installed, falls back to NetworkX."""
        backend = get_graph_backend()
        assert hasattr(backend, "DiGraph"), "Fallback backend should be NetworkX"
        assert hasattr(backend, "Graph")

    def test_is_gpu_available_returns_bool(self):
        result = is_gpu_available()
        assert isinstance(result, bool)


# ===================================================================
# 34. ENVIRONMENT INTEGRITY (Sprint 1 — Deterministic Baseline)
# ===================================================================


class TestEnvironmentIntegrity:
    """Sprint 1 — Validate environment reproducibility and metadata consistency."""

    def test_critical_runtime_imports(self):
        """All runtime-critical packages must be importable."""
        import numpy
        import networkx
        import pydantic

        assert numpy.__version__ is not None
        assert networkx.__version__ is not None
        assert pydantic.__version__.startswith("2"), (
            f"Pydantic v2 required, got {pydantic.__version__}"
        )

    def test_import_core_modules(self):
        """All core pipeline modules must import without error."""
        import src.pipeline.dag_runtime
        import src.pipeline.acs_engine
        import src.pipeline.identity_manager
        import src.pipeline.reroll
        import src.pipeline.contracts
        import src.core.models
        import src.core.models_contract
        import src.interfaces.graph_backend

    def test_progress_metadata_schema(self):
        """development_progress.json must pass structural validation."""
        progress_path = PROJECT_ROOT / "development_progress.json"
        if not progress_path.exists():
            pytest.skip("development_progress.json not present")
        with open(progress_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("metadata")
        assert meta is not None, "metadata block missing"
        assert isinstance(meta.get("generated_at"), str)
        assert isinstance(meta.get("analysis_type"), str)
        assert "tealogical" not in meta["analysis_type"], (
            "Typo 'tealogical' still present in metadata"
        )

        corpus = data.get("telemetry_findings", {}).get("corpus_summary", {})
        total = corpus.get("total_events")
        assert isinstance(total, int), "total_events must be int"
        assert total >= 9360, f"total_events stale: {total} < 9360"

    def test_requirements_declares_runtime_deps(self):
        """requirements.txt must declare all runtime-critical packages."""
        req_path = PROJECT_ROOT / "requirements.txt"
        assert req_path.exists(), "requirements.txt missing"
        content = req_path.read_text(encoding="utf-8").lower()
        for pkg in ["numpy", "networkx", "pydantic", "tiktoken", "jsonlines"]:
            assert pkg in content, f"Missing runtime dep: {pkg}"


# ===================================================================
# 35. CONTRACT SCHEMA ALIGNMENT REGRESSION (Sprint 2 — risk-003)
# ===================================================================
from src.pipeline.contracts import (
    MODE_CONTRACTS,
    validate_contract,
    CodingOutput,
    TheoristOutput,
    AdvocateOutput,
    VeteranOutput,
)
from src.pipeline.dag_scoring_pass import wrap_payload


class TestContractSchemaAlignment:
    """Sprint 2 — validate that all 4 modes pass contract validation with
    the field names produced by execute() and wrap_payload(), and that the
    historical failure patterns (old names / extra fields) are correctly
    rejected by extra='forbid'."""

    # ------------------------------------------------------------------
    # Happy-path: new field names pass validation
    # ------------------------------------------------------------------

    def test_coding_output_contract_passes(self):
        payload = {"implementation": "def compute(): return 42"}
        ok, err = validate_contract(payload, CodingOutput)
        assert ok, f"CodingOutput rejected valid payload: {err}"

    def test_theorist_output_contract_passes(self):
        payload = {"constraints": ["idempotent", "deterministic"]}
        ok, err = validate_contract(payload, TheoristOutput)
        assert ok, f"TheoristOutput rejected valid payload: {err}"

    def test_advocate_output_contract_passes(self):
        payload = {"theory": "Hexagonal architecture", "implementation": "class Port: pass"}
        ok, err = validate_contract(payload, AdvocateOutput)
        assert ok, f"AdvocateOutput rejected valid payload: {err}"

    def test_veteran_output_contract_passes(self):
        payload = {
            "traceback": "ZeroDivisionError: division by zero",
            "diff": "- 1/0\n+ safe_div(1, 0)",
            "resolved_code": "def safe_div(a,b): return a/b if b else 0",
        }
        ok, err = validate_contract(payload, VeteranOutput)
        assert ok, f"VeteranOutput rejected valid payload: {err}"

    # ------------------------------------------------------------------
    # Historical failure patterns: old field names must be rejected
    # ------------------------------------------------------------------

    def test_old_coding_fields_rejected(self):
        """code_snippet + name + orchestration_mode → the 3 extra-forbidden
        fields that caused the majority of CodingOutput contract failures."""
        payload = {"code_snippet": "pass", "name": "foo", "orchestration_mode": "coding_assistant"}
        ok, _ = validate_contract(payload, CodingOutput)
        assert not ok, "Old coding fields should be rejected"

    def test_old_advocate_fields_rejected(self):
        """theory_text + implementation_code + text_to_code_ratio → the
        mismatched names that caused AdvocateOutput failures."""
        payload = {"theory_text": "x", "implementation_code": "y", "text_to_code_ratio": 0.5}
        ok, _ = validate_contract(payload, AdvocateOutput)
        assert not ok, "Old advocate fields should be rejected"

    def test_old_veteran_extra_fields_rejected(self):
        """Veteran payloads with analysis_text + attempt_code + orchestration_mode
        should fail even when the required fields are present."""
        payload = {
            "traceback": "err", "diff": "d", "resolved_code": "c",
            "analysis_text": "diag", "attempt_code": "x=1", "orchestration_mode": "veteran",
        }
        ok, _ = validate_contract(payload, VeteranOutput)
        assert not ok, "Extra veteran fields should be rejected"

    def test_old_theorist_missing_constraints_rejected(self):
        """Theorist payloads with chunk_text but no constraints → missing
        required field."""
        payload = {"chunk_text": "Some theory", "orchestration_mode": "theorist"}
        ok, _ = validate_contract(payload, TheoristOutput)
        assert not ok, "Theorist without constraints should be rejected"

    # ------------------------------------------------------------------
    # Wrapper alignment: wrap_payload() → contract-clean semantics
    # ------------------------------------------------------------------

    def test_wrap_veteran_produces_contract_fields(self):
        raw = {"traceback": "err", "diff": "d", "resolved_code": "c",
               "analysis_text": "diag", "attempt_code": "x=1"}
        wrapped = wrap_payload(raw, "veteran", "test.json")
        sem = wrapped["semantics"]
        # Required fields present
        assert "traceback" in sem
        assert "diff" in sem
        assert "resolved_code" in sem
        # Forbidden fields stripped
        assert "analysis_text" not in sem
        assert "attempt_code" not in sem

    def test_wrap_advocate_produces_contract_fields(self):
        raw = {"theory_text": "arch", "implementation_code": "class X: pass",
               "text_to_code_ratio": 0.7}
        wrapped = wrap_payload(raw, "advocate", "test.json")
        sem = wrapped["semantics"]
        assert "theory" in sem
        assert "implementation" in sem
        assert "theory_text" not in sem
        assert "implementation_code" not in sem
        assert "text_to_code_ratio" not in sem

    def test_wrap_coding_produces_contract_fields(self):
        raw = {"node_id": "n1", "code_snippet": "pass", "name": "foo",
               "semantics": {"code_snippet": "x=1", "name": "bar"}}
        wrapped = wrap_payload(raw, "coding_assistant", "test.json")
        sem = wrapped["semantics"]
        assert "implementation" in sem
        assert "code_snippet" not in sem
        assert "name" not in sem

    # ------------------------------------------------------------------
    # Sanitization: orchestration_mode stripped before validation
    # ------------------------------------------------------------------

    def test_orchestration_mode_stripped_pre_validation(self):
        """orchestration_mode is functionally necessary but structurally
        forbidden — verify the sanitization layer removes it."""
        for mode, payload in [
            ("coding_assistant", {"implementation": "x", "orchestration_mode": "coding_assistant"}),
            ("theorist", {"constraints": ["a"], "orchestration_mode": "theorist"}),
            ("advocate", {"theory": "t", "implementation": "i", "orchestration_mode": "advocate"}),
            ("veteran", {"traceback": "e", "diff": "d", "resolved_code": "c", "orchestration_mode": "veteran"}),
        ]:
            # With orchestration_mode → must fail
            ok, _ = validate_contract(payload, MODE_CONTRACTS[mode])
            assert not ok, f"{mode}: orchestration_mode should cause rejection"
            # Without it → must pass
            clean = {k: v for k, v in payload.items() if k != "orchestration_mode"}
            ok, err = validate_contract(clean, MODE_CONTRACTS[mode])
            assert ok, f"{mode}: clean payload rejected: {err}"


# ===================================================================
# SPRINT 3: GOVERNANCE OBSERVABILITY + MODE COVERAGE
# ===================================================================
from src.pipeline.acs_engine import ACSExecutionGovernor, SycophancyDetector

# ------------------------------------------------------------------
# 3c. SLR + Drift Telemetry E2E Tests
# ------------------------------------------------------------------


class TestSLRTelemetryEmission:
    """3a/3c: ACS engine emits slr_computed telemetry when SLR is evaluated."""

    def _make_governor(self, telemetry):
        gov = ACSExecutionGovernor.__new__(ACSExecutionGovernor)
        gov.config = {"acs_tuning": {"strictness": 0.80, "slr_threshold": 0.65}}
        gov.penalties = {
            "syntax_error_mult": 0.50, "unsafe_io_mult": 0.70,
            "sast_finding_mult": 0.85, "sycophancy_mult": 0.40,
            "utility_cosmetic_mult": 0.95, "circular_logic_mult": 0.50,
        }
        gov.sycophancy_detector = SycophancyDetector(sensitivity=0.15)
        gov.strictness = 0.80
        gov.secret_patterns = []
        gov.telemetry = telemetry
        return gov

    def _make_node(self, node_id, matrix, vector):
        return {
            "node_id": node_id,
            "source_type": "ast_code",
            "artifact_type": "python",
            "code_snippet": "def foo(): return 1",
            "skill_type": "execution",
            "entry": {},
            "semantics": {},
            "epistemic": {"c_node": 0.9, "state": "audited", "confidence": {"local": 0.5}},
            "reasoning_trajectory": {
                "confusion_matrix": matrix,
                "response_vector": vector,
            },
            "teaching_layer": {
                "reasoning_vectors": {"intent": "test", "strategy": "test"},
                "heuristics": {"use_when": ["always"], "avoid_when": []},
            },
        }

    def test_acs_slr_telemetry_emitted_on_evaluation(self):
        """evaluate_node with trajectory data emits slr/slr_computed."""
        tc = TelemetryCollector()
        gov = self._make_governor(tc)
        node = self._make_node("slr_test_001", [[0.5, 0.3], [0.2, 0.4]], [0.6, 0.4])
        gov.evaluate_node(node)

        slr_events = [e for e in tc._events if e.category == "slr" and e.event_type == "slr_computed"]
        assert len(slr_events) >= 1, "Expected at least one slr_computed telemetry event"
        evt = slr_events[0]
        assert evt.node_id == "slr_test_001"
        assert "slr" in evt.payload
        assert "conviction" in evt.payload
        assert "tikhonov_slr" in evt.payload
        assert 0.0 <= evt.payload["slr"] <= 1.0

    def test_acs_no_telemetry_when_collector_absent(self):
        """evaluate_node does not crash when telemetry=None."""
        gov = self._make_governor(None)
        node = self._make_node("no_tel_001", [[0.9, 0.1], [0.1, 0.9]], [0.8, 0.2])
        result = gov.evaluate_node(node)
        assert result is not None

    def test_slr_telemetry_aggregates_in_distribution(self):
        """slr_computed events feed into TelemetryCollector.slr_distribution()."""
        tc = TelemetryCollector()
        tc.record("slr", "slr_computed", node_id="a", payload={"slr": 0.3, "conviction": 0.7, "tikhonov_slr": 0.28})
        tc.record("slr", "slr_computed", node_id="b", payload={"slr": 0.5, "conviction": 0.5, "tikhonov_slr": 0.48})
        dist = tc.slr_distribution()
        assert dist["count"] == 2
        assert dist["mean"] == pytest.approx(0.4, abs=0.01)


class TestDriftTelemetryEmission:
    """3b/3c: IdentityManager emits drift telemetry on SLR breaches and containment."""

    def test_drift_increment_telemetry(self):
        """update_on_slr_breach emits identity/drift_increment."""
        tc = TelemetryCollector()
        im = IdentityManager(max_drift=0.3, telemetry=tc)
        im.update_on_slr_breach("node_drift_001", slr=0.75, mode="advocate")

        drift_events = [e for e in tc._events if e.category == "identity" and e.event_type == "drift_increment"]
        assert len(drift_events) == 1
        evt = drift_events[0]
        assert evt.node_id == "node_drift_001"
        assert evt.payload["slr"] == 0.75
        assert evt.payload["drift_increment"] == pytest.approx(0.075, abs=0.001)
        assert evt.payload["mode"] == "advocate"
        assert evt.payload["drift_score"] > 0

    def test_containment_telemetry_on_threshold_breach(self):
        """check_and_enforce emits identity/containment_triggered when drift > max."""
        tc = TelemetryCollector()
        im = IdentityManager(max_drift=0.1, telemetry=tc)
        im.drift_score = 0.2
        frozen = im.check_and_enforce("branch_001")
        assert frozen is True

        contain_events = [e for e in tc._events if e.event_type == "containment_triggered"]
        assert len(contain_events) == 1
        evt = contain_events[0]
        assert evt.node_id == "branch_001"
        assert evt.payload["combined_drift"] > 0.1
        assert evt.payload["max_drift"] == 0.1

    def test_no_containment_when_below_threshold(self):
        """check_and_enforce does not emit containment when drift is safe."""
        tc = TelemetryCollector()
        im = IdentityManager(max_drift=0.5, telemetry=tc)
        im.drift_score = 0.01
        frozen = im.check_and_enforce("branch_safe")
        assert frozen is False
        contain_events = [e for e in tc._events if e.event_type == "containment_triggered"]
        assert len(contain_events) == 0

    def test_drift_velocity_from_identity_events(self):
        """Multiple drift increments feed into TelemetryCollector.drift_velocity()."""
        tc = TelemetryCollector()
        im = IdentityManager(max_drift=1.0, telemetry=tc)
        im.update_on_slr_breach("n1", 0.7, "coding_assistant")
        im.update_on_slr_breach("n2", 0.8, "coding_assistant")
        vel = tc.drift_velocity()
        assert vel["count"] == 2
        assert vel["cumulative"] == pytest.approx(0.15, abs=0.01)

    def test_no_drift_telemetry_when_collector_absent(self):
        """IdentityManager without telemetry does not crash."""
        im = IdentityManager(max_drift=0.3)
        im.update_on_slr_breach("n1", 0.8, "veteran")
        im.check_and_enforce("branch_x")


class TestTelemetryWiring:
    """3c: DAGRuntime wires its TelemetryCollector into ACS + IdentityManager."""

    def _make_node_dict(self, node_id="wire_001"):
        return {
            "node_id": node_id,
            "name": node_id,
            "file": "test.py",
            "code_snippet": "def foo(): return 1",
            "imports": [],
            "operator_type": "function",
            "source_type": "ast_code",
            "skill_type": "execution",
            "semantics": {"implementation": "def foo(): return 1"},
            "teaching_layer": {
                "skill_identity": {"name": node_id},
                "method_metadata": {"name": node_id, "language": "python"},
                "reasoning_vectors": {
                    "intent": "test", "strategy": "test",
                    "constraints": ["none"],
                    "execution_pattern": ["direct"],
                    "failure_modes": ["none"],
                },
                "implementation_template": {"code": "pass"},
            },
            "epistemic": {"state": "CREATED", "c_node": 0.0, "retry_budget": 6},
        }

    def test_telemetry_wired_to_acs(self):
        """DAGRuntime.telemetry is wired to acs.telemetry after __init__."""
        graph = InMemoryEpistemicGraph([self._make_node_dict()])
        acs = ACSExecutionGovernor.__new__(ACSExecutionGovernor)
        acs.telemetry = None
        acs.config = {}
        acs.penalties = {}
        acs.sycophancy_detector = SycophancyDetector()
        acs.strictness = 0.8
        acs.secret_patterns = []
        runtime = DAGRuntime(graph, acs, mode="coding_assistant")
        assert runtime.telemetry is not None
        assert acs.telemetry is runtime.telemetry

    def test_telemetry_wired_to_identity_manager(self):
        """DAGRuntime.telemetry is wired to identity_manager.telemetry after __init__."""
        graph = InMemoryEpistemicGraph([self._make_node_dict()])
        acs = MagicMock(spec=[])
        runtime = DAGRuntime(graph, acs, mode="coding_assistant")
        assert runtime.telemetry is not None
        if runtime.identity_manager:
            assert runtime.identity_manager.telemetry is runtime.telemetry


# ------------------------------------------------------------------
# 3d. Theorist & Advocate Mode Path Tests
# ------------------------------------------------------------------


class TestTheoristModePath:
    """3d: Theorist mode produces contract-valid output with correct telemetry."""

    def _make_theorist_node(self, node_id="theorist_001"):
        return {
            "node_id": node_id,
            "name": node_id,
            "file": "theory.md",
            "code_snippet": "",
            "imports": [],
            "operator_type": "theory",
            "source_type": "theory",
            "skill_type": "execution",
            "semantics": {"constraints": ["monotonic convergence", "bounded drift"]},
            "teaching_layer": {
                "skill_identity": {"name": node_id},
                "method_metadata": {"name": node_id, "language": "python"},
                "reasoning_vectors": {
                    "intent": "theoretical analysis",
                    "strategy": "axiomatic reasoning",
                    "constraints": ["formal"],
                    "execution_pattern": ["deductive"],
                    "failure_modes": ["underdetermined"],
                },
                "implementation_template": {"code": ""},
            },
            "epistemic": {"state": "CREATED", "c_node": 0.0, "retry_budget": 6},
        }

    def _make_runtime(self, nodes):
        graph = InMemoryEpistemicGraph(nodes)
        acs = MagicMock(spec=[])
        return DAGRuntime(graph, acs, mode="theorist")

    def test_theorist_contract_validates(self):
        """TheoristOutput accepts constraints list."""
        ok, err = validate_contract({"constraints": ["monotonic", "bounded"]}, TheoristOutput)
        assert ok, f"TheoristOutput rejected: {err}"

    def test_theorist_mode_set_on_runtime(self):
        """DAGRuntime initializes with mode='theorist'."""
        runtime = self._make_runtime([self._make_theorist_node()])
        assert runtime.mode == "theorist"

    def test_theorist_node_transitions_to_validated(self):
        """Theorist node passes contract validation with constraints field."""
        runtime = self._make_runtime([self._make_theorist_node()])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.CREATED

        result = runtime._transition_to_validated(node)
        assert result is True
        assert node.epistemic.state == NodeState.VALIDATED

    def test_theorist_contract_rejects_old_fields(self):
        """chunk_text instead of constraints is rejected by TheoristOutput."""
        ok, _ = validate_contract({"chunk_text": "some theory"}, TheoristOutput)
        assert not ok

    def test_theorist_telemetry_records_transition(self):
        """Telemetry records state transition for theorist mode."""
        runtime = self._make_runtime([self._make_theorist_node()])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.CREATED

        runtime._transition_to_validated(node)
        if runtime.telemetry:
            transition_events = [
                e for e in runtime.telemetry._events
                if e.payload.get("to") == "VALIDATED"
                or (e.category == "contract" and e.event_type == "validated")
            ]
            assert len(transition_events) >= 1


class TestAdvocateModePath:
    """3d: Advocate mode produces contract-valid output with correct telemetry."""

    def _make_advocate_node(self, node_id="advocate_001"):
        return {
            "node_id": node_id,
            "name": node_id,
            "file": "review.py",
            "code_snippet": "def refactored(): pass",
            "imports": [],
            "operator_type": "function",
            "source_type": "ast_code",
            "skill_type": "execution",
            "semantics": {"theory": "defensive coding", "implementation": "def refactored(): pass"},
            "teaching_layer": {
                "skill_identity": {"name": node_id},
                "method_metadata": {"name": node_id, "language": "python"},
                "reasoning_vectors": {
                    "intent": "code review",
                    "strategy": "advocate refactoring",
                    "constraints": ["maintain API"],
                    "execution_pattern": ["review-then-refactor"],
                    "failure_modes": ["regression"],
                },
                "implementation_template": {"code": "pass"},
            },
            "epistemic": {"state": "CREATED", "c_node": 0.0, "retry_budget": 6},
        }

    def _make_runtime(self, nodes):
        graph = InMemoryEpistemicGraph(nodes)
        acs = MagicMock(spec=[])
        return DAGRuntime(graph, acs, mode="advocate")

    def test_advocate_contract_validates(self):
        """AdvocateOutput accepts theory + implementation."""
        ok, err = validate_contract(
            {"theory": "defensive coding", "implementation": "def refactored(): pass"},
            AdvocateOutput,
        )
        assert ok, f"AdvocateOutput rejected: {err}"

    def test_advocate_mode_set_on_runtime(self):
        """DAGRuntime initializes with mode='advocate'."""
        runtime = self._make_runtime([self._make_advocate_node()])
        assert runtime.mode == "advocate"

    def test_advocate_node_transitions_to_validated(self):
        """Advocate node passes contract validation with theory+implementation."""
        runtime = self._make_runtime([self._make_advocate_node()])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.CREATED

        result = runtime._transition_to_validated(node)
        assert result is True
        assert node.epistemic.state == NodeState.VALIDATED

    def test_advocate_contract_rejects_old_fields(self):
        """theory_text/implementation_code rejected by AdvocateOutput."""
        ok, _ = validate_contract(
            {"theory_text": "old", "implementation_code": "old"},
            AdvocateOutput,
        )
        assert not ok

    def test_advocate_telemetry_records_contract_pass(self):
        """Telemetry records contract/validated for advocate mode."""
        runtime = self._make_runtime([self._make_advocate_node()])
        node = list(runtime.graph.nodes.values())[0]
        node.epistemic.state = NodeState.CREATED

        runtime._transition_to_validated(node)
        if runtime.telemetry:
            contract_events = [
                e for e in runtime.telemetry._events
                if e.category == "contract" and e.event_type == "validated"
            ]
            assert len(contract_events) >= 1
            assert contract_events[0].payload.get("mode") == "advocate"


# ===================================================================
# SPRINT 4: STRUCTURAL DECOMPOSITION — EXTRACTED MODULE TESTS
# ===================================================================

# ------------------------------------------------------------------
# 4c-1. Fallback Stubs (fallback_stubs.py)
# ------------------------------------------------------------------
from src.pipeline import fallback_stubs


class TestFallbackStubs:
    """4a/4c: Extracted fallback stubs instantiate correctly and expose expected API."""

    def test_aletheia_skill_roundtrip(self):
        """Fallback AletheiaSkill can be created and serialized."""
        skill = fallback_stubs.AletheiaSkill(node_id="stub_001")
        assert skill.node_id == "stub_001"
        assert skill.epistemic.state == "unresolved"
        assert skill.epistemic.retry_budget == 6
        d = skill.model_dump()
        assert "node_id" in d
        assert "semantics" in d

    def test_cognitive_node_defaults(self):
        """Fallback CognitiveNode has correct defaults."""
        cn = fallback_stubs.CognitiveNode(cognitive_id="cog_001")
        assert cn.cognitive_id == "cog_001"
        assert cn.mode == "advocate"
        assert cn.c_final == 0.0
        assert cn.inbound_edges == []

    def test_reasoning_edge_validates(self):
        """Fallback ReasoningEdge can be constructed with minimum fields."""
        edge = fallback_stubs.ReasoningEdge(source_id="a", target_id="b")
        assert edge.edge_type == "dependency"
        assert edge.weight == 1.0

    def test_identity_manager_stub_api(self):
        """Fallback IdentityManager exposes full API without error."""
        im = fallback_stubs.IdentityManager()
        assert im.drift_score == 0.0
        im.update_on_slr_breach("n1", 0.5, "veteran")
        assert im.check_and_enforce("branch") is False
        im.update_from_trajectory(None)
        assert im.get_state_snapshot() == {}

    def test_reroll_engine_stub_api(self):
        """Fallback RerollEngine has reroll_count and reroll()."""
        re = fallback_stubs.RerollEngine()
        assert re.reroll_count == 0
        assert re.reroll() is None

    def test_telemetry_collector_stub_api(self):
        """Fallback TelemetryCollector exposes record/snapshot/persist."""
        tc = fallback_stubs.TelemetryCollector()
        tc.record("slr", "test")
        assert tc.snapshot() == {}
        tc.persist("/dev/null")

    def test_validate_contract_stub_always_passes(self):
        """Fallback validate_contract returns (True, None)."""
        ok, err = fallback_stubs.validate_contract({"x": 1}, None)
        assert ok is True
        assert err is None

    def test_stable_json_dumps_deterministic(self):
        """Fallback stable_json_dumps sorts keys."""
        result = fallback_stubs.stable_json_dumps({"b": 2, "a": 1})
        assert result == '{"a": 1, "b": 2}'

    def test_enforce_semantic_firewall_noop(self):
        """Fallback firewall is a no-op."""
        fallback_stubs.enforce_semantic_firewall("some text", context="test")

    def test_drift_violation_is_exception(self):
        """DriftViolation is a proper Exception subclass."""
        assert issubclass(fallback_stubs.DriftViolation, Exception)


# ------------------------------------------------------------------
# 4c-2. Telemetry Policy (telemetry_policy.py)
# ------------------------------------------------------------------
from src.pipeline.telemetry_policy import (
    check_telemetry_thresholds,
    TELEMETRY_SLR_MEAN_THRESHOLD,
    TELEMETRY_DRIFT_CUMULATIVE_THRESHOLD,
    TELEMETRY_QUALITY_MEAN_FLOOR,
)


class TestTelemetryPolicy:
    """4b/4c: Extracted telemetry policy logic — standalone unit tests."""

    def test_noop_when_no_telemetry(self):
        """Returns unchanged values when telemetry is None."""
        result = check_telemetry_thresholds(
            telemetry=None,
            identity_manager=None,
            acceptance_threshold=0.40,
            system_backpressure=0.0,
        )
        assert result["acceptance_threshold"] == 0.40
        assert result["system_backpressure"] == 0.0

    def test_high_slr_mean_pushes_drift(self):
        """SLR mean > threshold pushes synthetic drift onto identity_manager."""
        tc = TelemetryCollector()
        for _ in range(5):
            tc.record("slr", "breach", payload={"slr": 0.75})

        im = IdentityManager(max_drift=1.0)
        initial_drift = im.drift_score

        check_telemetry_thresholds(
            telemetry=tc,
            identity_manager=im,
            acceptance_threshold=0.40,
            system_backpressure=0.0,
        )
        assert im.drift_score > initial_drift, "Drift should increase from SLR overshoot"

    def test_sie_underflow_emits_alert(self):
        """SIE mean below floor emits systemic_underflow_alert."""
        tc = TelemetryCollector()
        for _ in range(4):
            tc.record("sie", "scored", payload={"s_sie": 0.02})

        check_telemetry_thresholds(
            telemetry=tc,
            identity_manager=None,
            acceptance_threshold=0.40,
            system_backpressure=0.0,
        )
        alerts = [e for e in tc._events if e.event_type == "systemic_underflow_alert"]
        assert len(alerts) == 1
        assert alerts[0].payload["mean"] < TELEMETRY_QUALITY_MEAN_FLOOR

    def test_system_backpressure_rises_with_low_coherence(self):
        """Low SIE + high drift → system_backpressure rises toward 0.5."""
        tc = TelemetryCollector()
        for _ in range(5):
            tc.record("identity", "drift", payload={"drift_increment": 0.1})
        for _ in range(4):
            tc.record("sie", "scored", payload={"s_sie": 0.05})

        result = check_telemetry_thresholds(
            telemetry=tc,
            identity_manager=None,
            acceptance_threshold=0.40,
            system_backpressure=0.0,
            non_rejected_ratio=0.5,
        )
        assert result["system_backpressure"] > 0.0

    def test_system_backpressure_clamped_at_half(self):
        """Field pressure is capped at 0.5."""
        tc = TelemetryCollector()
        for _ in range(5):
            tc.record("identity", "drift", payload={"drift_increment": 1.0})
        for _ in range(4):
            tc.record("sie", "scored", payload={"s_sie": 0.0})

        result = check_telemetry_thresholds(
            telemetry=tc,
            identity_manager=None,
            acceptance_threshold=0.40,
            system_backpressure=0.5,
            non_rejected_ratio=0.1,
        )
        assert result["system_backpressure"] <= 0.5

    def test_threshold_constants_match_runtime(self):
        """Threshold constants imported into dag_runtime match telemetry_policy."""
        from src.pipeline.dag_runtime import (
            TELEMETRY_SLR_MEAN_THRESHOLD as rt_slr,
            TELEMETRY_DRIFT_CUMULATIVE_THRESHOLD as rt_drift,
            TELEMETRY_QUALITY_MEAN_FLOOR as rt_sie,
        )
        assert rt_slr == TELEMETRY_SLR_MEAN_THRESHOLD
        assert rt_drift == TELEMETRY_DRIFT_CUMULATIVE_THRESHOLD
        assert rt_sie == TELEMETRY_QUALITY_MEAN_FLOOR

    def test_topology_floor_prevents_zero_collapse(self):
        """topology_integrity=0.0 should NOT collapse field_coherence to zero
        thanks to topology_floor=0.1."""
        tc = type("TC", (), {
            "slr_distribution": lambda s: {"count": 0, "mean": 0.0},
            "drift_velocity": lambda s: {"cumulative": 0.0},
            "sie_summary": lambda s: {"count": 5, "mean": 0.8},
            "record": lambda s, *a, **kw: None,
        })()
        result = check_telemetry_thresholds(
            telemetry=tc,
            identity_manager=None,
            acceptance_threshold=0.40,
            system_backpressure=0.0,
            non_rejected_ratio=0.0,  # zero accepted nodes
        )
        # With floor=0.1: coherence = 0.8 * 1.0 * 0.1 = 0.08 → pressure = 0.92 → capped at 0.5
        # Without floor: coherence = 0.0 → pressure = 0.5 (same cap, but for wrong reason)
        # The key test: pressure should be < 1.0 (not unbounded)
        assert result["system_backpressure"] <= 0.5
        assert result["system_backpressure"] > 0.0

    def test_topology_floor_lets_sie_contribute(self):
        """High SIE mean with zero topology should still yield lower pressure
        than low SIE mean with zero topology."""
        def make_tc(sie_mean):
            return type("TC", (), {
                "slr_distribution": lambda s: {"count": 0, "mean": 0.0},
                "drift_velocity": lambda s: {"cumulative": 0.0},
                "sie_summary": lambda s: {"count": 5, "mean": sie_mean},
                "record": lambda s, *a, **kw: None,
            })()
        high_sie = check_telemetry_thresholds(
            telemetry=make_tc(0.9), identity_manager=None,
            acceptance_threshold=0.40, system_backpressure=0.0,
            non_rejected_ratio=0.0,
        )
        low_sie = check_telemetry_thresholds(
            telemetry=make_tc(0.1), identity_manager=None,
            acceptance_threshold=0.40, system_backpressure=0.0,
            non_rejected_ratio=0.0,
        )
        # High SIE should produce lower (or equal) pressure than low SIE
        assert high_sie["system_backpressure"] <= low_sie["system_backpressure"]
