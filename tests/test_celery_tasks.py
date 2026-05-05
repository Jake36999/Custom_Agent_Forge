"""
Sprint 5e — Celery task unit tests.

Run with:  python -m pytest tests/test_celery_tasks.py -v

All tasks execute eagerly (in-process, no broker) so these tests
validate the full scoring pipeline without Docker or Redis.
"""

import os
import json
import pytest
from pathlib import Path
from typing import Any, Dict

# Force eager execution before any Celery import
os.environ.setdefault("CELERY_BROKER_URL", "memory://")
os.environ.setdefault("CELERY_RESULT_BACKEND", "cache+memory://")

from src.celery_app import app as celery_app
celery_app.conf.update(
    task_always_eager=True,
    task_eager_propagates=True,
)

from src.tasks.mode_tasks import (
    run_advocate,
    run_coding,
    run_theorist,
    run_veteran,
)
from src.tasks.pipeline_workflow import (
    _detect_mode,
    collect_and_format,
    UnrecognizedModeError,
)


# ── Fixture: minimal payloads per mode ─────────────────────────

def _make_coding_payload(node_id: str = "code_001") -> Dict[str, Any]:
    return {
        "node_id": node_id,
        "name": f"test_coding_{node_id}",
        "file": "test.py",
        "code_snippet": "def hello(): return 42",
        "imports": [],
        "operator_type": "function",
        "source_type": "ast_code",
        "skill_type": "execution",
        "teaching_layer": {
            "skill_identity": {"name": f"test_coding_{node_id}"},
            "method_metadata": {"name": node_id, "language": "python"},
            "reasoning_vectors": {
                "intent": "Test intent",
                "strategy": "Test strategy",
                "constraints": [],
                "execution_pattern": ["step1"],
                "failure_modes": [],
            },
            "implementation_template": {"code": "def hello(): return 42"},
        },
        "semantics": {
            "implementation": "def hello(): return 42",
            "orchestration_mode": "coding_assistant",
        },
        "epistemic": {"state": "CREATED", "c_node": 0.5},
    }


def _make_theorist_payload(node_id: str = "theo_001") -> Dict[str, Any]:
    return {
        "node_id": node_id,
        "name": f"test_theorist_{node_id}",
        "file": "theory.py",
        "code_snippet": "",
        "imports": [],
        "operator_type": "theory",
        "source_type": "theoretical_reasoning",
        "skill_type": "execution",
        "teaching_layer": {
            "skill_identity": {"name": f"test_theorist_{node_id}"},
            "method_metadata": {"name": node_id, "language": "python"},
            "reasoning_vectors": {
                "intent": "Formal specification",
                "strategy": "Constraint extraction",
                "constraints": ["type_safe"],
                "execution_pattern": ["axiom", "derive"],
                "failure_modes": [],
            },
            "implementation_template": {"code": ""},
        },
        "semantics": {
            "constraints": ["type_safe", "bounded"],
            "orchestration_mode": "theorist",
        },
        "epistemic": {"state": "CREATED", "c_node": 0.5},
    }


def _make_advocate_payload(node_id: str = "adv_001") -> Dict[str, Any]:
    return {
        "node_id": node_id,
        "name": f"test_advocate_{node_id}",
        "file": "advocate.py",
        "code_snippet": "class Solver: pass",
        "imports": [],
        "operator_type": "advocate_theory",
        "source_type": "advocate_theory",
        "skill_type": "execution",
        "teaching_layer": {
            "skill_identity": {"name": f"test_advocate_{node_id}"},
            "method_metadata": {"name": node_id, "language": "python"},
            "reasoning_vectors": {
                "intent": "Architectural review",
                "strategy": "Dialectical",
                "constraints": [],
                "execution_pattern": ["theory", "implement"],
                "failure_modes": [],
            },
            "implementation_template": {"code": "class Solver: pass"},
        },
        "semantics": {
            "theory": "SOLID principles applied to solver architecture",
            "implementation": "class Solver: pass",
            "orchestration_mode": "advocate",
        },
        "epistemic": {"state": "CREATED", "c_node": 0.5},
    }


def _make_veteran_payload(node_id: str = "vet_001") -> Dict[str, Any]:
    return {
        "node_id": node_id,
        "name": f"test_veteran_{node_id}",
        "file": "veteran.py",
        "code_snippet": "fixed_code = True",
        "imports": [],
        "operator_type": "veteran_diagnostic",
        "source_type": "veteran_diagnostic",
        "skill_type": "execution",
        "teaching_layer": {
            "skill_identity": {"name": f"test_veteran_{node_id}"},
            "method_metadata": {"name": node_id, "language": "python"},
            "reasoning_vectors": {
                "intent": "Diagnose runtime failure",
                "strategy": "Trace analysis",
                "constraints": [],
                "execution_pattern": ["error", "fix"],
                "failure_modes": [],
            },
            "implementation_template": {"code": "fixed_code = True"},
        },
        "semantics": {
            "traceback": "Traceback (most recent call last): ...",
            "diff": "- old\n+ new",
            "resolved_code": "fixed_code = True",
            "orchestration_mode": "veteran",
        },
        "epistemic": {"state": "CREATED", "c_node": 0.5},
    }


# ═══════════════════════════════════════════════════════════════
# MODE TASK TESTS
# ═══════════════════════════════════════════════════════════════

class TestCodingTask:
    def test_coding_returns_scored_result(self):
        result = run_coding([_make_coding_payload()], "TEST-CODING-001")
        assert result["mode"] == "coding_assistant"
        assert result["run_id"] == "TEST-CODING-001"
        assert len(result["scored_nodes"]) == 1
        assert result["accepted"] + result["rejected"] == 1

    def test_coding_multiple_payloads(self):
        payloads = [_make_coding_payload(f"c_{i}") for i in range(3)]
        result = run_coding(payloads, "TEST-CODING-MULTI")
        assert len(result["scored_nodes"]) == 3


class TestTheoristTask:
    def test_theorist_returns_scored_result(self):
        result = run_theorist([_make_theorist_payload()], "TEST-THEO-001")
        assert result["mode"] == "theorist"
        assert len(result["scored_nodes"]) == 1

    def test_theorist_telemetry_present(self):
        result = run_theorist([_make_theorist_payload()], "TEST-THEO-TEL")
        assert isinstance(result.get("telemetry"), dict)


class TestAdvocateTask:
    def test_advocate_returns_scored_result(self):
        result = run_advocate([_make_advocate_payload()], "TEST-ADV-001")
        assert result["mode"] == "advocate"
        assert len(result["scored_nodes"]) == 1


class TestVeteranTask:
    def test_veteran_returns_scored_result(self):
        result = run_veteran([_make_veteran_payload()], "TEST-VET-001")
        assert result["mode"] == "veteran"
        assert len(result["scored_nodes"]) == 1


# ═══════════════════════════════════════════════════════════════
# PIPELINE WORKFLOW TESTS
# ═══════════════════════════════════════════════════════════════

class TestModeDetection:
    def test_detect_veteran_from_traceback(self):
        assert _detect_mode({"traceback": "..."}) == "veteran"

    def test_detect_advocate_from_theory_and_impl(self):
        assert _detect_mode({"theory_text": "...", "implementation_code": "..."}) == "advocate"

    def test_detect_coding_default(self):
        assert _detect_mode({"code_snippet": "x = 1"}) == "coding_assistant"

    def test_detect_from_semantics_mode(self):
        assert _detect_mode({"semantics": {"orchestration_mode": "theorist"}}) == "theorist"

    def test_unrecognized_payload_raises(self):
        """HD4: Unrecognized payloads must hard-stop, not silently fall back."""
        import pytest
        with pytest.raises(UnrecognizedModeError):
            _detect_mode({"unknown_field": "some value"})


class TestCollectAndFormat:
    def test_collect_merges_mode_results(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        # Patch OUTPUT_DIR for this test
        import src.tasks.pipeline_workflow as pw
        orig_out = pw.OUTPUT_DIR
        pw.OUTPUT_DIR = tmp_path / "output"
        pw.OUTPUT_DIR.mkdir()

        mode_results = [
            {"scored_nodes": [{"node_id": "a"}], "accepted": 1, "rejected": 0, "telemetry": {}},
            {"scored_nodes": [{"node_id": "b"}, {"node_id": "c"}], "accepted": 1, "rejected": 1, "telemetry": {}},
        ]

        result = collect_and_format(mode_results, project_name="test_proj", run_id="TEST-001")
        assert result["total_nodes"] == 3
        assert result["accepted"] == 2
        assert result["rejected"] == 1
        assert result["status"] == "completed"

        # Verify scored file was written
        scored_file = Path(result["scored_file"])
        assert scored_file.exists()
        with open(scored_file) as f:
            data = json.load(f)
        assert len(data) == 3

        pw.OUTPUT_DIR = orig_out


# ═══════════════════════════════════════════════════════════════
# API TESTS (FastAPI TestClient)
# ═══════════════════════════════════════════════════════════════

class TestAPI:
    @pytest.fixture(autouse=True)
    def _client(self):
        try:
            from fastapi.testclient import TestClient
            from src.api import app as fastapi_app
            self.client = TestClient(fastapi_app)
            self.available = True
        except ImportError:
            self.available = False

    def test_health_endpoint(self):
        if not self.available:
            pytest.skip("fastapi not installed")
        resp = self.client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "redis" in body
        assert "worker_count" in body

    def test_ingest_requires_repo_url(self):
        if not self.available:
            pytest.skip("fastapi not installed")
        resp = self.client.post("/ingest", json={})
        assert resp.status_code == 422  # Validation error

    def test_score_rejects_empty_payloads(self):
        if not self.available:
            pytest.skip("fastapi not installed")
        resp = self.client.post("/score", json={"payloads": [], "mode": "coding_assistant"})
        assert resp.status_code == 422  # min_length=1
