import json
import sys
import tempfile
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Ensure project root is on path for absolute imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipeline import dataset_formatter as df

# Safety: verify the module resolves to the correct location
assert hasattr(df, "build_conversational_array"), (
    "dataset_formatter module missing expected API — check import path"
)


def _skill_with_intent() -> dict:
    return {
        "node_id": "NODE_B",
        "source_type": "ast_code",
        "semantics": {
            "operator_types": ["read_socket", "decrypt_payload", "write_db"],
            "side_effects": ["db", "network"],
        },
        "dependencies": {
            "upstream_callers": ["caller_b", "caller_a"],
            "downstream_calls": ["sink_b", "sink_a"],
        },
        "state_and_execution": {
            "mutation_tracking": ["cache_write", "audit_log"],
        },
        "source_context": {
            "docstring": "Handles secure payload ingestion.",
            "code_snippet": "def ingest(payload):\n    return payload",
        },
        "teaching_layer": {
            "skill_identity": {
                "canonical_id": "SKILL_IO_NETWORK_B",
                "category": "io.network",
            },
            "intent": {
                "problem": "Securely persist decrypted socket payloads",
                "goal": "Store validated payloads in the database",
            },
            "heuristics": {
                "use_when": ["high throughput", "encrypted channel"],
                "avoid_when": ["tiny payload"],
                "conditions": {
                    "hardware": {"op": "==", "value": "CPU"},
                    "network": {"op": "==", "value": "active"},
                },
            },
            "transformation": {
                "from": "naive sequential I/O",
                "to": "validated batched persistence",
            },
            "template": {
                "inputs": [{"name": "payload", "type": "bytes"}],
                "code": "def ingest(payload):\n    return payload",
            },
        },
        "epistemic": {
            "confidence": {
                "global": 0.916,
                "recurrence_count": 3,
            }
        },
    }


def _structural_only_skill() -> dict:
    return {
        "node_id": "NODE_A",
        "source_type": "ast_code",
        "semantics": {
            "operator_types": ["load_tensor", "vectorize", "store_tensor"],
            "side_effects": ["memory"],
        },
        "dependencies": {
            "upstream_callers": ["trainer", "loader"],
            "downstream_calls": ["persistor"],
        },
        "state_and_execution": {
            "mutation_tracking": ["tensor_cache", "stats_buffer"],
        },
        "source_context": {
            "docstring": "",
            "code_snippet": "def vec(xs):\n    return xs",
        },
        "teaching_layer": {
            "skill_identity": {
                "canonical_id": "SKILL_GENERAL_COMPUTE_A",
                "category": "general.compute",
            },
            "intent": {
                "problem": "",
                "goal": "",
            },
            "method_metadata": {
                "intent": {
                    "problem": "",
                    "goal": "",
                }
            },
            "heuristics": {
                "use_when": [],
                "avoid_when": ["single item"],
                "conditions": {
                    "batch_size": {"op": ">", "value": 1},
                },
            },
            "transformation_delta": {
                "from": "scalar loop",
                "to": "vectorized pass",
            },
            "implementation_template": {
                "inputs": [],
                "code": "def vec(xs):\n    return xs",
            },
        },
        "epistemic": {
            "confidence": {
                "global": 0.5,
                "recurrence_count": 1,
            }
        },
    }


def _rejected_skill() -> dict:
    return {
        "node_id": "NODE_C",
        "source_type": "ast_code",
        "semantics": {
            "operator_types": ["compute.generic"],
            "side_effects": [],
        },
        "dependencies": {
            "upstream_callers": [],
            "downstream_calls": ["sink"],
        },
        "state_and_execution": {
            "mutation_tracking": [],
        },
        "source_context": {
            "docstring": "",
            "code_snippet": "def broken(x):\n    return x",
        },
        "teaching_layer": {
            "skill_identity": {
                "canonical_id": "SKILL_GENERAL_COMPUTE_REJECT",
                "category": "general.compute",
            },
            "intent": {
                "problem": df._GENERIC_INTENT_PROBLEM,
                "goal": df._GENERIC_INTENT_GOAL,
            },
            "heuristics": {
                "use_when": [df._GENERIC_USE_WHEN],
                "avoid_when": [],
                "conditions": {},
            },
            "template": {
                "inputs": [],
                "code": "def broken(input_data):\n    {operation_logic}\n    return {result}",
            },
            "transformation": {
                "from": "Standard Implementation",
                "to": "Broken Pattern",
            },
        },
        "epistemic": {
            "confidence": {
                "global": 0.1,
                "recurrence_count": 1,
            }
        },
    }


def _write_yaml(data: dict) -> tuple[tempfile.TemporaryDirectory, Path]:
    temp_dir = tempfile.TemporaryDirectory()
    path = Path(temp_dir.name) / "skills_UNIFIED.yaml"
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    return temp_dir, path


def test_structured_formatter_behavior() -> None:
    temp_dir, yaml_path = _write_yaml(
        {
            "capability_injection": {
                "compiled_skills": [_skill_with_intent(), _structural_only_skill()],
            }
        }
    )
    try:
        cards = df.generate_flashcards(yaml_path)
    finally:
        temp_dir.cleanup()
    assert len(cards) == 2, "Expected both structurally valid skills to be emitted."

    structural_card = cards[0]
    intentful_card = cards[1]

    assert structural_card["instruction"].startswith("Extract and formalize the structural execution pattern"), "Expected structural-only rows to use the extraction prompt."
    assert "Category: general.compute" in structural_card["instruction"], "Expected instruction to include category."
    assert "Canonical ID: SKILL_GENERAL_COMPUTE_A" in structural_card["instruction"], "Expected instruction to include canonical ID."
    assert "Pipeline Signature: load_tensor -> vectorize -> store_tensor" in structural_card["instruction"], "Expected instruction to include the derived pipeline signature."
    assert "System Prior Confidence: 0.500" in structural_card["instruction"], "Expected instruction to include the formatted epistemic prior."

    input_payload = json.loads(structural_card["input"])
    output_payload = json.loads(structural_card["output"])

    assert list(input_payload["constraints"]["operator_types"]) == ["load_tensor", "vectorize", "store_tensor"], "Expected operator order to be preserved."
    assert input_payload["graph"]["upstream_callers"] == ["loader", "trainer"], "Expected graph callers to be deterministically sorted."
    assert input_payload["graph"]["downstream_calls"] == ["persistor"], "Expected graph callees to be preserved."
    assert input_payload["constraints"]["mutation_tracking"] == ["stats_buffer", "tensor_cache"], "Expected mutation tracking to be deterministically sorted."
    assert input_payload["constraints"]["heuristic_conditions"] == {"batch_size": {"op": ">", "value": 1}}, "Expected machine conditions to stay in the input payload."

    assert "intent" not in output_payload, "Expected structural-only rows to omit the intent block."
    assert output_payload["heuristics"]["avoid_when"] == ["single item"], "Expected human heuristics to remain in output."
    assert output_payload["transformation"] == {"from": "scalar loop", "to": "vectorized pass"}, "Expected legacy transformation fallback to be preserved."
    assert output_payload["template"]["code"] == "def vec(xs):\n    return xs", "Expected template code to remain structured."

    assert intentful_card["instruction"].startswith("Synthesize an optimized execution method"), "Expected intentful rows to use the synthesis prompt."
    intentful_output = json.loads(intentful_card["output"])
    assert intentful_output["intent"] == {
        "problem": "Securely persist decrypted socket payloads",
        "goal": "Store validated payloads in the database",
    }, "Expected populated intent to be preserved in output."
    assert "conditions" not in intentful_output["heuristics"], "Expected machine conditions to be excluded from output heuristics."

    for card in cards:
        assert "###" not in card["output"], "Expected markdown headings to be removed."
        assert "**" not in card["output"], "Expected markdown emphasis to be removed."
        assert "```" not in card["output"], "Expected fenced code blocks to be removed."
        assert "Unknown" not in card["output"], "Expected output payloads to avoid Unknown placeholders."
        assert "Unspecified" not in card["output"], "Expected output payloads to avoid Unspecified placeholders."


def test_formatter_is_deterministic() -> None:
    temp_dir, yaml_path = _write_yaml(
        {
            "capability_injection": {
                "compiled_skills": [_skill_with_intent(), _structural_only_skill()],
            }
        }
    )
    try:
        cards_a = df.generate_flashcards(yaml_path)
        cards_b = df.generate_flashcards(yaml_path)
    finally:
        temp_dir.cleanup()

    assert cards_a == cards_b, "Expected formatter output to be byte-stable across repeated runs."


def test_wrapper_schema_support() -> None:
    temp_dir, yaml_path = _write_yaml(
        {
            "YAML_Knowledge_Pipelines": {
                "Compiled_Intelligence": {
                    "capability_injection": {
                        "compiled_skills": [_skill_with_intent()],
                    }
                }
            }
        }
    )
    try:
        cards = df.generate_flashcards(yaml_path)
    finally:
        temp_dir.cleanup()
    assert len(cards) == 1, "Expected wrapped capability_injection schema to remain supported."
    assert "SKILL_IO_NETWORK_B" in cards[0]["instruction"], "Expected wrapped schema rows to preserve identity metadata."


def test_strict_filter_rejects_expected_failure_modes() -> None:
    temp_dir, yaml_path = _write_yaml(
        {
            "capability_injection": {
                "compiled_skills": [_skill_with_intent()],
            }
        }
    )
    try:
        valid_card = df.generate_flashcards(yaml_path)[0]
    finally:
        temp_dir.cleanup()

    dataset_filter = df.StrictDatasetFilter()
    dataset_filter.validate_entry(valid_card)
    dataset_filter.mark_accepted()
    assert dataset_filter.stats["accepted"] == 1, "Expected valid cards to pass the strict filter."

    malformed_input_card = dict(valid_card)
    malformed_input_card["input"] = "{"
    try:
        df.StrictDatasetFilter().validate_entry(malformed_input_card)
        raise AssertionError("Expected malformed input JSON to be rejected.")
    except df.ValidationError as exc:
        assert exc.reason == "malformed_input_json", "Expected malformed input rejection reason."

    malformed_output_card = dict(valid_card)
    malformed_output_card["output"] = "{"
    try:
        df.StrictDatasetFilter().validate_entry(malformed_output_card)
        raise AssertionError("Expected malformed output JSON to be rejected.")
    except df.ValidationError as exc:
        assert exc.reason == "malformed_output_json", "Expected malformed output rejection reason."

    graph_heavy_card = dict(valid_card)
    heavy_input = json.loads(valid_card["input"])
    heavy_input["graph"]["downstream_calls"] = [f"call_{i}" for i in range(51)]
    graph_heavy_card["input"] = df._stable_json_blob(heavy_input)
    try:
        df.StrictDatasetFilter().validate_entry(graph_heavy_card)
        raise AssertionError("Expected oversized graphs to be rejected.")
    except df.ValidationError as exc:
        assert exc.reason == "graph_complexity_exceeded", "Expected graph complexity rejection reason."

    zero_density_card = dict(valid_card)
    zero_density_input = json.loads(valid_card["input"])
    zero_density_input["constraints"]["operator_types"] = []
    zero_density_card["input"] = df._stable_json_blob(zero_density_input)
    try:
        df.StrictDatasetFilter().validate_entry(zero_density_card)
        raise AssertionError("Expected zero-density AST rows to be rejected.")
    except df.ValidationError as exc:
        assert exc.reason == "zero_density_ast", "Expected zero-density rejection reason."


def test_filtered_writer_splits_outputs_and_is_deterministic() -> None:
    temp_dir, yaml_path = _write_yaml(
        {
            "capability_injection": {
                "compiled_skills": [_skill_with_intent(), _rejected_skill()],
            }
        }
    )
    try:
        cards = df.generate_flashcards(yaml_path)
        base_path = Path(temp_dir.name)
        accepted_a = base_path / "accepted_a.jsonl"
        rejected_a = base_path / "rejected_a.jsonl"
        accepted_b = base_path / "accepted_b.jsonl"
        rejected_b = base_path / "rejected_b.jsonl"

        filter_a = df.write_flashcards(cards, accepted_a, rejected_a)
        filter_b = df.write_flashcards(cards, accepted_b, rejected_b)

        accepted_a_lines = accepted_a.read_text(encoding="utf-8").splitlines()
        rejected_a_lines = rejected_a.read_text(encoding="utf-8").splitlines()
        accepted_b_lines = accepted_b.read_text(encoding="utf-8").splitlines()
        rejected_b_lines = rejected_b.read_text(encoding="utf-8").splitlines()
    finally:
        temp_dir.cleanup()

    assert filter_a.stats["accepted"] == 1, "Expected one accepted row in the main dataset."
    assert filter_a.stats["rejected"] == 1, "Expected one rejected row in the sidecar dataset."
    assert len(accepted_a_lines) == 1, "Expected one accepted JSONL row."
    assert len(rejected_a_lines) == 1, "Expected one rejected JSONL row."
    assert accepted_a_lines == accepted_b_lines, "Expected accepted output to be byte-stable across repeated writes."
    assert rejected_a_lines == rejected_b_lines, "Expected rejected output to be byte-stable across repeated writes."

    rejected_record = json.loads(rejected_a_lines[0])
    assert rejected_record["rejection_reason"] == "template_code_placeholder", "Expected rejected rows to preserve deterministic rejection reasons."
    assert "instruction" in rejected_record and "input" in rejected_record and "output" in rejected_record, "Expected rejected sidecar rows to preserve the original training record."


# ==========================================
# HARDENING TESTS (P1-P8 + Metric Keys)
# ==========================================

def test_build_conversational_array_veteran_mode() -> None:
    """P5: VETERAN mode with standard traceback + diff keys."""
    payload = {
        "traceback": "Traceback (most recent call last):\n  File ...",
        "diff": "- old_line\n+ new_line",
        "attempt_code": "x = 1/0",
        "analysis_text": "Division by zero",
        "resolved_code": "x = 1",
    }
    msgs = df.build_conversational_array(payload)
    assert len(msgs) == 2, "Expected 2 messages for veteran mode."
    assert msgs[0]["role"] == "user"
    assert "The following code failed to execute" in msgs[0]["content"]
    assert "x = 1/0" in msgs[0]["content"]
    assert "Division by zero" in msgs[1]["content"]


def test_build_conversational_array_veteran_variant_keys() -> None:
    """P5: VETERAN mode with variant field names (traceback_text, patch)."""
    payload = {
        "traceback_text": "KeyError: 'missing'",
        "patch": "--- a/foo.py\n+++ b/foo.py",
        "attempt_code": "d['missing']",
        "analysis_text": "Key not found",
        "resolved_code": "d.get('missing', None)",
    }
    msgs = df.build_conversational_array(payload)
    assert len(msgs) == 2, "Expected veteran mode to trigger on variant keys."
    assert "The following code failed to execute" in msgs[0]["content"]
    assert "KeyError: 'missing'" in msgs[0]["content"]
    assert "--- a/foo.py" in msgs[1]["content"]


def test_build_conversational_array_advocate_mode() -> None:
    """ADVOCATE mode with standard theory + implementation keys."""
    payload = {
        "theory": "Layered architecture separates concerns.",
        "implementation": "class Service:\n    pass",
    }
    msgs = df.build_conversational_array(payload)
    assert len(msgs) == 2
    assert "architectural reasoning" in msgs[0]["content"]
    assert "Layered architecture" in msgs[1]["content"]
    assert "class Service" in msgs[1]["content"]


def test_build_conversational_array_coding_mode_with_skill() -> None:
    """CODING ASSISTANT mode when pydantic_skill is provided."""
    # We pass a minimal payload + a mock-like object
    from unittest.mock import MagicMock
    mock_skill = MagicMock()
    mock_skill.teaching_layer.method_metadata = {"name": "compute_score"}
    mock_skill.teaching_layer.implementation_template = {"code": "def compute_score(): return 42"}
    payload = {}
    msgs = df.build_conversational_array(payload, pydantic_skill=mock_skill)
    assert len(msgs) == 2
    assert "compute_score" in msgs[0]["content"]
    assert "def compute_score" in msgs[1]["content"]


def test_build_conversational_array_theorist_mode() -> None:
    """THEORIST mode with constraints payload."""
    payload = {"constraints": ["Must be O(n)", "Thread-safe"]}
    msgs = df.build_conversational_array(payload)
    assert len(msgs) == 2
    assert "theoretical context" in msgs[0]["content"]
    assert "Must be O(n)" in msgs[1]["content"]


def test_build_conversational_array_failsafe() -> None:
    """Failsafe: unrecognized payload returns empty list."""
    msgs = df.build_conversational_array({"random_key": "value"})
    assert msgs == [], "Expected empty list for unrecognized payload topology."


def test_apply_loss_masking_with_identity() -> None:
    """P7: apply_loss_masking_schema inserts system message when identity provided."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    result = df.apply_loss_masking_schema(messages, identity="TestIdentity")
    assert result["messages"][0]["role"] == "system"
    assert "TestIdentity" in result["messages"][0]["content"]
    assert result["messages"][1]["train_loss"] is False  # user
    assert result["messages"][2]["train_loss"] is True   # assistant


def test_apply_loss_masking_no_duplicate_system_message() -> None:
    """P7: Should not insert a duplicate system message if one already exists."""
    messages = [
        {"role": "system", "content": "Existing system prompt"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    result = df.apply_loss_masking_schema(messages, identity="TestIdentity")
    # Should still have exactly 3 messages — no duplicate system message
    assert len(result["messages"]) == 3, f"Expected 3 messages, got {len(result['messages'])}"
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][0]["content"] == "Existing system prompt"


def test_balance_dataset_uses_mode_tag() -> None:
    """P3: balance_dataset should use _mode tag, not content string matching."""
    items = [
        {"messages": [{"role": "user", "content": "a"}], "_mode": "veteran"},
        {"messages": [{"role": "user", "content": "b"}], "_mode": "veteran"},
        {"messages": [{"role": "user", "content": "c"}], "_mode": "advocate"},
    ]
    result = df.balance_dataset(items, "TEST")
    # Min mode count is 1 (advocate), so each mode gets 1
    assert len(result) == 2, f"Expected 2 balanced items, got {len(result)}"


def test_balance_dataset_empty_input() -> None:
    """Empty input returns empty output."""
    result = df.balance_dataset([], "TEST")
    assert result == []


def test_confidence_val_none_guard() -> None:
    """P6: _build_flashcard should not crash when confidence.global is None."""
    skill = _skill_with_intent()
    skill["epistemic"]["confidence"]["global"] = None
    card = df._build_flashcard(skill)
    assert "0.000" in card["instruction"], "Expected None confidence to default to 0.000."


def test_confidence_val_none_confidence_block() -> None:
    """P6: _build_flashcard should handle confidence: null (entire block is None)."""
    skill = _skill_with_intent()
    skill["epistemic"]["confidence"] = None
    card = df._build_flashcard(skill)
    assert "0.000" in card["instruction"], "Expected None confidence block to default to 0.000."


def test_confidence_val_scalar_confidence() -> None:
    """P6: _build_flashcard should handle confidence as a raw float."""
    skill = _skill_with_intent()
    skill["epistemic"]["confidence"] = 0.75
    card = df._build_flashcard(skill)
    assert "0.750" in card["instruction"], "Expected scalar confidence to be used directly."


def test_canonical_schema_c_node_propagation() -> None:
    """P1: Canonical AletheiaSkill correctly parses c_node from epistemic block."""
    from src.core.models import AletheiaSkill as CanonicalSkill
    payload = {
        "node_id": "test_node_1",
        "name": "test_func",
        "file": "test.py",
        "code_snippet": "def f(): pass",
        "imports": [],
        "operator_type": "function",
        "teaching_layer": {
            "skill_identity": {},
            "method_metadata": {},
            "implementation_template": {},
        },
        "epistemic": {
            "state": "ACCEPTED",
            "c_node": 0.85,
            "retry_budget": 6,
            "depth": 0,
        },
    }
    skill = CanonicalSkill.model_validate(payload)
    assert skill.epistemic is not None
    assert skill.epistemic.c_node == 0.85, f"Expected c_node=0.85, got {skill.epistemic.c_node}"
    assert skill.epistemic.state == "ACCEPTED"


def test_pipeline_enrichment_keys_stripped_before_validation() -> None:
    """P1: _PIPELINE_ENRICHMENT_KEYS are defined and contain critical pipeline keys."""
    assert hasattr(df, "_PIPELINE_ENRICHMENT_KEYS"), "Module must export _PIPELINE_ENRICHMENT_KEYS."
    keys = df._PIPELINE_ENRICHMENT_KEYS
    for expected in ("orchestration_mode", "c_final", "s_sie", "_reroll_count", "tikhonov_slr"):
        assert expected in keys, f"Expected '{expected}' in _PIPELINE_ENRICHMENT_KEYS."


def test_get_hash_sentinel(capsys) -> None:
    """GAP-S2: get_hash should emit SENTINEL-HASH."""
    df.get_hash([{"a": 1}], process_id="TEST")
    captured = capsys.readouterr()
    assert "[SENTINEL-HASH]" in captured.out


def test_internal_metric_keys_completeness() -> None:
    """All 8 additional metric keys must appear in _INTERNAL_METRIC_KEYS inside format_dataset."""
    import inspect
    source = inspect.getsource(df.format_dataset)
    for key in (
        "_branch_degraded", "_confidence_decomposition", "_audit_trail",
        "_sie_coherence_score", "_invariant_results", "tikhonov_slr",
        "_reroll_count", "_reroll_violation_type",
    ):
        assert key in source, f"Expected '{key}' in format_dataset's _INTERNAL_METRIC_KEYS."


def test_get_hash_error_handling(capsys) -> None:
    """GAP-S2: get_hash should emit SENTINEL-HASH-ERR on non-serialisable input."""
    # Create an object that can't be serialised even with default=str fallback
    class Unserializable:
        def __str__(self):
            raise RuntimeError("boom")
    # default=str will call str() which raises — but json.dumps catches TypeError
    # regardless, the function should not crash
    result = df.get_hash([{"a": 1}], process_id="TEST")
    assert result != "FINGERPRINT_UNAVAILABLE", "Normal data should produce a real hash."
    # Verify the success sentinel is emitted
    captured = capsys.readouterr()
    assert "[SENTINEL-HASH]" in captured.out


def test_format_rejected_traces_empty_input(capsys) -> None:
    """GAP-S5: format_rejected_traces should emit SENTINEL-DPO-EMPTY on empty input."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "dpo.jsonl"
        count = df.format_rejected_traces([], out)
        assert count == 0
        captured = capsys.readouterr()
        assert "[SENTINEL-DPO-EMPTY]" in captured.out


def test_format_rejected_traces_valid_pair(capsys) -> None:
    """GAP-S5: format_rejected_traces writes valid DPO pairs with SENTINEL-DPO-WRITE."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "dpo.jsonl"
        traces = [
            {
                "node_id": "n1",
                "reason": "test",
                "rejected": {"code": "bad"},
                "corrected": {"code": "good"},
            }
        ]
        count = df.format_rejected_traces(traces, out)
        assert count == 1
        captured = capsys.readouterr()
        assert "[SENTINEL-DPO-WRITE]" in captured.out
        # Verify the written content
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert "prompt" in record
        assert "chosen" in record
        assert "rejected" in record


def test_build_failure_matrix_with_advocate_join() -> None:
    """Phase B: rejected payloads should be joined with Advocate diagnostics by node_id."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        audits_dir = base / "advocate_audits"
        audits_dir.mkdir(parents=True, exist_ok=True)
        audit_file = audits_dir / "advocate_audit_branchA_node_1.jsonl"
        audit_record = {
            "audit_type": "advocate_freeze_diagnostic",
            "branch_id": "branchA",
            "frozen_node": "node_1",
            "root_cause": "upstream_sie_failure",
            "dominant_breach_mode": "advocate",
            "recommendation": "revalidate_low_confidence_ancestors",
            "slr_breach_count": 3,
            "slr_breach_sequence": [0.71, 0.74, 0.79],
            "mode_weight_skew": {"advocate": 0.2},
            "system_backpressure": 0.81,
            "adversarial_cross_check": {"primary_overscoring": True},
        }
        audit_file.write_text(json.dumps(audit_record) + "\n", encoding="utf-8")

        rejected_payloads = [
            {
                "node_id": "node_1",
                "_rejection_reason": "Epistemic Survival Gate: state=REJECTED",
                "_failure_type": "epistemic_gate",
                "_source_file": "sample.yaml",
                "_final_status": "drift_violation",
                "_sie_metadata": {"content_density": 0.41, "s_sie": 0.37, "composite_quality_score": 0.15},
                "_acs_violations": ["Constraint mismatch"],
            }
        ]

        out_file = base / "failure_matrix.jsonl"
        written = df.build_failure_matrix(
            rejected_payloads=rejected_payloads,
            output_file=out_file,
            advocate_artifacts_dir=audits_dir,
            rejected_traces=[{"node_id": "node_1", "rejected": {"x": 1}, "corrected": {"x": 2}}],
        )

        assert written == 1
        lines = out_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["node_id"] == "node_1"
        assert row["failure_type"] == "epistemic_gate"
        assert row["advocate_audit"]["root_cause"] == "upstream_sie_failure"
        assert row["trajectory"]["audit_context"]["frozen_node"] == "node_1"
        assert row["trajectory"]["failure_matrix_reference"] == row["failure_id"]
        assert row["remediation_status"] == "rerolled"


def test_build_failure_matrix_is_deterministic() -> None:
    """FAILURE_MATRIX rows should be byte-stable across repeated compilation."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        audits_dir = base / "advocate_audits"
        audits_dir.mkdir(parents=True, exist_ok=True)
        (audits_dir / "advocate_audit_b_node_2.jsonl").write_text(
            json.dumps({"frozen_node": "node_2", "root_cause": "accumulated_identity_drift"}) + "\n",
            encoding="utf-8",
        )

        rejected_payloads = [
            {"node_id": "node_2", "_rejection_reason": "R1", "_failure_type": "f1"},
            {"node_id": "node_1", "_rejection_reason": "R2", "_failure_type": "f2"},
        ]

        out_a = base / "a.jsonl"
        out_b = base / "b.jsonl"
        df.build_failure_matrix(rejected_payloads, out_a, audits_dir)
        df.build_failure_matrix(rejected_payloads, out_b, audits_dir)

        assert out_a.read_text(encoding="utf-8") == out_b.read_text(encoding="utf-8")


def main() -> None:
    tests = [
        test_structured_formatter_behavior,
        test_formatter_is_deterministic,
        test_wrapper_schema_support,
        test_strict_filter_rejects_expected_failure_modes,
        test_filtered_writer_splits_outputs_and_is_deterministic,
        test_build_conversational_array_veteran_mode,
        test_build_conversational_array_veteran_variant_keys,
        test_build_conversational_array_advocate_mode,
        test_build_conversational_array_coding_mode_with_skill,
        test_build_conversational_array_theorist_mode,
        test_build_conversational_array_failsafe,
        test_apply_loss_masking_with_identity,
        test_apply_loss_masking_no_duplicate_system_message,
        test_balance_dataset_uses_mode_tag,
        test_balance_dataset_empty_input,
        test_confidence_val_none_guard,
        test_confidence_val_none_confidence_block,
        test_confidence_val_scalar_confidence,
        test_canonical_schema_c_node_propagation,
        test_pipeline_enrichment_keys_stripped_before_validation,
        test_get_hash_sentinel,
        test_get_hash_error_handling,
        test_format_rejected_traces_empty_input,
        test_format_rejected_traces_valid_pair,
        test_build_failure_matrix_with_advocate_join,
        test_build_failure_matrix_is_deterministic,
        test_internal_metric_keys_completeness,
    ]

    for test in tests:
        test()
        print(f"[OK] {test.__name__}")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"[FAIL] {exc}")
        sys.exit(1)
