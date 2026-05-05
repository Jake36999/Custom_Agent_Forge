"""
Tests for src.pipeline.adversarial_lens — Adversarial Lens Layer (ALL).
Uses lightweight SimpleNamespace stubs; no DAGRuntime dependency.
"""
import sys, os
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so ``src.*`` imports resolve.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.pipeline.adversarial_lens import (
    AdversarialLens,
    AdversarialVerdict,
    CONFLICT_THRESHOLDS,
    SENTINEL_ALL_CONFLICT,
    SENTINEL_ALL_SUSPICIOUS,
    SENTINEL_ALL_VALIDATED,
)


# ---------------------------------------------------------------------------
# Helpers — lightweight node stubs
# ---------------------------------------------------------------------------

def _make_node(
    node_id="test_node",
    s_sie=0.8,
    operator_types=None,
    failure_modes=None,
    constraints=None,
    strategy="",
):
    """Build a SimpleNamespace that quacks like an AletheiaSkill for the lens."""
    reasoning_vectors = SimpleNamespace(
        failure_modes=failure_modes or [],
        constraints=constraints or [],
        strategy=strategy,
    )
    teaching_layer = SimpleNamespace(reasoning_vectors=reasoning_vectors)
    sie_node = SimpleNamespace(s_sie=s_sie)
    return SimpleNamespace(
        node_id=node_id,
        sie_node=sie_node,
        semantics={"operator_types": operator_types or []},
        teaching_layer=teaching_layer,
    )


# ---------------------------------------------------------------------------
# Parsimony
# ---------------------------------------------------------------------------

def test_parsimony_high_sie_few_ops():
    """High s_sie with few operator_types → high parsimony."""
    lens = AdversarialLens()
    node = _make_node(s_sie=0.9, operator_types=["call", "assign"])
    score = lens._compute_parsimony_score(node)
    assert score >= 0.5, f"Expected high parsimony, got {score}"


def test_parsimony_high_sie_many_ops():
    """Same s_sie but many operator_types → lower parsimony."""
    lens = AdversarialLens()
    few = _make_node(s_sie=0.9, operator_types=["a", "b"])
    many = _make_node(s_sie=0.9, operator_types=[f"op{i}" for i in range(15)])
    p_few = lens._compute_parsimony_score(few)
    p_many = lens._compute_parsimony_score(many)
    assert p_few > p_many, f"few_ops={p_few} should exceed many_ops={p_many}"


def test_parsimony_zero_sie():
    """s_sie=0 → parsimony=0 regardless of ops."""
    lens = AdversarialLens()
    node = _make_node(s_sie=0.0, operator_types=["a"])
    assert lens._compute_parsimony_score(node) == 0.0


# ---------------------------------------------------------------------------
# Boundary sensitivity
# ---------------------------------------------------------------------------

def test_boundary_sensitivity_with_failure_modes():
    """3 failure_modes + 2 constraints → 5/6 ≈ 0.8333."""
    lens = AdversarialLens()
    node = _make_node(
        failure_modes=["f1", "f2", "f3"],
        constraints=["c1", "c2"],
    )
    score = lens._compute_boundary_sensitivity(node)
    assert 0.83 <= score <= 0.84, f"Expected ~0.8333, got {score}"


def test_boundary_sensitivity_no_teaching_layer():
    """Node without teaching_layer → 0.0 boundary."""
    lens = AdversarialLens()
    node = SimpleNamespace(
        node_id="bare",
        sie_node=SimpleNamespace(s_sie=0.5),
        semantics={"operator_types": []},
        teaching_layer=None,
    )
    assert lens._compute_boundary_sensitivity(node) == 0.0


def test_boundary_sensitivity_capped_at_one():
    """More than 6 items → capped at 1.0."""
    lens = AdversarialLens()
    node = _make_node(
        failure_modes=[f"f{i}" for i in range(5)],
        constraints=[f"c{i}" for i in range(5)],
    )
    assert lens._compute_boundary_sensitivity(node) == 1.0


# ---------------------------------------------------------------------------
# Structural novelty
# ---------------------------------------------------------------------------

def test_structural_novelty_first_seen():
    """First-ever fingerprint → novelty 1.0."""
    lens = AdversarialLens()
    node = _make_node(operator_types=["unique_op"], strategy="unique_strat")
    novelty, fp = lens._compute_structural_novelty(node)
    assert novelty == 1.0, f"Expected 1.0, got {novelty}"
    assert len(fp) == 32  # MD5 hex digest length


def test_structural_novelty_repeat():
    """Repeated fingerprint decays: 1st=1.0, 2nd=0.8, 3rd=0.6."""
    lens = AdversarialLens()
    node = _make_node(operator_types=["op_a"], strategy="strat_a")
    n1, _ = lens._compute_structural_novelty(node)
    n2, _ = lens._compute_structural_novelty(node)
    n3, _ = lens._compute_structural_novelty(node)
    assert n1 == 1.0
    assert n2 == 0.8
    assert n3 == 0.6


def test_structural_novelty_floor():
    """Novelty cannot drop below 0.3."""
    lens = AdversarialLens()
    node = _make_node(operator_types=["same"], strategy="same")
    for _ in range(10):
        novelty, _ = lens._compute_structural_novelty(node)
    assert novelty >= 0.3


def test_structural_novelty_window_cap():
    """Rolling window never exceeds 200 entries."""
    lens = AdversarialLens()
    for i in range(250):
        node = _make_node(operator_types=[f"op_{i}"], strategy=f"s_{i}")
        lens._compute_structural_novelty(node)
    assert len(lens._prior_fingerprints) <= 200


# ---------------------------------------------------------------------------
# Evaluate — verdict classification
# ---------------------------------------------------------------------------

def test_evaluate_agreement():
    """When primary ≈ adversarial → agreement class."""
    lens = AdversarialLens()
    # Build a node that produces an adversarial score near 0.5
    node = _make_node(
        s_sie=0.7,
        operator_types=["op"],
        failure_modes=["f1"],
        constraints=["c1"],
    )
    verdict = lens.evaluate(node, primary_c_final=0.5)
    # With s_sie=0.7, 1 op → parsimony=0.7/1=0.7, boundary=2/6=0.3333, novelty=1.0
    # adversarial ≈ 0.7*0.4 + 0.3333*0.35 + 1.0*0.25 = 0.28+0.1167+0.25 = 0.6467
    # |0.5 - 0.647| ≈ 0.147 → minor_conflict or agreement depending on exact rounding
    assert isinstance(verdict, AdversarialVerdict)
    assert verdict.conflict_class in ("agreement", "minor_conflict")


def test_evaluate_major_conflict():
    """Primary >> adversarial → major_conflict with primary_overscoring flag."""
    lens = AdversarialLens()
    # Low adversarial score: no SIE, no teaching layer, no novelty
    node = SimpleNamespace(
        node_id="weak",
        sie_node=SimpleNamespace(s_sie=0.0),
        semantics={"operator_types": []},
        teaching_layer=None,
    )
    verdict = lens.evaluate(node, primary_c_final=0.95)
    assert verdict.conflict_class == "major_conflict"
    assert "primary_overscoring" in verdict.flags


def test_evaluate_adversarial_overscoring():
    """Primary << adversarial → major_conflict with adversarial_overscoring."""
    lens = AdversarialLens()
    node = _make_node(
        s_sie=1.0,
        operator_types=["op"],
        failure_modes=["f1", "f2", "f3"],
        constraints=["c1", "c2", "c3"],
        strategy="unique",
    )
    # This yields high adversarial: parsimony ~1.0, boundary=1.0, novelty=1.0
    # adversarial ≈ 0.4*1.0 + 0.35*1.0 + 0.25*1.0 = 1.0
    verdict = lens.evaluate(node, primary_c_final=0.2)
    assert verdict.conflict_class == "major_conflict"
    assert "adversarial_overscoring" in verdict.flags


def test_evaluate_low_novelty_flag():
    """Repeated structure → low_novelty_repeat_structure flag."""
    lens = AdversarialLens()
    node = _make_node(operator_types=["same_op"], strategy="same")
    # Seed 4 repeats so novelty drops below 0.5
    for _ in range(4):
        lens._compute_structural_novelty(node)
    verdict = lens.evaluate(node, primary_c_final=0.5)
    assert "low_novelty_repeat_structure" in verdict.flags


def test_evaluate_complexity_without_coherence_flag():
    """Low parsimony → complexity_without_coherence flag."""
    lens = AdversarialLens()
    node = _make_node(
        s_sie=0.05,
        operator_types=[f"op{i}" for i in range(20)],
    )
    verdict = lens.evaluate(node, primary_c_final=0.5)
    assert "complexity_without_coherence" in verdict.flags


def test_conflict_thresholds_boundary():
    """Exactly at threshold boundaries."""
    assert CONFLICT_THRESHOLDS["agreement"] == 0.15
    assert CONFLICT_THRESHOLDS["minor_conflict"] == 0.30
    assert CONFLICT_THRESHOLDS["major_conflict"] == 0.30


def test_sentinel_constants_defined():
    """Sentinel constants exist and follow naming convention."""
    assert SENTINEL_ALL_CONFLICT.startswith("[ALL-")
    assert SENTINEL_ALL_SUSPICIOUS.startswith("[ALL-")
    assert SENTINEL_ALL_VALIDATED.startswith("[ALL-")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    tests = [
        test_parsimony_high_sie_few_ops,
        test_parsimony_high_sie_many_ops,
        test_parsimony_zero_sie,
        test_boundary_sensitivity_with_failure_modes,
        test_boundary_sensitivity_no_teaching_layer,
        test_boundary_sensitivity_capped_at_one,
        test_structural_novelty_first_seen,
        test_structural_novelty_repeat,
        test_structural_novelty_floor,
        test_structural_novelty_window_cap,
        test_evaluate_agreement,
        test_evaluate_major_conflict,
        test_evaluate_adversarial_overscoring,
        test_evaluate_low_novelty_flag,
        test_evaluate_complexity_without_coherence_flag,
        test_conflict_thresholds_boundary,
        test_sentinel_constants_defined,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
