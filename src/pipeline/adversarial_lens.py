"""
Adversarial Lens Layer (ALL) — Orthogonal evaluation channel.

Does NOT share assumptions with ACSExecutionGovernor.
Evaluates nodes on parsimony, structural surprise, and boundary sensitivity.
Disagreement with the primary is recorded, not used to block acceptance.
"""

import math
import hashlib
from typing import Any, List, Tuple
from dataclasses import dataclass, field


# ── Sentinel codes for this module ───────────────────────────────────
SENTINEL_ALL_CONFLICT   = "[ALL-CONFLICT]"     # Primary and adversarial disagree strongly
SENTINEL_ALL_SUSPICIOUS = "[ALL-SUSPICIOUS]"   # Node passes primary but adversarial flags it
SENTINEL_ALL_VALIDATED  = "[ALL-VALIDATED]"    # Both evaluators agree — high confidence


CONFLICT_THRESHOLDS = {
    "agreement":      0.15,   # |primary - adversarial| < 0.15 → both agree
    "minor_conflict": 0.30,   # 0.15–0.30 → note it, don't act
    "major_conflict": 0.30,   # > 0.30 → emit SENTINEL-ALL-CONFLICT, tag node
}


@dataclass
class AdversarialVerdict:
    node_id: str
    adversarial_score: float          # 0.0–1.0 from adversarial evaluator
    primary_score: float              # c_final from primary evaluator
    conflict_magnitude: float         # abs(primary - adversarial)
    conflict_class: str               # "agreement" | "minor_conflict" | "major_conflict"
    flags: List[str] = field(default_factory=list)
    structural_fingerprint: str = ""  # MD5 of operator_types + reasoning strategy


class AdversarialLens:
    """
    Evaluates nodes from an orthogonal perspective to the ACS primary evaluator.

    Primary evaluator rewards: completeness, structural density, SIE coherence.
    This lens rewards: parsimony, boundary sensitivity, structural novelty.

    Key design constraints:
    - Never blocks acceptance (advisory only)
    - Never shares penalty multipliers with ACS
    - Conflict magnitude is recorded in telemetry and attached to the node
    - High-conflict nodes are still accepted but tagged _adversarial_conflict=True
    """

    def __init__(self, conflict_threshold: float = 0.30):
        self.conflict_threshold = conflict_threshold
        self._prior_fingerprints: List[str] = []  # rolling window of seen fingerprints

    def _compute_parsimony_score(self, node: Any) -> float:
        """
        Reward parsimony: nodes that achieve high SIE with fewer operator_types
        are more general and less overfit to a specific pattern.

        score = s_sie / max(1, log2(operator_count + 1))
        """
        sie_node = getattr(node, "sie_node", None)
        s_sie = getattr(sie_node, "s_sie", 0.0) if sie_node else 0.0

        semantics = getattr(node, "semantics", {})
        if isinstance(semantics, dict):
            ops = semantics.get("operator_types", [])
        else:
            ops = getattr(semantics, "operator_types", [])
        op_count = len(ops) if ops else 1

        parsimony = s_sie / max(1.0, math.log2(op_count + 1))
        return min(1.0, round(parsimony, 4))

    def _compute_boundary_sensitivity(self, node: Any) -> float:
        """
        Reward nodes that explicitly model their own failure modes.

        score = min(1.0, (failure_mode_count + constraint_count) / 6.0)
        """
        teaching = getattr(node, "teaching_layer", None)
        reasoning = getattr(teaching, "reasoning_vectors", None) if teaching else None

        failure_modes: List[str] = []
        constraints: List[Any] = []
        if reasoning:
            failure_modes = getattr(reasoning, "failure_modes", []) or []
            constraints = getattr(reasoning, "constraints", []) or []

        count = len(failure_modes) + len(constraints)
        return min(1.0, round(count / 6.0, 4))

    def _compute_structural_novelty(self, node: Any) -> Tuple[float, str]:
        """
        Reward structural novelty relative to previously seen nodes in this run.
        Uses MD5 fingerprint of operator_types + strategy to detect repetition.

        score = 1.0 if unseen, decays toward 0.3 with repetition frequency.
        """
        semantics = getattr(node, "semantics", {}) or {}
        ops = semantics.get("operator_types", []) if isinstance(semantics, dict) else []
        strategy = ""
        teaching = getattr(node, "teaching_layer", None)
        if teaching:
            rv = getattr(teaching, "reasoning_vectors", None)
            if rv:
                strategy = getattr(rv, "strategy", "") or ""

        fingerprint = hashlib.md5(
            (str(sorted(ops)) + strategy).encode("utf-8")
        ).hexdigest()

        repeat_count = self._prior_fingerprints.count(fingerprint)
        novelty = max(0.3, 1.0 - (repeat_count * 0.2))

        # Register fingerprint (keep rolling window of last 200)
        self._prior_fingerprints.append(fingerprint)
        if len(self._prior_fingerprints) > 200:
            self._prior_fingerprints.pop(0)

        return round(novelty, 4), fingerprint

    def evaluate(self, node: Any, primary_c_final: float) -> AdversarialVerdict:
        """
        Run the adversarial evaluation. Returns a verdict that is advisory only.
        The DAGRuntime uses the verdict for telemetry and tagging, not gating.
        """
        node_id = getattr(node, "node_id", "unknown")

        parsimony = self._compute_parsimony_score(node)
        boundary = self._compute_boundary_sensitivity(node)
        novelty_val, fingerprint = self._compute_structural_novelty(node)

        # Adversarial score: equal weight on each dimension
        # Deliberately NOT using SIE, ACS, or topology — orthogonal by design
        adversarial_score = round(
            (parsimony * 0.40) + (boundary * 0.35) + (novelty_val * 0.25),
            4,
        )

        conflict = abs(primary_c_final - adversarial_score)

        if conflict < CONFLICT_THRESHOLDS["agreement"]:
            conflict_class = "agreement"
        elif conflict < CONFLICT_THRESHOLDS["minor_conflict"]:
            conflict_class = "minor_conflict"
        else:
            conflict_class = "major_conflict"

        flags = []
        if conflict_class == "major_conflict":
            if primary_c_final > adversarial_score:
                flags.append("primary_overscoring")
            else:
                flags.append("adversarial_overscoring")

        if novelty_val < 0.5:
            flags.append("low_novelty_repeat_structure")

        if parsimony < 0.3:
            flags.append("complexity_without_coherence")

        return AdversarialVerdict(
            node_id=node_id,
            adversarial_score=adversarial_score,
            primary_score=primary_c_final,
            conflict_magnitude=round(conflict, 4),
            conflict_class=conflict_class,
            flags=flags,
            structural_fingerprint=fingerprint,
        )
