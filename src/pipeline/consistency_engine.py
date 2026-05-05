"""
Global Consistency Engine v1.0

Detects contradictions and redundancy across validated nodes in the DAG.
Acts as a post-loop global gate: once all nodes have been individually
scored and accepted, this engine checks cross-node consistency before
final result aggregation.

Contradiction detection:
  - Constraint-level: two nodes share constraint type+tags but opposing
    ``valid`` flags.
  - Vector-level: two nodes' alignment_vector vectors are anti-correlated
    (cosine similarity < -0.5).

Redundancy detection:
  - Two nodes whose alignment_vector vectors have cosine similarity > threshold
    (default 0.95) are considered semantically redundant.

Resolution strategy: for each conflicting/redundant pair, reject the node
with the lower ``c_node`` (confidence).
"""

import logging
import numpy as np
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Cosine similarity between two vectors.  Returns 0.0 on degenerate inputs.
    """
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _get_alignment_vector(node: Any) -> List[float]:
    """Extract alignment_vector from a node's SIE data, or return zeros."""
    sie = getattr(node, "sie_node", None)
    if sie is not None:
        av = getattr(sie, "alignment_vector", None)
        if av and isinstance(av, (list, tuple)) and len(av) >= 3:
            return list(av)
    return [0.0, 0.0, 0.0]


def _get_constraints(node: Any) -> List[Any]:
    """Extract constraints list from a node."""
    c = getattr(node, "constraints", None)
    if c and isinstance(c, (list, tuple)):
        return list(c)
    if isinstance(c, dict):
        initial = c.get("initial_constraints", [])
        if isinstance(initial, (list, tuple)):
            return list(initial)
    return []


def _constraint_attr(constraint: Any, key: str, default: Any = None) -> Any:
    """Read a constraint attribute from either object-style or dict-style payloads."""
    if isinstance(constraint, dict):
        return constraint.get(key, default)
    return getattr(constraint, key, default)


def _get_c_node(node: Any) -> float:
    ep = getattr(node, "epistemic", None)
    if ep:
        return float(getattr(ep, "c_node", 0.0))
    return 0.0


class ConsistencyEngine:
    """Detect contradictions and redundancy across a set of validated nodes."""

    def __init__(
        self,
        redundancy_threshold: float = 0.95,
        anti_correlation_threshold: float = -0.5,
    ):
        self.redundancy_threshold = redundancy_threshold
        self.anti_correlation_threshold = anti_correlation_threshold

    # ------------------------------------------------------------------
    # Contradiction detection
    # ------------------------------------------------------------------
    def detect_contradictions(self, nodes: List[Any]) -> List[Dict[str, Any]]:
        """
        Compare every node-pair for contradictory constraints or
        anti-correlated phase_gradient vectors.

        Returns a list of dicts: ``{node_a, node_b, reason, severity}``.
        """
        contradictions: List[Dict[str, Any]] = []
        n = len(nodes)
        for i in range(n):
            for j in range(i + 1, n):
                na, nb = nodes[i], nodes[j]
                nid_a = getattr(na, "node_id", str(i))
                nid_b = getattr(nb, "node_id", str(j))

                # --- Constraint-level contradiction ---
                ca = _get_constraints(na)
                cb = _get_constraints(nb)
                for c1 in ca:
                    for c2 in cb:
                        c1_type = _constraint_attr(c1, "type")
                        c2_type = _constraint_attr(c2, "type")
                        c1_tags = set(_constraint_attr(c1, "tags", []) or [])
                        c2_tags = set(_constraint_attr(c2, "tags", []) or [])
                        c1_valid = _constraint_attr(c1, "valid")
                        c2_valid = _constraint_attr(c2, "valid")
                        if (
                            c1_type and c1_type == c2_type
                            and c1_tags and c1_tags == c2_tags
                            and c1_valid is not None and c2_valid is not None
                            and c1_valid != c2_valid
                        ):
                            contradictions.append({
                                "node_a": nid_a,
                                "node_b": nid_b,
                                "reason": f"constraint_contradiction: type={c1_type}, tags={c1_tags}",
                                "severity": "error",
                            })

                # --- Vector-level contradiction ---
                pg_a = _get_alignment_vector(na)
                pg_b = _get_alignment_vector(nb)
                cos = _cosine_similarity(pg_a, pg_b)
                if cos < self.anti_correlation_threshold:
                    contradictions.append({
                        "node_a": nid_a,
                        "node_b": nid_b,
                        "reason": f"alignment_vector_anti_correlated: cosine={cos:.4f}",
                        "severity": "warning",
                    })

        return contradictions

    # ------------------------------------------------------------------
    # Redundancy detection
    # ------------------------------------------------------------------
    def detect_redundancy(self, nodes: List[Any]) -> List[Tuple[str, str]]:
        """
        Pairwise cosine similarity of alignment_vector vectors.
        Pairs above ``redundancy_threshold`` are semantically redundant.
        """
        redundant: List[Tuple[str, str]] = []
        n = len(nodes)
        for i in range(n):
            for j in range(i + 1, n):
                pg_a = _get_alignment_vector(nodes[i])
                pg_b = _get_alignment_vector(nodes[j])
                # Skip zero-vectors (no SIE data)
                if np.linalg.norm(pg_a) < 1e-12 or np.linalg.norm(pg_b) < 1e-12:
                    continue
                cos = _cosine_similarity(pg_a, pg_b)
                if cos >= self.redundancy_threshold:
                    nid_a = getattr(nodes[i], "node_id", str(i))
                    nid_b = getattr(nodes[j], "node_id", str(j))
                    redundant.append((nid_a, nid_b))
        return redundant

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------
    def resolve_contradictions(
        self, nodes: List[Any], contradictions: List[Dict[str, Any]]
    ) -> List[str]:
        """
        For each contradiction pair, mark the node with lower ``c_node``
        for rejection.  Returns list of rejected node_ids.
        """
        node_map = {getattr(n, "node_id", str(i)): n for i, n in enumerate(nodes)}
        rejected: set = set()
        for c in contradictions:
            a_id, b_id = c["node_a"], c["node_b"]
            a_c = _get_c_node(node_map.get(a_id))
            b_c = _get_c_node(node_map.get(b_id))
            loser = b_id if a_c >= b_c else a_id
            rejected.add(loser)
        return list(rejected)

    def collapse_redundant(
        self, nodes: List[Any], redundant_pairs: List[Tuple[str, str]]
    ) -> List[str]:
        """
        For each redundant pair, mark the node with lower ``c_node``
        for removal.  Returns list of collapsed (rejected) node_ids.
        """
        node_map = {getattr(n, "node_id", str(i)): n for i, n in enumerate(nodes)}
        collapsed: set = set()
        for a_id, b_id in redundant_pairs:
            a_c = _get_c_node(node_map.get(a_id))
            b_c = _get_c_node(node_map.get(b_id))
            loser = b_id if a_c >= b_c else a_id
            collapsed.add(loser)
        return list(collapsed)

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
    def validate_global_consistency(self, nodes: List[Any]) -> Dict[str, Any]:
        """
        Run contradiction + redundancy detection and resolution.

        Returns::

            {
                "contradictions": [...],
                "redundancies": [...],
                "rejected_ids": [...],
                "status": "consistent" | "degraded" | "failed",
            }
        """
        if not nodes:
            return {
                "contradictions": [],
                "redundancies": [],
                "rejected_ids": [],
                "status": "consistent",
            }

        contradictions = self.detect_contradictions(nodes)
        redundancies = self.detect_redundancy(nodes)

        rejected_from_contradictions = self.resolve_contradictions(nodes, contradictions)
        rejected_from_redundancy = self.collapse_redundant(nodes, redundancies)

        all_rejected = list(set(rejected_from_contradictions) | set(rejected_from_redundancy))

        if not contradictions and not redundancies:
            status = "consistent"
        elif len(all_rejected) < len(nodes):
            status = "degraded"
        else:
            status = "failed"

        if contradictions or redundancies:
            logger.warning(
                f"Global consistency: {len(contradictions)} contradictions, "
                f"{len(redundancies)} redundancies, {len(all_rejected)} nodes rejected"
            )

        return {
            "contradictions": contradictions,
            "redundancies": redundancies,
            "rejected_ids": all_rejected,
            "status": status,
        }
