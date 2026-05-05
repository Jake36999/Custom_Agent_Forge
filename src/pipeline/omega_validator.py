"""
Omega Handshake Validator v1.0

System-wide validation gate that checks five dimensions of DAG health.
Split into two phases per Correction 4:

  1. **Partial check** (per-node, pre-acceptance):  validates the node-local
     dimensions before the node is committed as ACCEPTED.  If partial check
     fails, the node is routed to REROLL while context is still live.

  2. **Final check** (global, post-aggregation):  validates system-wide
     invariants after all nodes finish.  Can only downgrade status, never
     upgrade.

Dimensions:
  D1  SIE Coherence        — edge-pair coherence across accepted nodes
  D2  SLR Integrity         — rolling SLR mean below critical threshold
  D3  Identity Stability    — drift_score < max_drift AND not frozen
  D4  Constraint Satisfaction — no accepted node has fatal invalid constraint
  D5  Topological Closure   — all accepted nodes reachable from a root
"""

import logging
import time
import numpy as np
import networkx as nx
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Thresholds (importable for tests)
OMEGA_EDGE_COHERENCE_FLOOR = 0.3
OMEGA_SLR_CRITICAL = 0.65
OMEGA_MAX_DRIFT = 0.3


def _get_alignment_vector(node: Any) -> Optional[List[float]]:
    sie = getattr(node, "sie_node", None)
    if sie is None:
        return None
    av = getattr(sie, "alignment_vector", None)
    if av and isinstance(av, (list, tuple)) and len(av) >= 3:
        return list(av)
    return None


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


class OmegaValidator:
    """Five-dimension system validation gate (partial + final)."""

    # ------------------------------------------------------------------
    # D6: Orphan State Closure
    # ------------------------------------------------------------------
    def validate_orphan_state_closure(self, all_nodes: Optional[List[Any]]) -> Dict[str, Any]:
        """Detect nodes stranded in non-terminal states at run completion."""
        if not all_nodes:
            return {
                "pass": True,
                "orphan_count": 0,
                "orphan_node_ids": [],
                "orphan_states": {},
            }

        terminal_states = {"ACCEPTED", "REJECTED", "terminal", "rejected"}
        orphan_node_ids: List[str] = []
        orphan_states: Dict[str, Any] = {}

        for idx, node in enumerate(all_nodes):
            node_id = str(getattr(node, "node_id", f"idx_{idx}"))
            ep = getattr(node, "epistemic", None)
            state = getattr(ep, "state", None)
            if isinstance(state, str) and state in terminal_states:
                continue
            # Any missing/unknown/non-terminal state is considered orphaned.
            orphan_node_ids.append(node_id)
            orphan_states[node_id] = state

        return {
            "pass": len(orphan_node_ids) == 0,
            "orphan_count": len(orphan_node_ids),
            "orphan_node_ids": orphan_node_ids,
            "orphan_states": orphan_states,
        }

    # ------------------------------------------------------------------
    # D1: Edge Coherence
    # ------------------------------------------------------------------
    def validate_sie_coherence(
        self, nodes: List[Any], nx_graph: Optional[Any] = None,
    ) -> Dict[str, Any]:
        violations = []
        if nx_graph is not None:
            node_map = {getattr(n, "node_id", str(i)): n for i, n in enumerate(nodes)}
            for u, v in nx_graph.edges():
                nu = node_map.get(u)
                nv = node_map.get(v)
                if nu is None or nv is None:
                    continue
                av_u = _get_alignment_vector(nu)
                av_v = _get_alignment_vector(nv)
                if av_u is None or av_v is None:
                    continue
                coh = _cosine_similarity(av_u, av_v)
                if coh < OMEGA_EDGE_COHERENCE_FLOOR:
                    violations.append({
                        "edge": (u, v),
                        "coherence": round(coh, 4),
                    })
        return {"pass": len(violations) == 0, "violations": violations}

    # ------------------------------------------------------------------
    # D2: SLR Integrity
    # ------------------------------------------------------------------
    def validate_slr_integrity(self, identity_manager: Any) -> Dict[str, Any]:
        if identity_manager is None:
            return {"pass": True, "mean": 0.0}
        mean = 0.0
        if hasattr(identity_manager, "get_rolling_slr_mean"):
            mean = identity_manager.get_rolling_slr_mean()
        return {"pass": mean < OMEGA_SLR_CRITICAL, "mean": round(mean, 4)}

    # ------------------------------------------------------------------
    # D3: Identity Stability
    # ------------------------------------------------------------------
    def validate_identity_stability(self, identity_manager: Any) -> Dict[str, Any]:
        if identity_manager is None:
            return {"pass": True, "drift": 0.0, "frozen": False}
        drift = getattr(identity_manager, "drift_score", 0.0)
        frozen = getattr(identity_manager, "frozen", False)
        max_drift = getattr(identity_manager, "max_drift", OMEGA_MAX_DRIFT)
        ok = drift < max_drift and not frozen
        return {"pass": ok, "drift": round(drift, 4), "frozen": frozen}

    # ------------------------------------------------------------------
    # D4: Constraint Satisfaction
    # ------------------------------------------------------------------
    def validate_constraint_satisfaction(self, nodes: List[Any]) -> Dict[str, Any]:
        violations = []
        for node in nodes:
            constraints = getattr(node, "constraints", None)
            if not constraints:
                continue
            for c in constraints:
                sev = getattr(c, "severity", "warning")
                valid = getattr(c, "valid", True)
                if sev == "fatal" and not valid:
                    violations.append({
                        "node_id": getattr(node, "node_id", "?"),
                        "constraint": getattr(c, "description", ""),
                    })
        return {"pass": len(violations) == 0, "violations": violations}

    # ------------------------------------------------------------------
    # D5: Topological Closure
    # ------------------------------------------------------------------
    def validate_topological_closure(
        self, nodes: List[Any], nx_graph: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if nx_graph is None or len(nodes) == 0:
            return {"pass": True, "orphan_count": 0}

        accepted_ids = {getattr(n, "node_id", str(i)) for i, n in enumerate(nodes)}
        # Roots = nodes with no predecessors in the full graph
        roots = {n for n in nx_graph.nodes() if nx_graph.in_degree(n) == 0}

        # Reachable from any root via accepted-only subgraph
        subgraph = nx_graph.subgraph(accepted_ids & set(nx_graph.nodes()))
        reachable = set()
        for r in roots:
            if r in subgraph:
                reachable |= nx.descendants(subgraph, r) | {r}

        orphans = accepted_ids - reachable
        # Nodes that ARE roots themselves are not orphans even if not
        # "reachable from another root"
        orphans -= roots
        return {"pass": len(orphans) == 0, "orphan_count": len(orphans)}

    # ------------------------------------------------------------------
    # Partial check (per-node, before acceptance)
    # ------------------------------------------------------------------
    def partial_check(
        self,
        node: Any,
        identity_manager: Any = None,
    ) -> Dict[str, Any]:
        """
        Lightweight per-node omega check run BEFORE a node is accepted.
        Checks D3 (identity stability) and D4 (constraint satisfaction)
        which are meaningful at node-level.  D1, D2, D5 require global context
        and are deferred to ``omega_handshake()``.

        Returns ``{"pass": bool, "dimensions": {...}}``.
        """
        d3 = self.validate_identity_stability(identity_manager)
        d4 = self.validate_constraint_satisfaction([node])

        ok = d3["pass"] and d4["pass"]
        return {
            "pass": ok,
            "dimensions": {
                "identity_stability": d3,
                "constraint_satisfaction": d4,
            },
        }

    # ------------------------------------------------------------------
    # Final check (global, after all nodes finished)
    # ------------------------------------------------------------------
    def omega_handshake(
        self,
        nodes: List[Any],
        nx_graph: Optional[Any] = None,
        identity_manager: Any = None,
        telemetry: Any = None,
        all_nodes: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Full system-wide validation across all 5 dimensions.

        Returns::

            {
                "status": "OMEGA_PASS" | "OMEGA_DEGRADED" | "OMEGA_FAIL",
                "dimensions": { ... per-dimension results ... },
                "timestamp": float,
            }

        Classification:
          - OMEGA_PASS:     all 5 dimensions pass
          - OMEGA_DEGRADED: 1-2 non-critical failures
          - OMEGA_FAIL:     ≥3 failures OR any critical (identity frozen)
        """
        d1 = self.validate_sie_coherence(nodes, nx_graph)
        d2 = self.validate_slr_integrity(identity_manager)
        d3 = self.validate_identity_stability(identity_manager)
        d4 = self.validate_constraint_satisfaction(nodes)
        d5 = self.validate_topological_closure(nodes, nx_graph)
        d6 = self.validate_orphan_state_closure(all_nodes)

        dims = {
            "sie_coherence": d1,
            "slr_integrity": d2,
            "identity_stability": d3,
            "constraint_satisfaction": d4,
            "topological_closure": d5,
            "orphan_state_closure": d6,
        }

        failures = sum(1 for d in dims.values() if not d["pass"])
        # Critical: identity frozen is always fatal
        critical = d3.get("frozen", False)

        if failures == 0:
            status = "OMEGA_PASS"
        elif critical or failures >= 3:
            status = "OMEGA_FAIL"
        else:
            status = "OMEGA_DEGRADED"

        if telemetry and hasattr(telemetry, "record"):
            telemetry.record(
                "omega", status.lower(),
                payload={"failures": failures, "critical": critical},
            )

        return {
            "status": status,
            "dimensions": dims,
            "timestamp": time.time(),
        }
