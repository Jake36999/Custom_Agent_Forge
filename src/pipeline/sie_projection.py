"""
SIE Projection Layer v1.0

Provides a canonical mapping from any AletheiaSkill (or CognitiveNode) into
a flat ``Dict[str, float]`` of invariant-accessible metrics.

This layer resolves Risk 2: different nodes expose metrics through different
namespaces (SIE vs ACS vs Epistemic).  The projection standardises everything
into one deterministic surface so that the InvariantEngine, ConsistencyEngine,
and OmegaValidator all read from the same truth.
"""

import logging
import numpy as np
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Canonical field registry — every invariant-accessible metric MUST be
# enumerated here.  Adding a new field means ONE edit in this dict.
_FIELD_RESOLVERS: Dict[str, Any] = {}   # populated below


def _resolve_content_density(node: Any) -> float:
    sie = getattr(node, "sie_node", None)
    return float(getattr(sie, "content_density", 0.0)) if sie else 0.0


def _resolve_s_sie(node: Any) -> float:
    sie = getattr(node, "sie_node", None)
    return float(getattr(sie, "s_sie", 0.0)) if sie else 0.0


def _resolve_composite_quality_score(node: Any) -> float:
    sie = getattr(node, "sie_node", None)
    return float(getattr(sie, "composite_quality_score", 0.0)) if sie else 0.0


def _resolve_mode_scaling_factor(node: Any) -> float:
    sie = getattr(node, "sie_node", None)
    return float(getattr(sie, "mode_scaling_factor", 1.0)) if sie else 1.0


def _resolve_c_node(node: Any) -> float:
    ep = getattr(node, "epistemic", None)
    return float(getattr(ep, "c_node", 0.0)) if ep else 0.0


def _resolve_depth(node: Any) -> float:
    ep = getattr(node, "epistemic", None)
    return float(getattr(ep, "depth", 0)) if ep else 0.0


def _resolve_v_score(node: Any) -> float:
    return float(getattr(node, "v_score", 0.0))


def _resolve_constraint_count(node: Any) -> float:
    constraints = getattr(node, "constraints", None)
    if constraints and isinstance(constraints, (list, tuple)):
        return float(len(constraints))
    return 0.0


def _resolve_alignment_vector_norm(node: Any) -> float:
    sie = getattr(node, "sie_node", None)
    if sie is not None:
        av = getattr(sie, "alignment_vector", None)
        if av and isinstance(av, (list, tuple)):
            return float(np.linalg.norm(av))
    return 0.0


# --- Field registry ---
_FIELD_RESOLVERS = {
    "content_density": _resolve_content_density,
    "rho": _resolve_content_density,                    # backward-compatible alias
    "s_sie": _resolve_s_sie,
    "composite_quality_score": _resolve_composite_quality_score,
    "J_info": _resolve_composite_quality_score,          # backward-compatible alias
    "j_info": _resolve_composite_quality_score,          # case-insensitive alias
    "mode_scaling_factor": _resolve_mode_scaling_factor,
    "kappa": _resolve_mode_scaling_factor,               # backward-compatible alias
    "c_node": _resolve_c_node,
    "depth": _resolve_depth,
    "v_score": _resolve_v_score,
    "constraint_count": _resolve_constraint_count,
    "alignment_vector_norm": _resolve_alignment_vector_norm,
    "phase_gradient_norm": _resolve_alignment_vector_norm,  # backward-compatible alias
}

# Canonical set of allowed field names (used by InvariantEngine for validation)
CANONICAL_FIELDS = frozenset({
    "content_density", "rho", "s_sie", "composite_quality_score", "J_info",
    "mode_scaling_factor", "kappa", "c_node",
    "depth", "v_score", "constraint_count",
    "alignment_vector_norm", "phase_gradient_norm",
})


class SIEProjection:
    """
    Project an AletheiaSkill / CognitiveNode into a flat metric dictionary.

    Usage::

        proj = SIEProjection()
        metrics = proj.project(node)         # all fields
        rho = proj.resolve(node, "rho")      # single field
    """

    @staticmethod
    def project(node: Any) -> Dict[str, float]:
        """Return the full canonical projection for *node*."""
        return {name: resolver(node) for name, resolver in _FIELD_RESOLVERS.items()}

    @staticmethod
    def resolve(node: Any, field_name: str) -> float:
        """Resolve a single canonical field, or 0.0 if unknown."""
        resolver = _FIELD_RESOLVERS.get(field_name)
        if resolver is None:
            # Try case-insensitive lookup
            resolver = _FIELD_RESOLVERS.get(field_name.lower())
        if resolver is None:
            logger.warning(f"SIEProjection: unknown field {field_name!r}")
            return 0.0
        return resolver(node)
