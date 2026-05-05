"""
Invariant Execution Engine v1.1

Parses declarative invariant strings (e.g. "rho > 0.5") into structured
comparisons and evaluates them against AletheiaSkill nodes as hard constraints.

All field resolution is delegated to the SIE Projection Layer so that
invariant evaluation is namespace-agnostic (same truth surface for SIE,
ACS, Epistemic, and derived fields).

SECURITY: All parsing uses regex + whitelisted fields/operators.
           NO eval(), NO exec(), NO ast.literal_eval() on user strings.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from src.pipeline.sie_projection import SIEProjection, CANONICAL_FIELDS

logger = logging.getLogger(__name__)

# --- Whitelisted fields (delegated to SIE Projection canonical set) ---
ALLOWED_FIELDS = CANONICAL_FIELDS

# --- Whitelisted comparison operators ---
ALLOWED_OPS = frozenset({">", ">=", "<", "<=", "==", "!="})

# Regex: <field> <op> <number>
_INVARIANT_RE = re.compile(
    r"^\s*([a-z_][a-z0-9_]*)\s*(>=|<=|!=|==|>|<)\s*(-?[\d]+(?:\.[\d]+)?)\s*$",
    re.IGNORECASE,
)


def _compare(left: float, op: str, right: float) -> bool:
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == "==":
        return abs(left - right) < 1e-9
    if op == "!=":
        return abs(left - right) >= 1e-9
    return False


class InvariantEngine:
    """Parse and validate declarative invariants against epistemic nodes."""

    def parse_invariant(self, invariant_str: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single invariant string into a structured dict.

        Returns ``{"field": str, "op": str, "value": float}`` on success,
        or ``None`` if the string is malformed or references disallowed
        fields / operators.
        """
        if not isinstance(invariant_str, str):
            return None

        m = _INVARIANT_RE.match(invariant_str)
        if not m:
            logger.warning(f"Invariant parse failure (rejected pattern): {invariant_str!r}")
            return None

        field = m.group(1).lower()
        op = m.group(2)
        value = float(m.group(3))

        if field not in ALLOWED_FIELDS:
            logger.warning(f"Invariant parse failure (disallowed field): {field!r}")
            return None
        if op not in ALLOWED_OPS:
            logger.warning(f"Invariant parse failure (disallowed op): {op!r}")
            return None

        return {"field": field, "op": op, "value": value}

    def resolve_field(self, node: Any, field_name: str) -> float:
        """
        Extract a numeric value from an AletheiaSkill node by field name.

        Delegates to the SIE Projection Layer for namespace-agnostic resolution.
        """
        return SIEProjection.resolve(node, field_name)

    def validate_invariant(self, node: Any, parsed: Dict[str, Any]) -> bool:
        """Evaluate a single parsed invariant against a node."""
        value = self.resolve_field(node, parsed["field"])
        return _compare(value, parsed["op"], parsed["value"])

    def validate_all(
        self, node: Any, invariants: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Parse and validate all invariant strings against a node.

        Returns ``(all_passed, list_of_failed_invariant_strings)``.
        Unparseable invariants are treated as failures.
        """
        if not invariants:
            return True, []

        failures: List[str] = []
        for inv_str in invariants:
            parsed = self.parse_invariant(inv_str)
            if parsed is None:
                failures.append(inv_str)
                continue
            if not self.validate_invariant(node, parsed):
                failures.append(inv_str)

        return (len(failures) == 0), failures
