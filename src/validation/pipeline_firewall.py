import re
from typing import Dict, Any, List
from src.core.models import AletheiaSkill
import tiktoken
import black
import sqlglot

REQUIRED_EPISTEMIC_FIELDS = ["intent", "strategy", "constraints", "execution_pattern", "failure_modes"]
PLACEHOLDER_PATTERNS = [
    r"\[REDACTED\]", r"<insert_key>", r"TODO:", r"pass\b", r"FIXME:"
]

# ===================================================================
# SEMANTIC DRIFT FIREWALL — D10
# Forbidden token patterns that indicate cross-domain contamination
# or hallucinated pseudo-physics leaking into production cognition.
#
# URF pseudoscience brand names — no legitimate use in any context.
# Compiler-internal phrases — certification concepts that should
# never appear in training output directed at downstream models.
#
# NOTE: Internal pipeline field names (content_density, alignment_vector,
# composite_quality_score, mode_scaling_factor, s_sie) are NOT blocked here.
# The pipeline must be able to compile legitimate physics/engineering stacks.
# Legacy physics aliases (phase_gradient, waveform_amplitude, resonance_density,
# rho, J_info, kappa) are deprecated but not firewall-blocked to avoid
# breaking archived telemetry deserialization.
# ===================================================================
FORBIDDEN_TOKENS: List[str] = [
    # URF brand tokens
    "IRER",
    "resonance field",
    "SSE 0.00087",
    "nura physics",
    # Compiler-internal self-description
    "omega handshake",
    "bounded inevitability",
]


class DriftViolation(Exception):
    """Raised when forbidden semantic tokens are detected in node content."""
    def __init__(self, token: str, context: str = ""):
        self.token = token
        self.context = context
        super().__init__(f"DriftViolation: forbidden token '{token}' detected{f' in {context}' if context else ''}")


def enforce_semantic_firewall(text: str, context: str = "") -> None:
    """
    Scans text for forbidden tokens. Raises DriftViolation on first match.
    Case-insensitive scan to catch all drift variants.

    Parameters
    ----------
    text : str
        The text to scan for forbidden tokens.
    context : str
        Optional context label (e.g. "node:abc") for error reporting.
    """
    if not isinstance(text, str) or not text:
        return
    text_lower = text.lower()
    for token in FORBIDDEN_TOKENS:
        if token.lower() in text_lower:
            raise DriftViolation(token, context)


def is_canonical(code: str) -> bool:
    """Checks if code is canonicalized (Python: black, SQL: sqlglot)."""
    code = code.strip()
    if code.startswith("SELECT") or code.startswith("WITH"):
        try:
            canonical = sqlglot.transpile(code)[0].strip()
            return code == canonical
        except Exception:
            return False
    try:
        canonical = black.format_str(code, mode=black.Mode())
        return code == canonical.strip()
    except Exception:
        return False


def validate_skill(skill: AletheiaSkill) -> Dict[str, Any]:
    diagnostics = {"status": "accepted", "reason": "all_invariants_passed", "details": {}}
    missing = []
    reasoning = getattr(getattr(skill, "teaching_layer", None), "reasoning_vectors", None)
    if not reasoning:
        diagnostics["status"] = "rejected"
        diagnostics["reason"] = "missing_epistemic_core"
        diagnostics["details"] = {"missing_fields": REQUIRED_EPISTEMIC_FIELDS, "node_id": getattr(skill, "node_id", None)}
        return diagnostics
    for field in REQUIRED_EPISTEMIC_FIELDS:
        value = getattr(reasoning, field, None)
        if value is None or (isinstance(value, (str, list)) and (not value or value == "unknown" or (isinstance(value, list) and (not value or all(v == "unknown" for v in value))))):
            missing.append(field)
    if missing:
        diagnostics["status"] = "rejected"
        diagnostics["reason"] = "incomplete_epistemic_core"
        diagnostics["details"] = {"missing_fields": missing, "node_id": getattr(skill, "node_id", None)}
        return diagnostics
    # Canonicalization
    code = getattr(skill, "code_snippet", "")
    if not is_canonical(code):
        diagnostics["status"] = "rejected"
        diagnostics["reason"] = "non_canonical_code"
        diagnostics["details"] = {"node_id": getattr(skill, "node_id", None)}
        return diagnostics
    # Token limit
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(str(skill))
    if len(tokens) >= 2048:
        diagnostics["status"] = "rejected"
        diagnostics["reason"] = "token_limit_exceeded"
        diagnostics["details"] = {"token_count": len(tokens), "node_id": getattr(skill, "node_id", None)}
        return diagnostics
    # Placeholder scan
    for pat in PLACEHOLDER_PATTERNS:
        if re.search(pat, code):
            diagnostics["status"] = "rejected"
            diagnostics["reason"] = "placeholder_detected"
            diagnostics["details"] = {"pattern": pat, "node_id": getattr(skill, "node_id", None)}
            return diagnostics
    diagnostics["details"]["token_count"] = len(tokens)
    return diagnostics
