"""
Aletheia Execution Contracts v2.0
Strict Pydantic v2 schemas for each Triad execution mode.
Every DAG node output must pass validate_contract() before pipeline progression.

All contracts inherit BaseAletheiaModel (extra="forbid", strict=True) to prevent
silent schema drift. Invalid cognition is eliminated at the source.
"""

from typing import Any, Dict, List, Optional, Tuple, Type
from pydantic import Field, ValidationError
from src.core.models_contract import BaseAletheiaModel
from src.validation.pipeline_firewall import enforce_semantic_firewall, DriftViolation


class TheoristOutput(BaseAletheiaModel):
    """Contract for Theorist mode: formal design patterns and epistemic axioms."""
    constraints: List[str] = Field(..., min_length=1)
    dependencies: List[str] = Field(default_factory=list)
    invariants: List[str] = Field(default_factory=list)
    interfaces: List[str] = Field(default_factory=list)


class CodingOutput(BaseAletheiaModel):
    """Contract for Coding Assistant mode: implementation + tests."""
    implementation: str = Field(..., min_length=1)
    tests: Optional[str] = None
    coverage_map: Optional[Dict[str, Any]] = None


class AdvocateOutput(BaseAletheiaModel):
    """Contract for Advocate mode: theory paired with implementation."""
    theory: str = Field(..., min_length=1)
    implementation: str = Field(..., min_length=1)
    validation_criteria: List[str] = Field(default_factory=list)


class VeteranOutput(BaseAletheiaModel):
    """Contract for Veteran mode: diagnostic debugging loop."""
    traceback: str = Field(..., min_length=1)
    diff: str = Field(..., min_length=1)
    resolved_code: str = Field(..., min_length=1)


# Map mode names to their contract schemas
MODE_CONTRACTS: Dict[str, Type[BaseAletheiaModel]] = {
    "theorist": TheoristOutput,
    "coding_assistant": CodingOutput,
    "advocate": AdvocateOutput,
    "veteran": VeteranOutput,
}


def validate_contract(output: Dict[str, Any], schema: Type[BaseAletheiaModel]) -> Tuple[bool, Optional[str]]:
    """
    Gate function: validates a mode output against its strict contract.
    Returns (True, None) on success, (False, error_description) on failure.
    Uses BaseAletheiaModel (extra='forbid', strict=True) — extra fields
    and type mismatches are hard failures.
    """
    try:
        schema.model_validate(output)
        # Post-validation drift scan: reject contaminated content
        import json
        text = json.dumps(output, sort_keys=True, default=str)
        try:
            enforce_semantic_firewall(text, context=f"contract:{schema.__name__}")
        except DriftViolation as dv:
            return (False, f"DriftViolation: forbidden token '{dv.token}' in contract output")
        return (True, None)
    except ValidationError as e:
        return (False, str(e))
