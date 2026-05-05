"""
Aletheia Branch Schema Contracts v2.0
======================================
Pydantic v2 schemas for branch-aware theorist SFT training records.

These contracts govern the training data layer only.
For DAG runtime contracts (TheoristOutput, CodingOutput, etc.) see
src/pipeline/contracts.py — those are separate systems.
"""

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Thresholds — kept as module constants so tests and callers can patch them.
# ---------------------------------------------------------------------------
MIN_BRANCH_COUNT: int = 2
BRANCH_MECHANISM_SIMILARITY_MAX: float = 0.70
REQUIRE_FAILURE_BOUNDARY: bool = True


# ---------------------------------------------------------------------------
# Schema models
# ---------------------------------------------------------------------------

class Branch(BaseModel):
    branch_id: str = Field(..., min_length=1,
                           description="Short unique label, e.g. 'upward_shift'")
    condition: str = Field(..., min_length=10,
                           description="Trigger condition: must begin with 'If' or 'When'")
    mechanism: str = Field(..., min_length=20,
                           description="Mechanistic explanation at molecular or atomic level")
    first_order_effect: str = Field(..., min_length=10,
                                    description="Immediate consequence of this branch firing")
    second_order_effect: str = Field(..., min_length=10,
                                     description="Downstream or cascading consequence")
    failure_boundary: str = Field(default="",
                                  description="Where this branch logic fails or breaks down")
    assumption_risk: Literal["low", "medium", "high"] = "medium"
    evidence_refs: List[str] = Field(default_factory=list,
                                     description="Source doc IDs or concept names supporting this branch")

    model_config = {"extra": "forbid", "str_strip_whitespace": True}

    @model_validator(mode="after")
    def check_failure_boundary_when_required(self) -> "Branch":
        if REQUIRE_FAILURE_BOUNDARY and not self.failure_boundary.strip():
            raise ValueError(
                f"Branch '{self.branch_id}' requires a non-empty failure_boundary "
                f"(REQUIRE_FAILURE_BOUNDARY=True)"
            )
        return self


class BranchTrace(BaseModel):
    schema_version: Literal["2.0"]
    baseline_state: str = Field(..., min_length=20,
                                description="State of the system before any branch fires")
    branches: List[Branch] = Field(..., min_length=MIN_BRANCH_COUNT,
                                   description="At least MIN_BRANCH_COUNT distinct branches required")
    synthesis: str = Field(..., min_length=20,
                           description="Integrating conclusion across all branches")
    final_answer: str = Field(..., min_length=50,
                              description="50-120 word prose summary of the mechanism")

    model_config = {"extra": "forbid", "str_strip_whitespace": True}


# ---------------------------------------------------------------------------
# Structural evaluator — deterministic, no LLM calls.
# ---------------------------------------------------------------------------

def _jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union else 0.0


def evaluate_branch_topology(bt: BranchTrace) -> dict:
    """Score the structural quality of a BranchTrace.

    Returns a dict with 'pass' bool and per-metric values.
    All metrics are deterministic — no LLM calls.

    Failure reasons map to structured rejection codes suitable for
    telemetry and future DPO pair construction:
        single_branch_only         — fewer than MIN_BRANCH_COUNT branches
        weak_mechanism_diversity   — any two branches share >BRANCH_MECHANISM_SIMILARITY_MAX Jaccard
        missing_failure_boundary   — one or more branches have empty failure_boundary
        incomplete_branch_conditions — fewer than 80% of branches have a non-trivial condition
    """
    branches = bt.branches
    n = len(branches)

    if n < MIN_BRANCH_COUNT:
        return {
            "pass": False,
            "reason": "single_branch_only",
            "branch_count": n,
            "failure_boundary_coverage": 0.0,
            "max_mechanism_similarity": 0.0,
            "condition_coverage": 0.0,
        }

    # Failure boundary coverage
    fb_coverage = sum(1 for b in branches if b.failure_boundary.strip()) / n

    # Mechanism diversity: worst-case pairwise Jaccard across all branch pairs.
    max_sim = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            sim = _jaccard(branches[i].mechanism, branches[j].mechanism)
            if sim > max_sim:
                max_sim = sim

    # Condition completeness: branches must have substantive trigger conditions (>= 20 chars).
    # Pydantic enforces min_length=10, so this threshold catches terse but technically-valid conditions.
    condition_coverage = sum(1 for b in branches if len(b.condition.strip()) >= 20) / n

    fake_contrast = max_sim > BRANCH_MECHANISM_SIMILARITY_MAX
    fb_fail = REQUIRE_FAILURE_BOUNDARY and fb_coverage < 1.0
    condition_fail = condition_coverage < 0.8

    passed = not fake_contrast and not fb_fail and not condition_fail

    reason: Optional[str] = None
    if fake_contrast:
        reason = "weak_mechanism_diversity"
    elif fb_fail:
        reason = "missing_failure_boundary"
    elif condition_fail:
        reason = "incomplete_branch_conditions"

    return {
        "pass": passed,
        "reason": reason,
        "branch_count": n,
        "failure_boundary_coverage": round(fb_coverage, 3),
        "max_mechanism_similarity": round(max_sim, 3),
        "condition_coverage": round(condition_coverage, 3),
    }
