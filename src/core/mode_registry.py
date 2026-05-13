"""Canonical mode names and legacy alias normalization.

Phase 1 only defines the registry. Runtime consumers keep their current mode
strings until later migration phases opt in.
"""

from __future__ import annotations

from enum import Enum
from typing import Union


class CanonicalMode(str, Enum):
    """Canonical mode names for new governance metadata."""

    SOFTWARE_TECH = "software_tech"
    SCHOLAR = "scholar"
    PROJECT_REVIEWER = "project_reviewer"
    MATURED = "matured"


MODE_ALIASES = {
    "coder": CanonicalMode.SOFTWARE_TECH,
    "coding": CanonicalMode.SOFTWARE_TECH,
    "coding_assistant": CanonicalMode.SOFTWARE_TECH,
    "software_tech": CanonicalMode.SOFTWARE_TECH,
    "theorist": CanonicalMode.SCHOLAR,
    "scholar": CanonicalMode.SCHOLAR,
    "advocate": CanonicalMode.PROJECT_REVIEWER,
    "project_manager": CanonicalMode.PROJECT_REVIEWER,
    "project_reviewer": CanonicalMode.PROJECT_REVIEWER,
    "veteran": CanonicalMode.MATURED,
    "experienced": CanonicalMode.MATURED,
    "matured": CanonicalMode.MATURED,
}


ModeInput = Union[str, CanonicalMode]


def normalize_mode(value: ModeInput) -> CanonicalMode:
    """Return the canonical mode enum for a canonical name or legacy alias."""

    if isinstance(value, CanonicalMode):
        return value
    key = str(value).strip().lower()
    try:
        return MODE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown mode: {value!r}") from exc


def canonical_mode_value(value: ModeInput) -> str:
    """Return the canonical mode string for a canonical name or legacy alias."""

    return normalize_mode(value).value


def is_mode(value: ModeInput, canonical: ModeInput) -> bool:
    """Return True when two mode names/aliases resolve to the same canonical mode."""

    try:
        return normalize_mode(value) is normalize_mode(canonical)
    except ValueError:
        return False
