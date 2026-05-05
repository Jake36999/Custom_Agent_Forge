"""
veteran_state_machine.py
Detects error→fix trajectories in a Jupyter cell stream and emits
VeteranPayload dataclasses consumable by veteran_extraction_runner.py.

State transitions:
  IDLE ──(cell with traceback)──► ERROR_DETECTED
  ERROR_DETECTED ──(next code cell, no error)──► EMIT + IDLE
  ERROR_DETECTED ──(another error cell)──► ERROR_DETECTED  (update traceback)
  ERROR_DETECTED ──(markdown)──► stays (markdown is diagnosis commentary)
"""

import uuid
import difflib
from dataclasses import dataclass, field
from typing import List, Iterator, Dict, Any, Optional


@dataclass
class VeteranPayload:
    node_id: str
    error_traceback: str
    resolved_code: str
    attempt_code: str
    successful_diff: str
    execution_pattern: List[str]
    diagnosis_markdown: str = ""
    source_notebook: str = ""


class VeteranStateMachine:
    """
    Single-pass state machine over a Jupyter cell stream.
    Each complete error→fix loop yields one VeteranPayload.
    """

    def process_stream(self, cells: Iterator[Dict[str, Any]]) -> List[VeteranPayload]:
        payloads: List[VeteranPayload] = []

        state = "IDLE"
        error_cell_source: str = ""
        traceback_text: str = ""
        diagnosis_parts: List[str] = []

        for cell in cells:
            ctype = cell["cell_type"]
            source = cell["source"]
            tb = cell.get("traceback", "")

            if state == "IDLE":
                if ctype == "code" and tb:
                    state = "ERROR_DETECTED"
                    error_cell_source = source
                    traceback_text = tb
                    diagnosis_parts = []

            elif state == "ERROR_DETECTED":
                if ctype == "markdown" and source:
                    diagnosis_parts.append(source)

                elif ctype == "code":
                    if tb:
                        error_cell_source = source
                        traceback_text = tb
                    else:
                        # Recovery cell found — emit payload
                        diff = self._compute_diff(error_cell_source, source)
                        payload = VeteranPayload(
                            node_id=f"vet_{uuid.uuid4().hex[:8]}",
                            error_traceback=traceback_text,
                            resolved_code=source,
                            attempt_code=error_cell_source,
                            successful_diff=diff,
                            execution_pattern=["attempt", "error", "diagnosis", "fix"],
                            diagnosis_markdown="\n\n".join(diagnosis_parts),
                        )
                        payloads.append(payload)
                        state = "IDLE"
                        error_cell_source = ""
                        traceback_text = ""
                        diagnosis_parts = []

        return payloads

    @staticmethod
    def _compute_diff(before: str, after: str) -> str:
        before_lines = before.splitlines(keepends=True)
        after_lines = after.splitlines(keepends=True)
        diff = list(difflib.unified_diff(
            before_lines, after_lines,
            fromfile="attempt", tofile="fix", lineterm=""
        ))
        return "".join(diff) if diff else f"# Fix applied (non-textual change)\n# Before:\n{before}\n# After:\n{after}"
