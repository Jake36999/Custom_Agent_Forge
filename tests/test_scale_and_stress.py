"""
Priority V2.1 — Scale & Stress Tests (Deployment Readiness)
============================================================

These tests validate that the Cryptographic State Transition Ledger
introduced in Priority 3 scales cleanly to production-class graph
sizes without breaking the SHA256 hash chain, blowing up memory, or
degrading throughput.

The target of this harness is **Deliverable 1** from the V2.1 Pre-
Production roadmap:

    * 10,000 synthetic nodes
    * 4 transitions each (CREATED -> VALIDATED -> SCORED -> ACCEPTED)
    * 40,000+ sequential hashes recorded into `_transition_ledger`
    * Hash chain verifies end-to-end via `verify_transition_ledger()`
    * Elapsed wall-clock time + peak RAM captured via `tracemalloc`

If this test crashes or exceeds the memory budget, the Priority 3
ledger must be upgraded to chunked disk-backed JSONL persistence
before V2.1 can ship.
"""

import gc
import sys
import time
import tracemalloc
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipeline.dag_runtime import (
    DAGRuntime,
    InMemoryEpistemicGraph,
    NodeState,
    TransitionTrigger,
)


# ---------------------------------------------------------------------------
# Soft budgets (violations are warnings, not hard failures — the test's job
# is to *measure* and flag, not to gate on machine-dependent timings)
# ---------------------------------------------------------------------------

SCALE_NODE_COUNT           = 10_000
TRANSITIONS_PER_NODE       = 4
EXPECTED_TRANSITION_COUNT  = SCALE_NODE_COUNT * TRANSITIONS_PER_NODE  # 40,000
SOFT_WALLCLOCK_BUDGET_SEC  = 30.0      # 30s is a very generous ceiling
SOFT_PEAK_MEMORY_MB_BUDGET = 512       # 512 MB peak ledger footprint


def _make_empty_runtime() -> DAGRuntime:
    """Spin up a DAGRuntime without going through ingest / Pydantic."""
    graph = InMemoryEpistemicGraph([])
    acs = MagicMock(spec=[])
    return DAGRuntime(graph, acs, mode="coding_assistant")


# ===========================================================================
# Stress Harness — 10,000 synthetic nodes, 40,000 transitions
# ===========================================================================

class TestCryptographicLedgerScale:
    """Validates the hash-chain ledger under production-class load."""

    def test_ledger_scales_to_10k_nodes_40k_transitions(self, capsys):
        """
        Simulated run: drive 10,000 nodes through the full lifecycle
        (CREATED -> VALIDATED -> SCORED -> ACCEPTED), recording every
        transition in the cryptographic ledger.

        We call ``_log_transition`` directly rather than spinning up
        the full pipeline (SIE/ACS/Tikhonov/etc.) because this test's
        concern is the **ledger's** scaling behavior in isolation. The
        full-pipeline throughput is covered by other regression files.
        """
        runtime = _make_empty_runtime()

        # --- Capture baseline memory before the stress run ---
        gc.collect()
        tracemalloc.start()
        baseline_snapshot = tracemalloc.take_snapshot()

        t0 = time.perf_counter()
        for i in range(SCALE_NODE_COUNT):
            nid = f"scale_node_{i}"
            runtime._log_transition(
                nid, "NULL", NodeState.CREATED,
                trigger=TransitionTrigger.BOOTSTRAP,
            )
            runtime._log_transition(
                nid, NodeState.CREATED, NodeState.VALIDATED,
                trigger=TransitionTrigger.CONTRACT_PASSED,
            )
            runtime._log_transition(
                nid, NodeState.VALIDATED, NodeState.SCORED,
                trigger=TransitionTrigger.C_FINAL_SCORED,
            )
            runtime._log_transition(
                nid, NodeState.SCORED, NodeState.ACCEPTED,
                trigger=TransitionTrigger.C_FINAL_ACCEPTED,
            )
        elapsed_sec = time.perf_counter() - t0

        # --- Capture peak memory ---
        peak_snapshot = tracemalloc.take_snapshot()
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak_bytes / (1024 * 1024)

        # --- Structural assertions ---
        ledger = runtime.get_transition_ledger()
        assert len(ledger) == EXPECTED_TRANSITION_COUNT, (
            f"expected {EXPECTED_TRANSITION_COUNT} ledger entries, "
            f"got {len(ledger)}"
        )

        # Sequence numbers must be dense and monotonic [1 .. N]
        assert ledger[0]["seq"] == 1
        assert ledger[-1]["seq"] == EXPECTED_TRANSITION_COUNT
        assert runtime._transition_seq == EXPECTED_TRANSITION_COUNT

        # GENESIS anchor still holds at the head of the chain
        assert ledger[0]["prev_hash"] == "GENESIS"

        # --- Cryptographic integrity verification (end-to-end) ---
        t_verify = time.perf_counter()
        chain_ok = runtime.verify_transition_ledger()
        verify_sec = time.perf_counter() - t_verify
        assert chain_ok is True, (
            "verify_transition_ledger() failed on clean 40k-entry chain"
        )

        # --- Ledger-object-level footprint (for reporting) ---
        list_bytes = sys.getsizeof(ledger)
        record_bytes = sum(sys.getsizeof(r) for r in ledger[:100]) * (len(ledger) / 100)
        total_ledger_mb = (list_bytes + record_bytes) / (1024 * 1024)

        throughput = EXPECTED_TRANSITION_COUNT / max(elapsed_sec, 1e-9)

        # --- Report metrics (captured by pytest -s / capsys) ---
        report = (
            f"\n"
            f"=== Cryptographic Ledger Scale Test ===\n"
            f"  Nodes simulated       : {SCALE_NODE_COUNT:,}\n"
            f"  Transitions recorded  : {len(ledger):,}\n"
            f"  Wall-clock elapsed    : {elapsed_sec:.3f} s\n"
            f"  Throughput            : {throughput:,.0f} transitions/sec\n"
            f"  Verify (recompute)    : {verify_sec:.3f} s\n"
            f"  Peak traced memory    : {peak_mb:.2f} MB\n"
            f"  Ledger object footprint: {total_ledger_mb:.2f} MB\n"
            f"  Final chain hash      : {runtime._last_transition_hash[:16]}...\n"
            f"=======================================\n"
        )
        print(report)

        # --- Soft budget assertions (informational, not strict) ---
        assert elapsed_sec < SOFT_WALLCLOCK_BUDGET_SEC, (
            f"10k-node ledger stress test exceeded {SOFT_WALLCLOCK_BUDGET_SEC}s "
            f"budget (took {elapsed_sec:.2f}s) — investigate throughput regression"
        )
        assert peak_mb < SOFT_PEAK_MEMORY_MB_BUDGET, (
            f"Peak memory {peak_mb:.1f}MB exceeded "
            f"{SOFT_PEAK_MEMORY_MB_BUDGET}MB budget — "
            f"consider chunked disk flush for _transition_ledger"
        )

    def test_ledger_hash_chain_is_sequentially_linked_at_scale(self):
        """Every record's prev_hash must equal the previous record's this_hash.

        This is a structural guarantee (not just end-to-end verification):
        if any worker-level race condition were to splice a transition
        into the middle of the chain, this invariant would break.
        """
        runtime = _make_empty_runtime()

        N = 1_000  # Smaller sample for the per-link traversal
        for i in range(N):
            nid = f"chain_node_{i}"
            runtime._log_transition(
                nid, "NULL", NodeState.CREATED,
                trigger=TransitionTrigger.BOOTSTRAP,
            )
            runtime._log_transition(
                nid, NodeState.CREATED, NodeState.VALIDATED,
                trigger=TransitionTrigger.CONTRACT_PASSED,
            )

        ledger = runtime.get_transition_ledger()
        assert len(ledger) == N * 2

        # Walk the full chain record-by-record.
        prev_hash = "GENESIS"
        prev_seq = 0
        for rec in ledger:
            assert rec["prev_hash"] == prev_hash, (
                f"chain break at seq={rec['seq']}: "
                f"expected prev_hash={prev_hash[:12]}..., "
                f"got {rec['prev_hash'][:12]}..."
            )
            assert rec["seq"] == prev_seq + 1, (
                f"seq break: expected {prev_seq + 1}, got {rec['seq']}"
            )
            prev_hash = rec["this_hash"]
            prev_seq = rec["seq"]

    def test_ledger_determinism_at_scale(self):
        """Two independent runtimes driven with identical inputs must
        produce identical final hashes. This is the cross-process
        determinism guarantee that Celery workers will need when we
        eventually serialize the ledger across distributed nodes.
        """
        N = 500

        def build_and_seal() -> str:
            rt = _make_empty_runtime()
            for i in range(N):
                nid = f"det_node_{i:04d}"
                rt._log_transition(
                    nid, "NULL", NodeState.CREATED,
                    trigger=TransitionTrigger.BOOTSTRAP,
                )
                rt._log_transition(
                    nid, NodeState.CREATED, NodeState.VALIDATED,
                    trigger=TransitionTrigger.CONTRACT_PASSED,
                )
                rt._log_transition(
                    nid, NodeState.VALIDATED, NodeState.SCORED,
                    trigger=TransitionTrigger.C_FINAL_SCORED,
                )
                rt._log_transition(
                    nid, NodeState.SCORED, NodeState.ACCEPTED,
                    trigger=TransitionTrigger.C_FINAL_ACCEPTED,
                )
            return rt._last_transition_hash

        head_a = build_and_seal()
        head_b = build_and_seal()
        assert head_a == head_b, (
            "cross-process determinism broken: two identical runs produced "
            f"divergent ledger heads ({head_a[:12]}... vs {head_b[:12]}...)"
        )
