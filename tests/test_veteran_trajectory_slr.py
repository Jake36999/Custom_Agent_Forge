"""
Veteran Trajectory SLR Regression Tests
========================================

Protects the end-to-end wiring documented in
``DAG_Engine_Veteran_Trajectory_SLR_Update.md``:

    _wrap_veteran_payload / _enrich_veteran_payload  ──►  reasoning_trajectory
        {confusion_matrix, response_vector}  ──►  acs_engine.compute_slr()
            ──►  compute_tikhonov_slr()  (primary, not cosine fallback)

If any refactor silently drops the confusion_matrix / response_vector build,
or rewires ``compute_slr`` so that the Tikhonov branch stops firing on valid
veteran trajectories, these tests must break. That is intentional: the
Veteran extraction path's entire point is to feed the Tikhonov SLR.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipeline.dag_scoring_pass import _wrap_veteran_payload
from src.pipeline.acs_engine import SycophancyDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_veteran_payload() -> dict:
    """A canonical, non-degenerate veteran extraction payload."""
    return {
        "node_id": "vet_regression_001",
        "error_traceback": (
            "Traceback (most recent call last):\n"
            '  File "x.py", line 3, in <module>\n'
            "    raise ValueError('boom')\n"
            "ValueError: boom"
        ),
        "successful_diff": "- raise ValueError('boom')\n+ return 42",
        "resolved_code": "def fixed():\n    return 42\n",
        "execution_pattern": ["attempt", "error", "diagnosis", "fix"],
    }


# ---------------------------------------------------------------------------
# Producer: _wrap_veteran_payload must emit a Tikhonov-compatible trajectory
# ---------------------------------------------------------------------------

class TestVeteranWrapEmitsTrajectory:
    """Guard the producer half of the contract."""

    def test_wrap_attaches_reasoning_trajectory(self):
        wrapped = _wrap_veteran_payload(_make_veteran_payload(), "veteran_fixture.py")
        trajectory = wrapped["semantics"]["reasoning_trajectory"]
        assert isinstance(trajectory, dict), (
            "reasoning_trajectory must be attached by _wrap_veteran_payload"
        )

    def test_confusion_matrix_is_non_empty_2d_list(self):
        wrapped = _wrap_veteran_payload(_make_veteran_payload(), "veteran_fixture.py")
        cm = wrapped["semantics"]["reasoning_trajectory"]["confusion_matrix"]
        assert cm is not None, "confusion_matrix must not be None"
        assert isinstance(cm, list) and len(cm) > 0, "confusion_matrix must be non-empty"
        assert all(isinstance(row, list) for row in cm), (
            "confusion_matrix must be a 2D list (list of rows)"
        )
        # Each row should encode exactly four signals:
        #   [has_traceback, has_diff, has_resolution, step_confidence]
        assert all(len(row) == 4 for row in cm), (
            "each confusion_matrix row must encode four signals"
        )

    def test_response_vector_aligns_with_confusion_matrix(self):
        wrapped = _wrap_veteran_payload(_make_veteran_payload(), "veteran_fixture.py")
        traj = wrapped["semantics"]["reasoning_trajectory"]
        cm = traj["confusion_matrix"]
        rv = traj["response_vector"]
        assert rv is not None, "response_vector must not be None"
        assert isinstance(rv, list) and len(rv) > 0, "response_vector must be non-empty"
        assert len(rv) == len(cm), (
            f"response_vector length ({len(rv)}) must match confusion_matrix "
            f"row count ({len(cm)})"
        )

    def test_wrap_records_signal_metadata(self):
        """has_traceback / has_diff / has_resolution metadata must reflect the payload."""
        wrapped = _wrap_veteran_payload(_make_veteran_payload(), "veteran_fixture.py")
        meta = wrapped["semantics"]["reasoning_trajectory"]["metadata"]
        assert meta["has_traceback"] is True
        assert meta["has_diff"] is True
        assert meta["has_resolution"] is True


# ---------------------------------------------------------------------------
# Consumer: acs_engine.compute_slr must dispatch to the Tikhonov branch
# ---------------------------------------------------------------------------

class TestComputeSlrDispatchesTikhonov:
    """Guard the consumer half of the contract."""

    def test_compute_slr_returns_bounded_value_for_veteran_trajectory(self):
        wrapped = _wrap_veteran_payload(_make_veteran_payload(), "veteran_fixture.py")
        traj = wrapped["semantics"]["reasoning_trajectory"]
        detector = SycophancyDetector(sensitivity=0.05)
        slr = detector.compute_slr(
            confusion_matrix=traj["confusion_matrix"],
            response_vector=traj["response_vector"],
        )
        assert isinstance(slr, float)
        assert 0.0 <= slr <= 1.0, f"SLR must be in [0,1], got {slr}"

    def test_compute_slr_routes_through_tikhonov_not_cosine(self):
        """The Tikhonov branch must fire when both signals are present.

        We patch both implementations on the *instance* so we can observe
        which branch ``compute_slr`` chose without relying on numerical
        side effects.
        """
        wrapped = _wrap_veteran_payload(_make_veteran_payload(), "veteran_fixture.py")
        traj = wrapped["semantics"]["reasoning_trajectory"]
        detector = SycophancyDetector(sensitivity=0.05)

        with patch.object(
            detector, "compute_tikhonov_slr", wraps=detector.compute_tikhonov_slr
        ) as tikhonov_spy, patch.object(
            detector, "compute_cosine_slr", wraps=detector.compute_cosine_slr
        ) as cosine_spy:
            detector.compute_slr(
                confusion_matrix=traj["confusion_matrix"],
                response_vector=traj["response_vector"],
            )
            assert tikhonov_spy.call_count == 1, (
                "compute_slr must dispatch to compute_tikhonov_slr when a "
                "veteran trajectory is supplied"
            )
            assert cosine_spy.call_count == 0, (
                "compute_slr must NOT fall through to compute_cosine_slr "
                "when a valid 2D confusion_matrix is available"
            )
