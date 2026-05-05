import sys
from pathlib import Path

# Ensure project root is importable for absolute src imports.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipeline.dataset_formatter import (
    apply_mode_balance_weights,
    compute_mode_balance_factors,
)


def test_compute_mode_balance_factors_clips_bounds_for_skewed_modes() -> None:
    items = [
        {"_mode": "advocate"},
        {"_mode": "advocate"},
        {"_mode": "advocate"},
        {"_mode": "advocate"},
        {"_mode": "veteran"},
    ]

    factors = compute_mode_balance_factors(items)

    assert factors["advocate"] == 0.75
    assert factors["veteran"] == 1.25


def test_apply_mode_balance_weights_updates_sample_weight_and_metadata() -> None:
    items = [
        {"_mode": "advocate", "_sample_weight": 1.0},
        {"_mode": "advocate", "_sample_weight": 0.8},
        {"_mode": "advocate", "_sample_weight": 0.6},
        {"_mode": "advocate", "_sample_weight": 0.4},
        {"_mode": "veteran", "_sample_weight": 0.5},
    ]

    apply_mode_balance_weights(items, process_id="MODE-TEST")

    advocate_weights = [i["_sample_weight"] for i in items if i["_mode"] == "advocate"]
    veteran_weights = [i["_sample_weight"] for i in items if i["_mode"] == "veteran"]

    assert all(i["_mode_balance_factor"] == 0.75 for i in items if i["_mode"] == "advocate")
    assert all(i["_mode_balance_factor"] == 1.25 for i in items if i["_mode"] == "veteran")

    assert advocate_weights == [0.75, 0.6, 0.45, 0.3]
    assert veteran_weights == [0.625]
