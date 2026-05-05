"""
Aletheia Telemetry Policy — Signal-Driven Governance Actions.

Converts aggregate telemetry observations (SLR distribution, drift velocity,
SIE summary) into governance actions: synthetic drift pushes, global freeze
checks, advocate pressure, and field-pressure computation.

Extracted from dag_runtime.py (Sprint 4b — risk-002).
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Advocate.TelemetryPolicy")

# ── Sentinel codes (mirrored from dag_runtime for standalone use) ────
SENTINEL_TELEMETRY_ALERT = "[DAG_TELEMETRY_THRESHOLD]"
SENTINEL_IDENTITY_FREEZE = "[DAG_IDENTITY_FREEZE]"
SENTINEL_SYSTEM_BACKPRESSURE  = "[DAG_SYSTEM_BACKPRESSURE]"
SENTINEL_LENS_PRESSURE   = "[LENS-PRESSURE]"

# ── Thresholds ───────────────────────────────────────────────────────
TELEMETRY_SLR_MEAN_THRESHOLD = 0.65
TELEMETRY_DRIFT_CUMULATIVE_THRESHOLD = 0.2
TELEMETRY_QUALITY_MEAN_FLOOR = 0.1


def check_telemetry_thresholds(
    telemetry,
    identity_manager,
    acceptance_threshold: float,
    system_backpressure: float,
    non_rejected_ratio: float = 1.0,
) -> Dict[str, Any]:
    """
    Pure-logic telemetry policy evaluation.

    Parameters
    ----------
    telemetry : TelemetryCollector
        The runtime's telemetry collector (must support slr_distribution(),
        drift_velocity(), sie_summary(), record()).
    identity_manager : IdentityManager or None
        Current identity manager (may be mutated: drift_score, mode_weights, frozen).
    acceptance_threshold : float
        Current acceptance threshold (may be lowered by rolling SLR pressure).
    system_backpressure : float
        Current global system backpressure.
    non_rejected_ratio : float
        Fraction of nodes that are NOT rejected (topology integrity proxy).

    Returns
    -------
    dict with keys:
        acceptance_threshold   : float — possibly lowered
        system_backpressure    : float — recomputed
    """
    if not telemetry:
        return {
            "acceptance_threshold": acceptance_threshold,
            "system_backpressure": system_backpressure,
        }

    # --- Aggregate SLR: high mean → synthetic drift increment ---
    slr = telemetry.slr_distribution()
    if slr["count"] >= 3 and slr["mean"] > TELEMETRY_SLR_MEAN_THRESHOLD:
        logger.warning(
            f"{SENTINEL_TELEMETRY_ALERT} Aggregate SLR mean={slr['mean']:.4f} "
            f"> {TELEMETRY_SLR_MEAN_THRESHOLD} — systemic sycophancy detected."
        )
        if identity_manager and not identity_manager.frozen:
            overshoot = slr["mean"] - TELEMETRY_SLR_MEAN_THRESHOLD
            identity_manager.drift_score += overshoot * 0.1
            telemetry.record(
                "identity", "telemetry_drift_push",
                payload={
                    "overshoot": round(overshoot, 4),
                    "new_drift": round(identity_manager.drift_score, 4),
                },
            )

    # --- Cumulative drift velocity → proactive global freeze check ---
    drift = telemetry.drift_velocity()
    if drift["cumulative"] > TELEMETRY_DRIFT_CUMULATIVE_THRESHOLD:
        if identity_manager and not identity_manager.frozen:
            if identity_manager.check_and_enforce("global"):
                logger.critical(
                    f"{SENTINEL_IDENTITY_FREEZE} Telemetry-triggered global freeze "
                    f"(cumulative_drift={drift['cumulative']:.4f})."
                )
                telemetry.record(
                    "identity", "telemetry_global_freeze",
                    payload={"cumulative_drift": drift["cumulative"]},
                )

    # --- SIE aggregate floor → systemic underflow alert ---
    sie = telemetry.sie_summary()
    if sie["count"] >= 3 and sie["mean"] < TELEMETRY_QUALITY_MEAN_FLOOR:
        logger.warning(
            f"{SENTINEL_TELEMETRY_ALERT} Systemic SIE underflow: "
            f"mean={sie['mean']:.4f} < {TELEMETRY_QUALITY_MEAN_FLOOR}."
        )
        telemetry.record(
            "sie", "systemic_underflow_alert",
            payload={"mean": sie["mean"], "count": sie["count"]},
        )

    # --- Phase 3.6: Rolling SLR → advocate pressure + threshold shift ---
    if identity_manager and identity_manager.is_rolling_slr_critical():
        # Boost advocate weight
        identity_manager.mode_weights["advocate"] = min(
            0.5,
            identity_manager.mode_weights.get("advocate", 0.25) + 0.05,
        )
        # Re-normalize
        total_w = sum(identity_manager.mode_weights.values())
        if total_w > 0:
            identity_manager.mode_weights = {
                k: round(v / total_w, 4)
                for k, v in identity_manager.mode_weights.items()
            }
        # Lower acceptance threshold (bounded at 0.20)
        acceptance_threshold = max(0.20, acceptance_threshold - 0.02)
        logger.warning(
            f"{SENTINEL_TELEMETRY_ALERT} Rolling SLR critical — "
            f"advocate_weight={identity_manager.mode_weights.get('advocate', 0):.4f}, "
            f"acceptance_threshold={acceptance_threshold:.4f}"
        )
        telemetry.record(
            "identity", "rolling_slr_pressure",
            payload={
                "rolling_slr_mean": round(identity_manager.get_rolling_slr_mean(), 4),
                "advocate_weight": identity_manager.mode_weights.get("advocate", 0),
                "acceptance_threshold": acceptance_threshold,
            },
        )

    # --- Phase 4.7: Telemetry closed-loop — SLR mean feeds field pressure ---
    if slr["count"] >= 3 and slr["mean"] > 0.6:
        system_backpressure = round(
            min(0.5, system_backpressure + 0.05), 4
        )
        telemetry.record(
            "field", "slr_pressure_injection",
            payload={
                "slr_mean": round(slr["mean"], 4),
                "new_pressure": system_backpressure,
            },
        )

    # --- Global Field Pressure (Patch 4) ---
    # field_coherence = mean(SIE) × (1 - drift_ratio) × topology_integrity
    # topology_floor prevents topology from zeroing out SIE quality in sparse graphs
    # V2.1: Raised floor from 0.1 to 0.5 to prevent the first scoring pass from
    # receiving maximum backpressure when non_rejected_ratio is still 0.
    # Without this, theorist-mode nodes (with inherently lower SIE) are all rejected
    # on pass 1, preventing any node from ever reaching ACCEPTED.
    topology_floor = 0.5
    sie_mean = sie["mean"] if sie["count"] > 0 else 0.5
    drift_ratio = min(1.0, drift["cumulative"] / max(
        TELEMETRY_DRIFT_CUMULATIVE_THRESHOLD, 0.01
    ))
    topology_integrity = max(topology_floor, non_rejected_ratio)

    system_coherence = min(1.0, sie_mean * (1.0 - drift_ratio) * topology_integrity)
    # V2.1: Capped at 0.20 (was 0.5). The 50% cap was designed for catastrophic
    # drift scenarios but routinely fires on first-pass scoring when no nodes
    # have been accepted yet, creating a cold-start death spiral for modes with
    # inherently lower SIE scores (e.g., theorist mode natural language nodes).
    new_pressure = round(min(0.20, max(0.0, 1.0 - system_coherence)), 4)

    if abs(new_pressure - system_backpressure) > 0.01:
        logger.info(
            f"{SENTINEL_SYSTEM_BACKPRESSURE} System backpressure: "
            f"{system_backpressure:.4f} → {new_pressure:.4f} "
            f"(coherence={system_coherence:.4f}, sie_mean={sie_mean:.4f}, "
            f"drift_ratio={drift_ratio:.4f}, topo={topology_integrity:.4f})"
        )
        telemetry.record(
            "field", "pressure_update",
            payload={
                "old_pressure": system_backpressure,
                "new_pressure": new_pressure,
                "system_coherence": round(system_coherence, 4),
                "sie_mean": round(sie_mean, 4),
                "drift_ratio": round(drift_ratio, 4),
                "topology_integrity": round(topology_integrity, 4),
            },
        )

    system_backpressure = new_pressure

    return {
        "acceptance_threshold": acceptance_threshold,
        "system_backpressure": system_backpressure,
    }
