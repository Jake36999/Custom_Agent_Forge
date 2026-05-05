"""
Aletheia Telemetry Pipeline v1.0
Centralized metric aggregation for SLR distribution, identity drift velocity,
mode usage balancing, and SIE underflow tracking.

Replaces ad-hoc logger.warning() calls with a structured, queryable audit trail
that can be persisted to disk for longitudinal analysis.
"""

import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Advocate.Telemetry")


class TelemetryEvent:
    """Single typed telemetry event."""
    __slots__ = ("timestamp", "category", "event_type", "node_id", "payload")

    def __init__(self, category: str, event_type: str, node_id: str = "", payload: Optional[Dict[str, Any]] = None):
        self.timestamp = time.time()
        self.category = category
        self.event_type = event_type
        self.node_id = node_id
        self.payload = payload or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "category": self.category,
            "event_type": self.event_type,
            "node_id": self.node_id,
            "payload": self.payload,
        }


class TelemetryCollector:
    """
    Centralized telemetry aggregator.

    Categories:
        - slr: Sycophancy Leakage Rate events (breach detected, conviction score)
        - identity: Identity drift events (drift increment, freeze, revert)
        - mode: Mode usage events (execution mode per node)
        - sie: SIE computation events (underflow, scores)
        - contract: Contract validation events (pass/fail per mode)
        - reroll: Reroll events (reason, attempt number)

    Usage:
        tc = TelemetryCollector()
        tc.record("slr", "breach_detected", node_id="abc", payload={"slr": 0.72, "conviction": 0.28})
        tc.record("mode", "execution", node_id="abc", payload={"mode": "advocate"})
        snapshot = tc.snapshot()
        tc.persist("output/telemetry.jsonl")
    """

    def __init__(self):
        self._events: List[TelemetryEvent] = []
        self._mode_counter: Counter = Counter()
        self._slr_values: List[float] = []
        self._drift_velocity: List[float] = []
        self._sie_scores: List[float] = []
        self._runtime_durations_ms: List[float] = []

    def record(self, category: str, event_type: str, node_id: str = "", payload: Optional[Dict[str, Any]] = None) -> None:
        """Record a telemetry event and update running aggregates."""
        evt = TelemetryEvent(category, event_type, node_id, payload)
        self._events.append(evt)

        p = payload or {}
        if category == "slr" and "slr" in p:
            self._slr_values.append(float(p["slr"]))
        if category == "identity" and "drift_increment" in p:
            self._drift_velocity.append(float(p["drift_increment"]))
        if category == "mode" and "mode" in p:
            self._mode_counter[p["mode"]] += 1
        if category == "sie" and "s_sie" in p:
            self._sie_scores.append(float(p["s_sie"]))
        if category == "runtime" and "elapsed_ms" in p:
            self._runtime_durations_ms.append(float(p["elapsed_ms"]))

    def slr_distribution(self) -> Dict[str, Any]:
        """Returns SLR distribution statistics."""
        if not self._slr_values:
            return {"count": 0, "mean": 0.0, "max": 0.0, "min": 0.0}
        import statistics
        return {
            "count": len(self._slr_values),
            "mean": round(statistics.mean(self._slr_values), 4),
            "max": round(max(self._slr_values), 4),
            "min": round(min(self._slr_values), 4),
            "stdev": round(statistics.stdev(self._slr_values), 4) if len(self._slr_values) > 1 else 0.0,
        }

    def drift_velocity(self) -> Dict[str, Any]:
        """Returns identity drift velocity statistics."""
        if not self._drift_velocity:
            return {"count": 0, "cumulative": 0.0, "mean_increment": 0.0}
        return {
            "count": len(self._drift_velocity),
            "cumulative": round(sum(self._drift_velocity), 4),
            "mean_increment": round(sum(self._drift_velocity) / len(self._drift_velocity), 4),
        }

    def mode_balance(self) -> Dict[str, int]:
        """Returns mode usage counts for balance analysis."""
        return dict(self._mode_counter)

    def sie_summary(self) -> Dict[str, Any]:
        """Returns SIE score distribution statistics."""
        if not self._sie_scores:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
        import statistics
        return {
            "count": len(self._sie_scores),
            "mean": round(statistics.mean(self._sie_scores), 4),
            "min": round(min(self._sie_scores), 4),
            "max": round(max(self._sie_scores), 4),
        }

    def runtime_summary(self) -> Dict[str, Any]:
        """Returns DAG runtime duration statistics in milliseconds."""
        if not self._runtime_durations_ms:
            return {"count": 0, "mean_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
        import statistics
        return {
            "count": len(self._runtime_durations_ms),
            "mean_ms": round(statistics.mean(self._runtime_durations_ms), 3),
            "min_ms": round(min(self._runtime_durations_ms), 3),
            "max_ms": round(max(self._runtime_durations_ms), 3),
        }

    def snapshot(self) -> Dict[str, Any]:
        """Returns a full telemetry snapshot for audit."""
        return {
            "total_events": len(self._events),
            "slr_distribution": self.slr_distribution(),
            "drift_velocity": self.drift_velocity(),
            "mode_balance": self.mode_balance(),
            "sie_summary": self.sie_summary(),
            "runtime_summary": self.runtime_summary(),
        }

    def persist(self, output_path: str) -> None:
        """Persist all events to a JSONL file for longitudinal analysis."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for evt in self._events:
                f.write(json.dumps(evt.to_dict(), sort_keys=True) + "\n")
            # Append summary as final line
            f.write(json.dumps({"_summary": self.snapshot()}, sort_keys=True) + "\n")
        logger.info(f"[Telemetry] Persisted {len(self._events)} events to {output_path}")

    def reset(self) -> None:
        """Clear all accumulated telemetry."""
        self._events.clear()
        self._mode_counter.clear()
        self._slr_values.clear()
        self._drift_velocity.clear()
        self._sie_scores.clear()
        self._runtime_durations_ms.clear()
