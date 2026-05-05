"""
Aletheia Identity Manager v2.0
Tracks agent behavioral drift across DAG execution cycles.
If drift exceeds threshold, freezes the branch and reverts to stable state.
Supports trajectory-driven identity updates, version management, and KL divergence drift detection.
"""

import logging
import hashlib
import json
import math
from typing import Dict, List, Optional

logger = logging.getLogger("Advocate.IdentityManager")

try:
    from src.core.models import IdentityState
except ImportError:
    IdentityState = None


class IdentityManager:
    """
    Manages the agent's semantic identity state across the DAG.
    Detects behavioral drift from SLR breaches and enforces containment.
    Backed by an IdentityState Pydantic model for validated state management.
    """

    # HD3: Tikhonov-regularized asymptotic recovery rate.
    # λ acts as a damping factor preventing rapid, unstable resets.
    # drift_score *= (1 - λ / (1 + drift_score)) per recovery tick.
    TIKHONOV_RECOVERY_LAMBDA = 0.05

    def __init__(self, stable_hash_ref: str = "", max_drift: float = 0.3, telemetry=None):
        self.max_drift = max_drift
        self.telemetry = telemetry

        # Core identity state — backed by Pydantic model when available
        if IdentityState is not None:
            self._state = IdentityState(
                behavior_profile={
                    "theorist": 0.0, "coding_assistant": 0.0,
                    "advocate": 0.0, "veteran": 0.0,
                },
                mode_weights={
                    "theorist": 0.25, "coding_assistant": 0.25,
                    "advocate": 0.25, "veteran": 0.25,
                },
                drift_score=0.0,
                stable_hash_reference=stable_hash_ref,
                frozen=False,
                capabilities={},
                version=1,
            )
        else:
            self._state = None

        # Operational state (not part of identity model)
        self.slr_breach_count = 0
        self._behavior_history: List[Dict[str, float]] = []
        self.slr_history: List[float] = []
        self._slr_history_cap = 50
        self._slr_rolling_threshold = 0.65
        self._recovery_lambda_override: Optional[float] = None

        # Store stable snapshot for revert
        self._stable_state: Optional[object] = None
        if self._state is not None:
            self._stable_state = self._state.model_copy(deep=True)

    # --- Property proxies for backward compatibility ---
    @property
    def drift_score(self) -> float:
        return self._state.drift_score if self._state else 0.0

    @drift_score.setter
    def drift_score(self, value: float):
        if self._state:
            self._state.drift_score = value

    @property
    def frozen(self) -> bool:
        return self._state.frozen if self._state else False

    @frozen.setter
    def frozen(self, value: bool):
        if self._state:
            self._state.frozen = value

    @property
    def behavior_profile(self) -> Dict[str, float]:
        return self._state.behavior_profile if self._state else {}

    @behavior_profile.setter
    def behavior_profile(self, value: Dict[str, float]):
        if self._state:
            self._state.behavior_profile = value

    @property
    def mode_weights(self) -> Dict[str, float]:
        return self._state.mode_weights if self._state else {}

    @mode_weights.setter
    def mode_weights(self, value: Dict[str, float]):
        if self._state:
            self._state.mode_weights = value

    @property
    def stable_hash_reference(self) -> str:
        return self._state.stable_hash_reference if self._state else ""

    @stable_hash_reference.setter
    def stable_hash_reference(self, value: str):
        if self._state:
            self._state.stable_hash_reference = value

    @property
    def version(self) -> int:
        return self._state.version if self._state else 1

    @version.setter
    def version(self, value: int):
        if self._state:
            self._state.version = value

    @property
    def capabilities(self) -> Dict:
        return self._state.capabilities if self._state else {}

    @capabilities.setter
    def capabilities(self, value: Dict):
        if self._state:
            self._state.capabilities = value

    def set_recovery_lambda(self, new_lambda: float) -> None:
        """Set a bounded runtime override for Tikhonov recovery lambda."""
        if not 0.01 <= new_lambda <= 0.15:
            raise ValueError(f"recovery lambda must be in [0.01, 0.15], got {new_lambda}")
        self._recovery_lambda_override = float(new_lambda)

    def get_recovery_lambda(self) -> float:
        """Returns the active lambda value used for recovery ticks."""
        if self._recovery_lambda_override is not None:
            return self._recovery_lambda_override
        return self.TIKHONOV_RECOVERY_LAMBDA

    def create_from_source(self, nodes: list) -> None:
        """
        Bootstrap identity from source analysis (Spec §5).
        Extracts capabilities from a set of AletheiaSkill nodes (or dicts),
        sets stable_hash_reference, and initializes version=1.
        """
        cap_counts: Dict[str, int] = {}
        hash_material = []

        for node in nodes:
            # Accept both AletheiaSkill objects and dicts
            code = ""
            imports = []
            skill_type = "execution"
            node_id = ""
            if isinstance(node, dict):
                code = node.get("code_snippet", "")
                imports = node.get("imports", [])
                skill_type = node.get("skill_type", "execution")
                node_id = node.get("node_id", "")
            else:
                code = getattr(node, "code_snippet", "")
                imports = getattr(node, "imports", [])
                skill_type = getattr(node, "skill_type", "execution")
                node_id = getattr(node, "node_id", "")

            hash_material.append(node_id)
            lowered = (code or "").lower()

            # Capability extraction heuristics
            if imports:
                cap_counts["dependency_resolution"] = cap_counts.get("dependency_resolution", 0) + 1
            if "def " in lowered or "class " in lowered:
                cap_counts["code_generation"] = cap_counts.get("code_generation", 0) + 1
            if any(kw in lowered for kw in ("assert", "test", "raises", "expect")):
                cap_counts["validation"] = cap_counts.get("validation", 0) + 1
            if any(kw in lowered for kw in ("type", "annotation", "hint", "-> ")):
                cap_counts["type_safety"] = cap_counts.get("type_safety", 0) + 1
            if skill_type == "validation":
                cap_counts["validation"] = cap_counts.get("validation", 0) + 1

        self.capabilities = cap_counts
        self.version = 1

        # Snapshot as stable baseline
        if self._state is not None:
            self._stable_state = self._state.model_copy(deep=True)

        # Compute stable hash from behavioral state (canonical format shared with update_from_trajectory)
        state_material = json.dumps({
            "behavior_profile": {k: round(v, 6) for k, v in self.behavior_profile.items()},
            "mode_weights": self.mode_weights,
            "version": self.version,
            "capabilities": dict(self.capabilities),
        }, sort_keys=True)
        self.stable_hash_reference = hashlib.sha256(state_material.encode("utf-8")).hexdigest()

        logger.info(
            f"Identity bootstrapped from {len(nodes)} source nodes — "
            f"capabilities={list(cap_counts.keys())}, hash={self.stable_hash_reference[:12]}..."
        )

    def update_on_slr_breach(self, node_id: str, slr: float, mode: str = "") -> None:
        """
        Called when a sycophancy breach is detected.
        Increments drift_score proportional to SLR magnitude.
        Boosts veteran mode weight and re-normalizes (spec: SLR → mode rebalancing).
        Records SLR into rolling history for temporal trend detection.
        """
        self.slr_breach_count += 1
        drift_increment = slr * 0.1
        self.drift_score += drift_increment
        
        if mode and mode in self.behavior_profile:
            self.behavior_profile[mode] += drift_increment

        # SLR → boost veteran mode weight and re-normalize
        veteran_boost = slr * 0.05  # proportional to SLR severity
        self.mode_weights["veteran"] = self.mode_weights.get("veteran", 0.25) + veteran_boost
        total = sum(self.mode_weights.values())
        if total > 0:
            self.mode_weights = {k: round(v / total, 4) for k, v in self.mode_weights.items()}

        # Rolling SLR history
        self.slr_history.append(slr)
        if len(self.slr_history) > self._slr_history_cap:
            self.slr_history = self.slr_history[-self._slr_history_cap:]

        rolling_slr = self.get_rolling_slr_mean()

        # Drift telemetry: record every increment for governance observability
        if self.telemetry:
            self.telemetry.record(
                "identity", "drift_increment", node_id=node_id,
                payload={
                    "slr": slr,
                    "drift_increment": round(drift_increment, 4),
                    "drift_score": round(self.drift_score, 4),
                    "rolling_slr": round(rolling_slr, 4),
                    "mode": mode,
                },
            )

        logger.warning(
            f"[IdentityManager] SLR breach on {node_id}: slr={slr:.4f}, "
            f"drift_increment={drift_increment:.4f}, total_drift={self.drift_score:.4f}, "
            f"veteran_weight={self.mode_weights.get('veteran', 0):.4f}, "
            f"rolling_slr={rolling_slr:.4f}"
        )

    def get_rolling_slr_mean(self) -> float:
        """Returns the rolling mean of SLR history, or 0.0 if empty."""
        if not self.slr_history:
            return 0.0
        return sum(self.slr_history) / len(self.slr_history)

    def is_rolling_slr_critical(self) -> bool:
        """Returns True if the rolling SLR mean exceeds the threshold (0.65)."""
        return self.get_rolling_slr_mean() >= self._slr_rolling_threshold

    def calculate_drift(self) -> float:
        """Returns the current accumulated drift score."""
        return self.drift_score

    def revert_to_stable(self) -> None:
        """
        Resets identity state to the stable reference snapshot.
        Verifies cryptographic hash integrity before restoring.
        Raises RuntimeError if the stable state has been corrupted (hash mismatch).
        """
        # D8: Cryptographic hash verification before restoring
        if self._stable_state is not None and self.stable_hash_reference:
            verification_material = json.dumps({
                "behavior_profile": {k: round(v, 6) for k, v in self._stable_state.behavior_profile.items()},
                "mode_weights": dict(self._stable_state.mode_weights),
                "version": self._stable_state.version,
                "capabilities": dict(self._stable_state.capabilities),
            }, sort_keys=True)
            computed_hash = hashlib.sha256(verification_material.encode("utf-8")).hexdigest()
            if computed_hash != self.stable_hash_reference:
                raise RuntimeError(
                    f"Identity state corruption detected: stable hash mismatch. "
                    f"Expected {self.stable_hash_reference[:12]}..., "
                    f"computed {computed_hash[:12]}..."
                )
        # Restore from stable IdentityState snapshot
        if self._stable_state is not None and self._state is not None:
            self._state = self._stable_state.model_copy(deep=True)
        self.drift_score = 0.0
        self.frozen = False
        self.slr_breach_count = 0
        self.slr_history = []
        logger.info("[IdentityManager] Identity reverted to stable hash reference state.")

    def update_from_trajectory(self, trajectory) -> None:
        """
        Refines identity state based on a resolved trajectory.
        Extracts diagnostic patterns from fix steps to update capabilities and behavior profile.
        Increments version and recomputes stable hash.
        """
        steps = getattr(trajectory, 'steps', []) if trajectory else []
        if not steps:
            return

        # Extract capability signals from trajectory fixes
        for step in steps:
            fix_text = getattr(step, 'fix', '') or ''
            diagnosis_text = getattr(step, 'diagnosis', '') or ''
            combined = f"{fix_text} {diagnosis_text}".lower()

            # Classify capability from fix patterns
            if any(kw in combined for kw in ("import", "dependency", "module")):
                self.capabilities["dependency_resolution"] = self.capabilities.get("dependency_resolution", 0) + 1
            if any(kw in combined for kw in ("type", "annotation", "hint")):
                self.capabilities["type_safety"] = self.capabilities.get("type_safety", 0) + 1
            if any(kw in combined for kw in ("test", "assert", "validation")):
                self.capabilities["validation"] = self.capabilities.get("validation", 0) + 1
            if any(kw in combined for kw in ("refactor", "extract", "rename")):
                self.capabilities["refactoring"] = self.capabilities.get("refactoring", 0) + 1

        # Increment version
        self.version += 1

        # Recompute stable hash from current behavioral state
        state_material = json.dumps({
            "behavior_profile": {k: round(v, 6) for k, v in self.behavior_profile.items()},
            "mode_weights": self.mode_weights,
            "version": self.version,
            "capabilities": self.capabilities,
        }, sort_keys=True)
        self.stable_hash_reference = hashlib.sha256(state_material.encode("utf-8")).hexdigest()

        logger.info(
            f"[IdentityManager] Identity updated from trajectory: version={self.version}, "
            f"capabilities={list(self.capabilities.keys())}, hash={self.stable_hash_reference[:12]}..."
        )

    def compute_kl_divergence(self) -> float:
        """
        Computes KL divergence between current behavior distribution and the stable reference.
        Returns 0.0 if distributions are identical or insufficient data.
        KL(P || Q) = sum(P(x) * log(P(x) / Q(x))) where P=current, Q=stable.
        """
        epsilon = 1e-10  # Prevent log(0)
        current = self.behavior_profile
        stable = self._stable_state.behavior_profile if self._stable_state is not None else {}

        if not current or not stable:
            return 0.0

        # Normalize both distributions
        current_total = sum(abs(v) for v in current.values()) or 1.0
        stable_total = sum(abs(v) for v in stable.values()) or 1.0

        kl_div = 0.0
        for key in current:
            p = abs(current.get(key, 0.0)) / current_total + epsilon
            q = abs(stable.get(key, 0.0)) / stable_total + epsilon
            kl_div += p * math.log(p / q)

        return max(0.0, kl_div)

    def apply_recovery_tick(self) -> float:
        """
        HD3: Asymptotic Tikhonov-regularized recovery.

        Applies a single recovery step to drift_score using the formula:
            drift_score *= (1 - λ / (1 + drift_score))

        The (1 + drift_score) denominator provides asymptotic damping:
        recovery rate is smaller when drift is high (system is deeply
        destabilized), preventing rapid unstable resets.  As drift
        decays toward zero the effective rate approaches λ, giving
        smooth convergence.

        If drift drops below the containment threshold, the frozen
        state is cleared automatically.

        Returns the new drift_score after the tick.
        """
        lam = self.get_recovery_lambda()
        prev = self.drift_score
        effective_rate = lam / (1.0 + self.drift_score)
        self.drift_score = max(0.0, self.drift_score * (1.0 - effective_rate))

        # Unfreeze if drift has recovered below threshold
        if self.frozen and (self.drift_score + self.compute_kl_divergence()) <= self.max_drift:
            self.frozen = False
            logger.info(
                f"[IdentityManager] Recovery: drift {prev:.4f} -> {self.drift_score:.4f}, "
                f"unfreezing (below max_drift={self.max_drift})."
            )

        if self.telemetry:
            self.telemetry.record(
                "identity", "recovery_tick", node_id="global",
                payload={
                    "prev_drift": round(prev, 4),
                    "new_drift": round(self.drift_score, 4),
                    "recovery_lambda": round(lam, 6),
                    "effective_rate": round(effective_rate, 6),
                    "frozen": self.frozen,
                },
            )
        return self.drift_score

    def check_and_enforce(self, branch_id: str) -> bool:
        """
        Checks if drift exceeds threshold using both accumulated drift and KL divergence.
        Returns True if branch should be frozen (containment protocol).
        """
        kl_drift = self.compute_kl_divergence()
        combined_drift = self.drift_score + kl_drift

        if combined_drift > self.max_drift:
            self.frozen = True
            if self.telemetry:
                self.telemetry.record(
                    "identity", "containment_triggered", node_id=branch_id,
                    payload={
                        "combined_drift": round(combined_drift, 4),
                        "accumulated": round(self.drift_score, 4),
                        "kl": round(kl_drift, 4),
                        "max_drift": self.max_drift,
                    },
                )
            logger.critical(
                f"[IdentityManager] CONTAINMENT PROTOCOL: combined_drift={combined_drift:.4f} "
                f"(accumulated={self.drift_score:.4f}, kl={kl_drift:.4f}) > "
                f"max_drift={self.max_drift}. Freezing branch {branch_id}."
            )
            return True
        return False

    def get_state_snapshot(self) -> Dict:
        """Returns current identity state for telemetry/logging."""
        base = self._state.model_dump() if self._state is not None else {}
        base.update({
            "slr_breach_count": self.slr_breach_count,
            "kl_divergence": round(self.compute_kl_divergence(), 6),
            "rolling_slr_mean": round(self.get_rolling_slr_mean(), 4),
            "rolling_slr_critical": self.is_rolling_slr_critical(),
            "slr_history_len": len(self.slr_history),
            "recovery_lambda": round(self.get_recovery_lambda(), 6),
        })
        return base
