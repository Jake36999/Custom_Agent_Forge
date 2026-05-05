import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
"""
Aletheia Advanced Semantic Compiler v3.6 (Theorist Creator Mode)
The Self-Optimizing Epistemic System (Production Runtime)

Upgrades (v3.6 Production Safety & Control):
- Hard Safety Bounds: `_enforce_global_bounds` prevents system collapse and over-constraining.
- State Persistence: `export_state` and `load_state` maintain learned intelligence across runs.
- Train vs Inference Mode: Explicit `runtime_mode` freezes adaptation for deterministic execution.
- Parameter Impact Weighting: Gradient updates are now scaled by historical parameter impact.
- Sliding Window Context: Limits context signatures to a localized semantic chain (preventing hash decay).
- Ontology Pruning: Entropy-based decay purges unused learned predicates to prevent semantic bloat.
"""

import spacy
import logging
import uuid
import hashlib
import random
import math
from collections import Counter, deque
from typing import Iterator, Dict, Any, Tuple, Iterable, Optional, List

# --- AMR Dependencies ---
try:
    import amrlib
    import penman
    HAS_AMR = True
except ImportError:
    HAS_AMR = False

# --- PDF Extraction (Theorist Mode) ---
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import pytesseract
    from PIL import Image
    import io as _io
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("output/orchestrator_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Predicate normalization ---
PREDICATE_MAP = {
    "use": "apply",
    "utilize": "apply",
    "implement": "apply",
    "demonstrate": "show",
    "indicate": "show",
    "suggest": "propose"
}

class SemanticNodeBuilder:
    """
    Constructs the authoritative Node Schema v3.6.
    Enforces the Epistemic & Executable Contract Layer for `AletheiaSkill`.
    """
    def __init__(
        self, text: str, frames: List[Dict[str, Any]], discourse: Dict[str, Any], 
        graph: Optional[Dict[str, Any]], section: str, source_file: str, index: int,
        pipeline_type: str, pipeline_role: str, intent: str, lens: str, 
        verifiability: str, mode: str, config: Dict[str, Any], 
        context_window: List[str], previous_node: Optional[Dict[str, Any]] = None
    ):
        self.node_id = f"edu_{uuid.uuid4().hex[:8]}"
        self.mode = mode
        self.text = text
        self.frames = frames
        self.discourse = discourse
        self.graph = graph
        self.section = section
        self.source_file = source_file
        self.index = index
        self.config = config
        
        self.pipeline_type = pipeline_type
        self.pipeline_role = pipeline_role
        self.intent = intent
        self.lens = lens
        self.verifiability = verifiability
        self.previous_node = previous_node
        self.context_window = context_window
        self.context_signature = None # Computed during build()

    def _build_canonical_text(self) -> str:
        # Sorted to ensure order-independent, stable embeddings
        sorted_frames = sorted(self.frames, key=lambda f: f["predicate"]["lemma"])
        return " ".join(
            f"{f['predicate']['lemma']}({', '.join(a['role'] for a in f['arguments'])})"
            for f in sorted_frames
        )

    def _build_deterministic_id(self) -> str:
        # Anchors identity to the canonical semantic extraction rather than noisy raw text
        canonical = self._build_canonical_text()
        base_text = canonical if canonical.strip() else self.text
        base = f"{self.source_file}:{self.index}:{base_text}"
        return hashlib.sha256(base.encode('utf-8')).hexdigest()[:16]

    def _build_semantic_signature(self) -> Dict[str, Any]:
        pred_set = sorted(list(set(f["predicate"]["lemma"] for f in self.frames)))
        role_set = sorted(list(set(a["role"] for f in self.frames for a in f["arguments"])))
        arity_pattern = [len(f["arguments"]) for f in self.frames]

        sig_string = f"{self.pipeline_type}:{self.discourse['label']}:{'-'.join(pred_set)}:{'-'.join(role_set)}"
        sig_hash = hashlib.sha256(sig_string.encode('utf-8')).hexdigest()

        return {
            "predicate_set": pred_set,
            "role_set": role_set,
            "arity_pattern": arity_pattern,
            "discourse": self.discourse["label"],
            "pipeline_type": self.pipeline_type,
            "hash": sig_hash
        }

    def _build_edge_hints(self) -> List[Dict[str, Any]]:
        hints = []
        label = self.discourse.get("label")
        if label == "evidence":
            hints.append({"relation": "supports", "target_strategy": "previous", "confidence": 0.85})
        elif label == "method":
            hints.append({"relation": "applies", "target_strategy": "previous", "confidence": 0.80})
        elif label == "hypothesis":
            hints.append({"relation": "tests", "target_strategy": "forward", "confidence": 0.70})
        return hints

    def _build_constraints(self) -> Dict[str, Any]:
        risk = "medium"
        if self.intent == "execution" and self.verifiability == "low":
            risk = "high"
        elif self.verifiability == "high":
            risk = "low"

        return {
            "risk_level": risk,
            "failure_modes": ["pending_advocate_review"],
            "requires_validation": risk in ["high", "critical"]
        }

    def _build_confidence_profile(self) -> Dict[str, float]:
        if not self.frames:
            return {"overall": self.discourse["confidence"], "epistemic": self.discourse["confidence"], "structural": 0.0}

        pred_conf = sum(f["confidence"]["predicate"] for f in self.frames) / len(self.frames)
        arg_conf = sum(f["confidence"]["arguments"] for f in self.frames) / len(self.frames)

        decay_factor = max(0.7, 1.0 - (len(self.frames) * 0.02))

        structural = ((pred_conf + arg_conf) / 2.0) * decay_factor
        epistemic = self.discourse["confidence"] * decay_factor

        return {
            "overall": round((structural + epistemic) / 2.0, 3),
            "epistemic": round(epistemic, 3),
            "structural": round(structural, 3)
        }

    def _build_complexity(self) -> Dict[str, Any]:
        frame_count = len(self.frames)
        arg_count = sum(len(f["arguments"]) for f in self.frames)
        graph_density = 0.0
        
        if self.graph and self.graph.get("nodes"):
            n_len = len(self.graph["nodes"])
            e_len = len(self.graph.get("edges", []))
            if n_len > 1:
                graph_density = round(e_len / (n_len * (n_len - 1)), 3)

        complexity_score = round((frame_count * 1.0) + (arg_count * 0.5) + (graph_density * 2.0), 3)

        return {
            "frame_count": frame_count,
            "argument_count": arg_count,
            "graph_density": graph_density if graph_density > 0 else None,
            "complexity_score": complexity_score
        }

    def _build_learning_hooks(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        should_review = self.verifiability == "low" or constraints["risk_level"] == "high"
        improvement_targets = []
        if any(f["confidence"]["arguments"] < self.config["frame_min_arg_confidence"] for f in self.frames):
            improvement_targets.append("argument_extraction")
            
        return {
            "should_review": should_review,
            "improvement_targets": improvement_targets if improvement_targets else None
        }

    def _build_training_signals(self) -> Dict[str, Any]:
        is_hq = bool(self.verifiability == "high" and len(self.frames) > 0 and self.discourse["confidence"] > 0.5)
        return {
            "is_high_quality": is_hq,
            "label_candidates": [self.intent, self.pipeline_type, self.discourse["label"]],
            "instruction_template": f"Extract epistemic reasoning vectors for a {self.pipeline_type} text block."
        }

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _build_structural_signals(self) -> Dict[str, float]:
        frame_count = len(self.frames)
        arg_count = sum(len(f.get("arguments", [])) for f in self.frames)
        node_count = 0
        edge_count = 0
        if isinstance(self.graph, dict):
            node_count = len(self.graph.get("nodes", []) or [])
            edge_count = len(self.graph.get("edges", []) or [])
        relation_density = edge_count / max(1, node_count * (node_count - 1)) if node_count > 1 else 0.0
        frame_density = frame_count / max(1, frame_count + 4)
        entity_connectivity = arg_count / max(1, frame_count * 3)
        return {
            "frame_density": self._clamp01(frame_density),
            "entity_connectivity": self._clamp01(entity_connectivity),
            "relation_density": self._clamp01(relation_density),
        }

    def _build_logical_signals(self, constraints: Dict[str, Any]) -> Dict[str, float]:
        risk = str(constraints.get("risk_level", "medium"))
        risk_penalty = {"low": 0.0, "medium": 0.25, "high": 0.5, "critical": 0.75}.get(risk, 0.25)
        requires_validation = 1.0 if constraints.get("requires_validation") else 0.0
        failure_mode_count = len(constraints.get("failure_modes", []) or [])
        satisfiability = self._clamp01(1.0 - min(1.0, risk_penalty + (failure_mode_count * 0.1)))
        consistency_pressure = self._clamp01(1.0 - (requires_validation * 0.35))
        return {
            "satisfiability": satisfiability,
            "consistency_pressure": consistency_pressure,
            "constraint_load": self._clamp01(1.0 - ((satisfiability + consistency_pressure) / 2.0)),
        }

    def _build_reasoning_signals(self, confidence_profile: Dict[str, float]) -> Dict[str, float]:
        predicates = [f.get("predicate", {}).get("lemma", "") for f in self.frames]
        unique_predicates = len({p for p in predicates if p})
        novelty = self._clamp01(unique_predicates / max(1, len(predicates)))
        epistemic_conf = float(confidence_profile.get("epistemic", 0.0) or 0.0)
        structural_conf = float(confidence_profile.get("structural", 0.0) or 0.0)
        consistency = self._clamp01((epistemic_conf + structural_conf) / 2.0)
        coverage = self._clamp01(len(self.frames) / max(3, len(self.frames) + 2))
        return {
            "coverage": coverage,
            "novelty": novelty,
            "consistency": consistency,
        }

    def _build_domain_signals(self) -> Dict[str, float]:
        # Mode-aware abstract weighting only: no domain literals, no language locks.
        mode_bias = {
            "theorist": (1.05, 0.95, 1.0),
            "coding_assistant": (1.0, 1.0, 1.0),
            "advocate": (0.95, 1.05, 1.0),
            "veteran": (1.0, 0.9, 1.1),
        }.get(self.mode, (1.0, 1.0, 1.0))
        role_factor = 1.1 if self.pipeline_role == "validation" else 1.0
        return {
            "coverage_weight": round(mode_bias[0] * role_factor, 4),
            "novelty_weight": round(mode_bias[1], 4),
            "consistency_weight": round(mode_bias[2], 4),
            "flux_weight": round((mode_bias[0] + mode_bias[1] + mode_bias[2]) / 3.0, 4),
        }

    def _derive_invariant_signals(
        self,
        structural: Dict[str, float],
        logical: Dict[str, float],
        reasoning: Dict[str, float],
        domain: Dict[str, float],
    ) -> Dict[str, Any]:
        structural_blend = (structural["frame_density"] + structural["entity_connectivity"] + structural["relation_density"]) / 3.0
        logical_blend = (logical["satisfiability"] + logical["consistency_pressure"]) / 2.0
        content_density = self._clamp01((0.55 * structural_blend) + (0.45 * logical_blend))

        coverage = self._clamp01(reasoning["coverage"] * domain["coverage_weight"])
        novelty = self._clamp01(reasoning["novelty"] * domain["novelty_weight"])
        consistency = self._clamp01(reasoning["consistency"] * domain["consistency_weight"])
        alignment_vector = [coverage, novelty, consistency]

        semantic_vector = [structural_blend, logical_blend, reasoning["consistency"]]
        divergence = sum(abs(a - b) for a, b in zip(alignment_vector, semantic_vector)) / 3.0
        grad_norm = math.sqrt(sum(v * v for v in alignment_vector))
        composite_quality_score = max(0.0, content_density * (0.6 * grad_norm + 0.4 * divergence) * domain["flux_weight"])

        return {
            "channels": {
                "structural": {k: round(v, 4) for k, v in structural.items()},
                "logical": {k: round(v, 4) for k, v in logical.items()},
                "reasoning": {k: round(v, 4) for k, v in reasoning.items()},
                "domain": {k: round(v, 4) for k, v in domain.items()},
            },
            "derived": {
                "content_density": round(content_density, 4),
                "alignment_vector": [round(v, 4) for v in alignment_vector],
                "composite_quality_score": round(composite_quality_score, 4),
            },
        }

    def _build_legacy_layer(self) -> Dict[str, Any]:
        intents, strategies, patterns = [], [], []
        for f in self.frames:
            action = f["predicate"]["lemma"].upper()
            agents = [a.get("text", "") for a in f["arguments"] if a.get("role") == "agent" and a.get("text")]
            themes = [a.get("text", "") for a in f["arguments"] if a.get("role") == "theme" and a.get("text")]
            if themes:
                intents.append(f"{action} targeting [{', '.join(themes)}]")
                patterns.append(f"Predicate({action}) -> Args({', '.join(themes)})")
            else:
                intents.append(f"{action} (intransitive)")
                patterns.append(f"Predicate({action})")
            if agents:
                strategies.append(f"Executed via [{', '.join(agents)}]")
                
        return {
            "reasoning_vectors": {
                "intent": " | ".join(intents) if intents else "STATE_DECLARATION",
                "strategy": " | ".join(list(dict.fromkeys(strategies))) if strategies else "PASSIVE_OR_IMPLICIT_EXECUTION",
                "constraints": ["pending_advocate_review"],
                "execution_pattern": [" | ".join(patterns)] if patterns else ["STATIC_RELATION"],
                "failure_modes": ["pending_advocate_review"]
            }
        }

    def build(self) -> Dict[str, Any]:
        semantic_signature = self._build_semantic_signature()
        current_hash = semantic_signature["hash"]
        
        # Sliding Window Context Chaining (prevents meaningless infinite hashes)
        if self.context_window:
            chain_str = ":".join(self.context_window + [current_hash])
            self.context_signature = hashlib.sha256(chain_str.encode('utf-8')).hexdigest()[:16]
        else:
            self.context_signature = current_hash

        constraints = self._build_constraints()
        confidence_profile = self._build_confidence_profile()
        complexity = self._build_complexity()
        structural_signals = self._build_structural_signals()
        logical_signals = self._build_logical_signals(constraints)
        reasoning_signals = self._build_reasoning_signals(confidence_profile)
        domain_signals = self._build_domain_signals()
        invariant_signals = self._derive_invariant_signals(
            structural=structural_signals,
            logical=logical_signals,
            reasoning=reasoning_signals,
            domain=domain_signals,
        )
        derived = invariant_signals["derived"]
        execution_eligible = bool(derived["content_density"] > 0.0 and derived["composite_quality_score"] > 0.0)
        
        # --- DYNAMIC CONCEPT NAME EXTRACTION ---
        concept_name = f"edu_{self.index}_{self.pipeline_type}"  # Fallback
        try:
            for f in self.frames:
                # Look for theme, agent, or location arguments
                themes = [a["text"] for a in f.get("arguments", []) if a.get("role") in ("theme", "agent", "location") and a.get("text")]
                if themes:
                    # Optionally sanitize to snake_case or concise string if needed
                    concept_name = themes[0]
                    break
        except Exception:
            pass  # Always fallback if structure is malformed

        return {
            # --- MANDATORY ALETHEIASKILL FIELDS (models.py Compatibility) ---
            "node_id": self.node_id,
            "name": concept_name,
            "file": self.source_file,
            "code_snippet": self.text, 
            "imports": [],
            "operator_type": self.discourse["label"],
            "skill_type": "theoretical_reasoning",
            "source_type": "ocr_document",
            "teaching_layer": self._build_legacy_layer(),
            
            # --- EPISTEMIC STATE MATCHING ---
            "epistemic": {
                "state": self.verifiability,
                "c_node": confidence_profile["overall"],
                "confidence": confidence_profile,
                "retry_budget": 6,
                "depth": 0
            },

            # --- CORE V3.6 ARCHITECTURE FIELDS ---
            "metadata": {
                "deterministic_id": self._build_deterministic_id(),
                "schema_version": "text_semantic_node_v3",
                "compiler_version": "Aletheia_ASC_v3.6",
                "mode": self.mode,
                "canonical_text": self._build_canonical_text(),
                "context_signature": self.context_signature,
                "source_ref": {
                    "sentence_index": self.index,
                },
                "intent": self.intent,
                "lens": self.lens,
                "pipeline_type": self.pipeline_type,
                "pipeline_role": self.pipeline_role,
            },

            # --- SEMANTICS (Payload Container) ---
            "semantics": {
                "discourse": self.discourse,
                "semantic_frames": self.frames,
                "semantic_signature": semantic_signature,
                "invariant_signals": invariant_signals,
            },

            # --- DAG/ACS HOOKS ---
            "graph": self.graph,
            "identity": {
                "edge_hints": self._build_edge_hints(),
            },
            "constraints": constraints,
            "validation": {
                "learning_hooks": self._build_learning_hooks(constraints),
                "complexity": complexity,
                "training_signals": self._build_training_signals()
            },
            "execution_eligible": execution_eligible,
            "invariant_signals": invariant_signals,
        }


class AdvancedSemanticChunker:

    def __init__(self, model_name: str = "en_core_web_trf", mode: str = "theorist", runtime_mode: str = "train"):
        self.mode = mode
        self.runtime_mode = runtime_mode # "train" (adaptive) or "inference" (frozen)
        
        # --- SYSTEM STATE REPRESENTATION ---
        self.system_state = {
            "mode": "exploration", # exploration | exploitation | recovery
            "stability": 1.0,
            "drift": 0.0
        }
        
        # --- ADAPTIVE CONFIGURATION STATE ---
        self.config = {
            "tvg_max_upper_ratio": 0.6,
            "tvg_min_words": 5,
            "tvg_max_list_newlines": 3,
            "frame_max_arg_tokens": 25,
            "frame_min_pred_confidence": 0.8,
            "frame_min_arg_confidence": 0.5
        }
        
        # --- OPTIMIZATION STATE (PID + Rolling Memory) ---
        self.reward_history = deque(maxlen=10)
        self.optimizer_state = {
            "last_reward": 0.0,
            "momentum": {
                "tvg_max_upper_ratio": -0.05,
                "frame_max_arg_tokens": -2,
                "frame_min_pred_confidence": 0.05
            },
            "param_impact": { 
                "tvg_max_upper_ratio": 0.0,
                "frame_max_arg_tokens": 0.0,
                "frame_min_pred_confidence": 0.0
            }
        }
        
        # --- GOAL-DRIVEN OPTIMISATION METRICS ---
        self.metrics = {
            "nodes_processed": 0,
            "nodes_accepted": 0,
            "edges_generated": 0,
            "unique_edge_targets": set(), 
            "total_confidence": 0.0,
            "total_complexity": 0.0,
            "downstream_rejections": 0
        }
        
        # --- DYNAMIC ONTOLOGY ---
        self.predicate_map = {
            "use": "apply", "utilize": "apply", "implement": "apply",
            "demonstrate": "show", "indicate": "show", "suggest": "propose"
        }
        self.predicate_classes = {
            "execution": {"apply", "compute", "implement"},
            "evidence": {"show", "demonstrate", "find"},
            "hypothesis": {"propose", "suggest", "assume"},
            "diagnostic": {"detect", "identify", "measure"}
        }
        
        self.learned_predicates = set()
        self.learned_predicates_usage = {}

        self.pipeline_types = {
            "procedural": {"method", "apply", "compute"},
            "analytical": {"analyze", "evaluate", "compare"},
            "descriptive": {"describe", "define", "explain"},
            "causal": {"cause", "lead", "result"},
            "evidential": {"show", "demonstrate", "find"},
            "hypothetical": {"propose", "suggest", "assume"},
            "instructional": {"step", "process", "guide"},
            "diagnostic": {"detect", "identify", "measure"}
        }

        try:
            self.nlp = spacy.load(model_name, disable=["ner", "textcat"])
            self.nlp.max_length = 5000000
            logger.info(f"[OK] Loaded {model_name}")
        except OSError:
            logger.warning("[!] Falling back to en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
            self.nlp.max_length = 5000000

        if HAS_AMR:
            amrlib.setup_spacy_extension()

    # =========================
    # PERSISTENCE LAYER
    # =========================

    def export_state(self) -> Dict[str, Any]:
        """Snapshots compiler intelligence for cross-session continuity."""
        return {
            "config": self.config,
            "predicate_classes": {k: list(v) for k, v in self.predicate_classes.items()},
            "learned_predicates": list(self.learned_predicates),
            "learned_predicates_usage": self.learned_predicates_usage,
            "optimizer_state": self.optimizer_state,
            "reward_history": list(self.reward_history),
            "system_state": self.system_state
        }

    def load_state(self, state: Dict[str, Any]):
        """Restores compiled intelligence to prevent re-learning thrash."""
        self.config.update(state.get("config", {}))
        for k, v in state.get("predicate_classes", {}).items():
            self.predicate_classes[k] = set(v)
        self.learned_predicates = set(state.get("learned_predicates", []))
        self.learned_predicates_usage = state.get("learned_predicates_usage", {})
        self.optimizer_state.update(state.get("optimizer_state", {}))
        self.reward_history = deque(state.get("reward_history", []), maxlen=10)
        self.system_state.update(state.get("system_state", {}))
        logger.info("[*] Successfully loaded persistent intelligence state.")

    def _update_config(self, key: str, delta: float, min_val: float, max_val: float):
        current = self.config[key]
        new_val = max(min_val, min(max_val, current + delta))
        self.config[key] = round((current * 0.7) + (new_val * 0.3), 4)

    def _enforce_global_bounds(self):
        """Hard Safety Bounds: Global Kill Constraints preventing unrecoverable states."""
        if self.config["frame_min_pred_confidence"] > 0.93:
            logger.critical("⚠️ HARD LIMIT: Confidence threshold too high. Forcing reset to prevent yield collapse.")
            self.config["frame_min_pred_confidence"] = 0.85
            
        if self.config["frame_max_arg_tokens"] < 12:
            logger.critical("⚠️ HARD LIMIT: Argument window collapse detected. Forcing reset to preserve context.")
            self.config["frame_max_arg_tokens"] = 20

        if self.config["tvg_max_upper_ratio"] < 0.25:
            logger.critical("⚠️ HARD LIMIT: TVG Upper Ratio too strict. Forcing reset to avoid acronym destruction.")
            self.config["tvg_max_upper_ratio"] = 0.4

    # =========================
    # FORMAL OPTIMIZATION OBJECTIVE (THE CONTROL LOOP)
    # =========================

    def evaluate_and_optimize(self, downstream_rejection_logs: List[Dict[str, Any]] = None) -> float:
        """
        The formal objective function + PID control loop.
        Only actively adapts if `runtime_mode == "train"`.
        """
        if self.runtime_mode != "train":
            logger.debug("[*] Evaluation bypassed: Compiler is in INFERENCE mode.")
            return 0.0

        logs = downstream_rejection_logs or []
        self.metrics["downstream_rejections"] += len(logs)
        
        if self.metrics["nodes_processed"] == 0:
            return 0.0

        nodes_acc = max(self.metrics["nodes_accepted"], 1)
        yield_rate = self.metrics["nodes_accepted"] / self.metrics["nodes_processed"]
        avg_conf = self.metrics["total_confidence"] / nodes_acc
        avg_complexity = self.metrics["total_complexity"] / nodes_acc
        rejection_rate = self.metrics["downstream_rejections"] / nodes_acc
        
        # Graph Structure Quality
        connectivity = min(1.0, len(self.metrics["unique_edge_targets"]) / nodes_acc)
        
        # Signal Integrity Term
        signal_integrity = min(1.0, avg_conf * (1.0 - rejection_rate))
        complexity_penalty = min(0.3, avg_complexity * 0.05)
        
        reward = (
            (yield_rate * 0.25) + 
            (avg_conf * 0.30) + 
            (connectivity * 0.20) + 
            (signal_integrity * 0.25)
        ) - (rejection_rate * 0.6) - complexity_penalty
        
        current_reward = round(max(0.0, min(1.0, reward)), 4)
        self.reward_history.append(current_reward)
        
        logger.info(f"[*] Epistemic Optimization Run. Reward: {current_reward} (Yield: {yield_rate:.2f}, Conf: {avg_conf:.2f}, Conn: {connectivity:.2f}, Integrity: {signal_integrity:.2f})")
        
        if yield_rate > 0.9 and avg_conf < 0.5:
            logger.warning("⚠️ SYSTEM ALERT: High yield / low confidence drift detected (Garbage Ingestion).")
            self.system_state["mode"] = "recovery"
        elif connectivity > 0.9 and len(self.metrics["unique_edge_targets"]) < (self.metrics["edges_generated"] * 0.1):
            logger.warning("⚠️ SYSTEM ALERT: Graph collapse detected (Edge Spamming).")
        
        if len(self.reward_history) >= 6:
            trend = sum(list(self.reward_history)[-3:]) - sum(list(self.reward_history)[:3])
            self.system_state["drift"] = trend
            if trend < -0.05:
                self.system_state["mode"] = "recovery"
            elif abs(trend) < 0.01:
                self.system_state["mode"] = "exploration"
            else:
                self.system_state["mode"] = "exploitation"

        reward_delta = current_reward - self.optimizer_state["last_reward"]
        
        if abs(reward_delta) < 0.01 and self.system_state["mode"] != "exploration":
            logger.info("[*] Reward within dead zone tolerance. Stabilizing system state.")
        else:
            if reward_delta < 0:
                logger.info("[!] Reward decreased. Reversing momentum and annealing step size.")
                for key in self.optimizer_state["momentum"]:
                    self.optimizer_state["param_impact"][key] -= 0.1 
                    self.optimizer_state["momentum"][key] *= -0.5 
            else:
                for key in self.optimizer_state["momentum"]:
                    self.optimizer_state["param_impact"][key] += 0.1 
                    self.optimizer_state["momentum"][key] *= 1.1 

            if self.system_state["mode"] == "exploration" or random.random() < 0.1:
                logger.info("[*] Executing ε-exploration step to prevent local maxima lock.")
                self._update_config("frame_min_pred_confidence", random.uniform(-0.1, 0.1), 0.5, 0.95)
                self._update_config("tvg_max_upper_ratio", random.uniform(-0.05, 0.05), 0.2, 0.8)
            else:
                for key in self.optimizer_state["momentum"]:
                    # Scale update magnitude by the historical impact weight of the parameter
                    impact_weight = max(0.5, min(2.0, 1.0 + abs(self.optimizer_state["param_impact"][key])))
                    scaled_momentum = self.optimizer_state["momentum"][key] * impact_weight
                    
                    if key == "tvg_max_upper_ratio": self._update_config(key, scaled_momentum, 0.2, 0.8)
                    elif key == "frame_max_arg_tokens": self._update_config(key, scaled_momentum, 10, 40)
                    elif key == "frame_min_pred_confidence": self._update_config(key, scaled_momentum, 0.5, 0.95)
        
        # Ontology Pruning (Entropy-based decay to prevent semantic bloat)
        pruned = 0
        for p in list(self.learned_predicates_usage.keys()):
            self.learned_predicates_usage[p] *= 0.9 # 10% Decay per epoch
            if self.learned_predicates_usage[p] < 0.5:
                if p in self.predicate_classes["execution"]:
                    self.predicate_classes["execution"].remove(p)
                    self.learned_predicates.remove(p)
                    pruned += 1
                del self.learned_predicates_usage[p]
        if pruned > 0:
            logger.info(f"    -> Pruned {pruned} unused learned predicates (Semantic Entropy control).")

        # Dynamic Ontology Expansion (Validation Gated)
        new_preds = 0
        pred_counts = Counter(log.get("predicate") for log in logs if log.get("reason") == "unknown_predicate" and "predicate" in log)
        for pred, count in pred_counts.items():
            if count > 5: 
                self.predicate_classes["execution"].add(pred)
                self.learned_predicates.add(pred)
                self.learned_predicates_usage[pred] = 5.0 # Initial utility boost
                new_preds += 1
                
        if new_preds > 0:
            logger.info(f"    -> Learned {new_preds} new execution predicates from verified ACS feedback.")
                
        self.optimizer_state["last_reward"] = current_reward
        
        # Enforce Global Safety Bounds before exiting optimization epoch
        self._enforce_global_bounds()
        
        logger.info(f"[*] Active Configuration State: {self.config} | System Mode: {self.system_state['mode']}")
        
        self.metrics = {k: 0 if isinstance(v, int) else (set() if isinstance(v, set) else 0.0) for k, v in self.metrics.items()}
        
        return current_reward

    # =========================
    # TEXT VALIDITY GATE (TVG)
    # =========================

    def _is_index_block(self, text: str) -> bool:
        keywords = ["index", "contents", "volume", "edited by", "bibliography"]
        text_lower = text.lower()
        word_count = len(text.split())
        # Only flag as index block if it LOOKS like a TOC/index page (keyword-dense,
        # relatively short). Full documents that merely contain these words are not index blocks.
        if word_count > 500:
            return False  # Too long to be an index/TOC block — it's a real document
        if any(k in text_lower for k in keywords):
            if text.count("\n") > self.config["tvg_max_list_newlines"] or word_count > 200:
                return True
        return False

    def _is_valid_sentence(self, span: spacy.tokens.Span) -> bool:
        has_verb = any(t.pos_ == "VERB" for t in span)
        text = span.text.strip()
        
        if not text: return False

        upper_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if upper_ratio > self.config["tvg_max_upper_ratio"] and len(text.split()) >= 10:
            return False

        if len(text.split()) < self.config["tvg_min_words"]:
            return False

        if text.count("\n") > self.config["tvg_max_list_newlines"]:
            return False

        if not has_verb:
            return False

        return True

    # =========================
    # MAIN PIPELINE
    # =========================

    def stream_documents(
        self,
        text_stream: Iterable[Tuple[str, Dict[str, Any]]],
        batch_size: int = 50
    ) -> Iterator[Dict[str, Any]]:

        previous_node = None
        consecutive_drops = 0
        context_window = deque(maxlen=3) # Sliding window for Deep Context Chaining

        for doc, context in self.nlp.pipe(text_stream, as_tuples=True, batch_size=batch_size):
            source_file = context.get("source_file", "unknown_source")

            if self._is_index_block(doc.text):
                continue

            for sent_idx, sent in enumerate(doc.sents):
                self.metrics["nodes_processed"] += 1
                
                if not self._is_valid_sentence(sent):
                    continue

                text = sent.text.strip()
                section = self._detect_section(text)

                frames = self._extract_predicate_frames(sent)
                
                # Minimum Output Safeguard
                if not frames:
                    consecutive_drops += 1
                    if consecutive_drops > 20 and self.config["frame_min_pred_confidence"] > 0.85:
                        logger.warning("⚠️ Over-constrained system detected: 20+ drops. Relaxing constraints temporarily.")
                        self._update_config("frame_min_pred_confidence", -0.05, 0.6, 0.95)
                        consecutive_drops = 0
                    continue
                
                consecutive_drops = 0
                self.metrics["nodes_accepted"] += 1
                
                # Track usage of dynamically learned ontology
                if self.runtime_mode == "train":
                    for f in frames:
                        lemma = f["predicate"]["lemma"]
                        if lemma in self.learned_predicates:
                            self.learned_predicates_usage[lemma] = self.learned_predicates_usage.get(lemma, 0) + 1.0

                discourse = self._classify_discourse(sent)
                graph = self._generate_amr_graph(sent) or self._generate_dependency_graph(sent)
                
                pipeline_type = self._classify_pipeline_type(frames, discourse)
                pipeline_role = self._infer_pipeline_role(discourse)
                intent = self._extract_intent(frames, discourse)
                lens = self._select_lens(discourse, pipeline_type)
                verifiability = self._assess_verifiability(discourse)

                node = SemanticNodeBuilder(
                    text=text,
                    frames=frames,
                    discourse=discourse,
                    graph=graph,
                    section=section,
                    source_file=source_file,
                    index=sent_idx,
                    pipeline_type=pipeline_type,
                    pipeline_role=pipeline_role,
                    intent=intent,
                    lens=lens,
                    verifiability=verifiability,
                    mode=self.mode,
                    config=self.config,
                    context_window=list(context_window),
                    previous_node=previous_node
                ).build()
                
                # Append to sliding context window to maintain semantic locality
                context_window.append(node["semantics"]["semantic_signature"]["hash"])
                
                self.metrics["total_confidence"] += node["epistemic"]["c_node"]
                self.metrics["total_complexity"] += node["validation"]["complexity"]["complexity_score"]

                if previous_node:
                    edge = self._infer_edge(previous_node, node)
                    if edge:
                        self.metrics["edges_generated"] += 1
                        self.metrics["unique_edge_targets"].add(edge["target"])
                        yield edge

                yield node
                previous_node = node

    # =========================
    # EPISTEMIC ROUTING PRIMITIVES
    # =========================

    def _get_predicate_class(self, lemma: str) -> str:
        for cls, vals in self.predicate_classes.items():
            if lemma in vals:
                return cls
        return "other"

    def _extract_intent(self, frames: List[Dict[str, Any]], discourse: Dict[str, Any]) -> str:
        pred_classes = [self._get_predicate_class(f["predicate"]["lemma"]) for f in frames]
        if "execution" in pred_classes:
            return "execution"
        if discourse["label"] == "hypothesis":
            return "exploration"
        if discourse["label"] == "evidence":
            return "validation"
        return "analysis"

    def _select_lens(self, discourse: Dict[str, Any], pipeline_type: str) -> str:
        if pipeline_type == "analytical": return "AnalystLens"
        if pipeline_type == "procedural": return "ExecutorLens"
        if discourse["label"] == "hypothesis": return "ResearchLens"
        return "GeneralLens"

    def _assess_verifiability(self, discourse: Dict[str, Any]) -> str:
        if discourse["label"] in ("method", "evidence"): return "high"
        if discourse["label"] == "hypothesis": return "low"
        return "medium"

    def _infer_pipeline_role(self, discourse: Dict[str, Any]) -> str:
        mapping = {
            "hypothesis": "input",
            "method": "transform",
            "evidence": "validation",
            "summary": "output"
        }
        return mapping.get(discourse["label"], "input")

    def _classify_pipeline_type(self, frames: List[Dict[str, Any]], discourse: Dict[str, Any]) -> str:
        predicates = [f["predicate"]["lemma"] for f in frames]
        for p in predicates:
            for p_type, triggers in self.pipeline_types.items():
                if p in triggers: return p_type
        d_label = discourse.get("label", "")
        if d_label == "method": return "procedural"
        if d_label == "evidence": return "evidential"
        if d_label == "hypothesis": return "hypothetical"
        if d_label == "summary": return "analytical"
        return "descriptive"

    # =========================
    # FRAME EXTRACTION
    # =========================

    def _extract_predicate_frames(self, span) -> List[Dict[str, Any]]:
        frames = []

        predicates = [
            t for t in span
            if t.pos_ == "VERB"
            and t.dep_ not in ("aux", "auxpass")
            and t.lemma_ not in {"be", "have"}
        ]
        
        if not predicates:
            predicates = [t for t in span if t.dep_ == "ROOT" and t.lemma_ not in {"be", "have"}]

        for pred in predicates:
            lemma = pred.lemma_.lower()
            canonical = self.predicate_map.get(lemma, lemma)

            frame = {
                "frame_id": f"{span.start}_{pred.i}",
                "predicate": {
                    "text": pred.text,
                    "lemma": canonical,
                    "token_index": pred.i
                },
                "arguments": [],
                "features": {
                    "polarity": "negative" if any(c.dep_ == "neg" for c in pred.children) else "positive",
                    "tense": pred.morph.get("Tense", []),
                    "aspect": pred.morph.get("Aspect", [])
                },
                "confidence": {
                    "predicate": 0.9 if pred.dep_ == "ROOT" else 0.6,
                    "arguments": 0.0
                }
            }

            arg_count = 0
            for child in pred.children:
                role = self._map_role(child.dep_)
                if not role: continue
                
                if (child.right_edge.i - child.left_edge.i) > self.config["frame_max_arg_tokens"]: continue
                
                subtree_len = len(list(child.subtree))

                arg = {
                    "role": role,
                    "type": child.pos_,
                    "dep": child.dep_,
                    "text": " ".join(t.text for t in child.subtree),
                    "token_start": child.left_edge.i,
                    "token_end": child.right_edge.i,
                    "span_confidence": 1.0 if subtree_len < 10 else 0.7
                }

                frame["arguments"].append(arg)
                arg_count += 1

            frame["confidence"]["arguments"] = min(1.0, arg_count / 3)

            if len(frame["arguments"]) == 0 and frame["confidence"]["predicate"] < self.config["frame_min_pred_confidence"]:
                continue

            frames.append(frame)
        return frames

    def _map_role(self, dep: str) -> Optional[str]:
        if dep in ("nsubj", "nsubjpass", "csubj"): return "agent"
        if dep in ("dobj", "attr", "oprd"): return "theme"
        if dep in ("iobj", "dative"): return "recipient"
        if dep in ("prep", "pobj"): return "location"
        if dep in ("advmod", "advcl"): return "manner"
        return None

    # =========================
    # DISCOURSE & GRAPH
    # =========================

    def _classify_discourse(self, span) -> Dict[str, Any]:
        root = next((t for t in span if t.dep_ == "ROOT"), None)
        lemma = root.lemma_.lower() if root else ""

        taxonomy = {
            "hypothesis": ["propose", "suggest", "assume"],
            "evidence": ["show", "demonstrate", "find"],
            "method": ["apply", "use", "compute"],
            "summary": ["conclude", "summarize"],
            "claim": ["state", "argue"]
        }

        for label, words in taxonomy.items():
            if lemma in words:
                return {"label": label, "confidence": 0.85}

        return {"label": "background", "confidence": 0.4}

    def _generate_amr_graph(self, span):
        if not HAS_AMR: return None
        try:
            penman_str = span._.to_amr()[0]
            graph = penman.decode(penman_str)
            return {
                "type": "amr",
                "top": graph.top,
                "nodes": list(graph.variables()),
                "edges": [{"source": e.source, "role": e.role, "target": e.target} for e in graph.edges()]
            }
        except: return None

    def _generate_dependency_graph(self, span):
        return {
            "type": "dependency",
            "top": next((t.i for t in span if t.dep_ == "ROOT"), None),
            "nodes": [{"id": t.i, "text": t.text} for t in span],
            "edges": [{"source": t.head.i, "target": t.i} for t in span if t.head != t]
        }

    def _infer_edge(self, prev, curr):
        prev_preds = prev.get("semantics", {}).get("semantic_signature", {}).get("predicate_set", [])
        curr_preds = curr.get("semantics", {}).get("semantic_signature", {}).get("predicate_set", [])
        if prev_preds and curr_preds and prev_preds == curr_preds:
            return {"type": "edge", "relation": "equivalent", "source": prev["node_id"], "target": curr["node_id"]}
        
        prev_intent = prev.get("metadata", {}).get("intent", "")
        curr_intent = curr.get("metadata", {}).get("intent", "")
        if prev_intent == "exploration" and curr_intent == "validation":
            return {"type": "edge", "relation": "tests", "source": prev["node_id"], "target": curr["node_id"]}

        prev_type = prev.get("metadata", {}).get("pipeline_type", "")
        curr_type = curr.get("metadata", {}).get("pipeline_type", "")
        if prev_type == "procedural" and curr_type == "evidential":
            return {"type": "edge", "relation": "validates", "source": prev["node_id"], "target": curr["node_id"]}

        prev_role = prev.get("metadata", {}).get("pipeline_role", "")
        curr_role = curr.get("metadata", {}).get("pipeline_role", "")
        if prev_role == "input" and curr_role == "transform":
            return {"type": "edge", "relation": "transforms", "source": prev["node_id"], "target": curr["node_id"]}

        curr_label = curr.get("discourse", {}).get("label", "")
        prev_label = prev.get("discourse", {}).get("label", "")
        if curr_label == "evidence" and prev_label in ("claim", "hypothesis"):
            return {"type": "edge", "relation": "supports", "source": curr["node_id"], "target": prev["node_id"]}
        if curr_label == "method" and prev_label in ("hypothesis", "background", "claim"):
            return {"type": "edge", "relation": "applies", "source": prev["node_id"], "target": curr["node_id"]}

        return None

    def _detect_section(self, text: str) -> str:
        t = text.lower()
        if "introduction" in t: return "introduction"
        if "method" in t or "approach" in t: return "method"
        if "result" in t: return "result"
        if "conclusion" in t: return "summary"
        return "body"


if __name__ == "__main__":
    import argparse
    import yaml
    import json
    import os
    from pathlib import Path
    from src.pipeline.dag_runtime import DAGRuntime, InMemoryEpistemicGraph
    from src.pipeline.acs_engine import ACSExecutionGovernor

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="theorist")
    parser.add_argument("--runtime-mode", default="train")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    with open(args.manifest, 'r', encoding='utf-8') as f:
        targets = [line.strip() for line in f if line.strip()]

    chunker = AdvancedSemanticChunker(mode=args.mode, runtime_mode=args.runtime_mode)
    
    all_nodes = []
    for target in targets:
        if os.path.exists(target):
            # --- PDF Interceptor: extract text via PyMuPDF for binary PDFs ---
            if target.lower().endswith('.pdf'):
                if not HAS_PYMUPDF:
                    print(f"[SKIP] {target} — PyMuPDF not installed (pip install PyMuPDF)")
                    continue
                try:
                    doc = fitz.open(target)
                    pages = []
                    for page_num, page in enumerate(doc):
                        page_text = page.get_text("text")
                        # OCR fallback for scanned/image-based pages
                        if not page_text.strip() and HAS_TESSERACT:
                            mat = fitz.Matrix(300/72, 300/72)
                            pix = page.get_pixmap(matrix=mat)
                            img = Image.open(_io.BytesIO(pix.tobytes('png')))
                            page_text = pytesseract.image_to_string(img)
                        if page_text.strip():
                            pages.append(page_text)
                    doc.close()
                    text = "\n\n".join(pages)
                    if not text.strip():
                        print(f"[SKIP] {target} — PDF yielded no extractable text")
                        continue
                    print(f"[PDF] {os.path.basename(target)} — {len(pages)} pages, {len(text)} chars extracted")
                except Exception as e:
                    print(f"[ERROR] PDF extraction failed for {target}: {e}")
                    continue
            else:
                with open(target, 'r', encoding='utf-8') as f:
                    text = f.read()
            stream = [(text, {"source_file": target})]
            
            for item in chunker.stream_documents(stream):
                # Filter out yielded edges (The DAG handles them via edge_hints)
                if item.get("type") != "edge":
                    all_nodes.append(item)

    # Handoff to null-safe DAG Runtime
    graph = InMemoryEpistemicGraph(all_nodes)
    acs = ACSExecutionGovernor()
    runtime = DAGRuntime(graph=graph, acs=acs, mode=args.mode)
    
    result = runtime.run()

    # RUN ISOLATION ENFORCED
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "KNOWLEDGE_MATRIX_UNIFIED.yaml"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("validated_nodes:\n")
        for node in result.get("validated_nodes", []):
            node_yaml = yaml.safe_dump([node], default_flow_style=False, sort_keys=False, allow_unicode=True)
            f.write(node_yaml)
        f.write("rejected_nodes:\n")
        for node in result.get("rejected_nodes", []):
            node_yaml = yaml.safe_dump([node], default_flow_style=False, sort_keys=False, allow_unicode=True)
            f.write(node_yaml)
    print(f"[OK] Theorist Compilation complete. Generated {len(all_nodes)} nodes.")
