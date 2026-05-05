"""
Aletheia Advanced Semantic Compiler v3.5 (Theorist Creator Mode)
The Self-Optimizing Epistemic System

Upgrades (v3.5 Reinforcement Learning & System Stability):
- Epistemic Reward Alignment: Introduced `signal_integrity` and complexity penalties to prevent gameable proxies.
- Graph Quality Metric: Connectivity is now based on unique edge targets to penalize edge spamming.
- System State Representation: Tracks modes (`exploration`, `exploitation`, `recovery`) via rolling memory.
- Epsilon-Exploration (ε-greedy): Bounded randomness to prevent local maxima lock.
- Ontology Validation Gate: Prevents semantic drift by requiring recurrence before learning new predicates.
- Semantic Deterministic ID: Cryptographic identity now anchors to canonical reasoning, not OCR text.
- Control Loop Dead Zone: Tolerance bands stabilize parameters and prevent micro-oscillation.
"""

import re
import spacy
import logging
import uuid
import hashlib
import random
from collections import Counter, deque
from typing import Iterator, Dict, Any, Tuple, Iterable, Optional, List

# --- AMR Dependencies ---
try:
    import amrlib
    import penman
    HAS_AMR = True
except ImportError:
    HAS_AMR = False

logging.basicConfig(level=logging.INFO)
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
    Constructs the authoritative Node Schema v3.5.
    Enforces the Epistemic & Executable Contract Layer for `AletheiaSkill`.
    """
    def __init__(
        self, text: str, frames: List[Dict[str, Any]], discourse: Dict[str, Any], 
        graph: Optional[Dict[str, Any]], section: str, source_file: str, index: int,
        pipeline_type: str, pipeline_role: str, intent: str, lens: str, 
        verifiability: str, mode: str, config: Dict[str, Any], 
        previous_node: Optional[Dict[str, Any]] = None
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

        # Confidence Decay: Longer complex nodes have inherently looser structure
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

        # Advanced complexity formula incorporating density and argument load
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

    def _build_legacy_layer(self) -> Dict[str, Any]:
        intents, strategies, patterns = [], [], []
        for f in self.frames:
            action = f["predicate"]["lemma"].upper()
            agents = [a["text"] for a in f["arguments"] if a["role"] == "agent"]
            themes = [a["text"] for a in f["arguments"] if a["role"] == "theme"]
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
                "constraints": ["pending_advocate_review"],    # List[str] per EpistemicReasoning contract
                "execution_pattern": patterns if patterns else ["STATIC_RELATION"],  # List[str]
                "failure_modes": ["pending_advocate_review"],  # List[str]
            }
        }

    def build(self) -> Dict[str, Any]:
        semantic_signature = self._build_semantic_signature()
        current_hash = semantic_signature["hash"]
        
        # Multi-Hop Context Chaining
        if self.previous_node and "metadata" in self.previous_node:
            prev_context = self.previous_node["metadata"].get("context_signature")
            if prev_context:
                self.context_signature = hashlib.sha256(f"{prev_context}:{current_hash}".encode('utf-8')).hexdigest()[:16]
            else:
                prev_hash = self.previous_node["semantics"]["semantic_signature"]["hash"]
                self.context_signature = hashlib.sha256(f"{prev_hash}:{current_hash}".encode('utf-8')).hexdigest()[:16]
        else:
            self.context_signature = current_hash

        # Cache expensive computations to prevent redundant calls
        constraints = self._build_constraints()
        confidence_profile = self._build_confidence_profile()
        complexity = self._build_complexity()
        
        return {
            # --- MANDATORY ALETHEIASKILL FIELDS (models.py Compatibility) ---
            "node_id": self.node_id,
            "name": f"edu_{self.index}_{self.pipeline_type}",
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

            # --- CORE V3.5 ARCHITECTURE FIELDS ---
            "metadata": {
                "deterministic_id": self._build_deterministic_id(),
                "schema_version": "text_semantic_node_v3",
                "compiler_version": "Aletheia_ASC_v3.5",
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
            }
        }


class AdvancedSemanticChunker:

    def __init__(self, model_name: str = "en_core_web_trf", mode: str = "theorist"):
        self.mode = mode
        
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
            "unique_edge_targets": set(), # Used for structural connectivity
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
            logger.info(f"[OK] Loaded {model_name}")
        except OSError:
            logger.warning("[!] Falling back to en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

        if HAS_AMR:
            amrlib.setup_spacy_extension()

    def _update_config(self, key: str, delta: float, min_val: float, max_val: float):
        """Bounded + Smoothed Adaptation to prevent feedback drift."""
        current = self.config[key]
        new_val = max(min_val, min(max_val, current + delta))
        # Exponential smoothing prevents runaway oscillation
        self.config[key] = round((current * 0.7) + (new_val * 0.3), 4)

    # =========================
    # FORMAL OPTIMIZATION OBJECTIVE (THE CONTROL LOOP)
    # =========================

    def evaluate_and_optimize(self, downstream_rejection_logs: List[Dict[str, Any]] = None) -> float:
        """
        The formal objective function + PID control loop.
        Calculates the Epistemic Reward and automatically tunes compiler thresholds.
        """
        logs = downstream_rejection_logs or []
        self.metrics["downstream_rejections"] += len(logs)
        
        if self.metrics["nodes_processed"] == 0:
            return 0.0

        # 1. Calculate Target Metrics
        nodes_acc = max(self.metrics["nodes_accepted"], 1)
        yield_rate = self.metrics["nodes_accepted"] / self.metrics["nodes_processed"]
        avg_conf = self.metrics["total_confidence"] / nodes_acc
        avg_complexity = self.metrics["total_complexity"] / nodes_acc
        rejection_rate = self.metrics["downstream_rejections"] / nodes_acc
        
        # Graph Structure Quality (replaces naive edge counting)
        connectivity = min(1.0, len(self.metrics["unique_edge_targets"]) / nodes_acc)
        
        # Signal Integrity Term (punishes high-yield junk)
        signal_integrity = min(1.0, avg_conf * (1.0 - rejection_rate))
        complexity_penalty = min(0.3, avg_complexity * 0.05)
        
        # 2. The Objective Function (Reward)
        reward = (
            (yield_rate * 0.25) + 
            (avg_conf * 0.30) + 
            (connectivity * 0.20) + 
            (signal_integrity * 0.25)
        ) - (rejection_rate * 0.6) - complexity_penalty
        
        current_reward = round(max(0.0, min(1.0, reward)), 4)
        self.reward_history.append(current_reward)
        
        logger.info(f"[*] Epistemic Optimization Run. Reward: {current_reward} (Yield: {yield_rate:.2f}, Conf: {avg_conf:.2f}, Conn: {connectivity:.2f}, Integrity: {signal_integrity:.2f})")
        
        # System Alerts (Failure Mode Detection)
        if yield_rate > 0.9 and avg_conf < 0.5:
            logger.warning("⚠️ SYSTEM ALERT: High yield / low confidence drift detected (Garbage Ingestion).")
            self.system_state["mode"] = "recovery"
        elif connectivity > 0.9 and len(self.metrics["unique_edge_targets"]) < (self.metrics["edges_generated"] * 0.1):
            logger.warning("⚠️ SYSTEM ALERT: Graph collapse detected (Edge Spamming).")
        
        # 3. Temporal Memory & State Representation
        if len(self.reward_history) >= 6:
            trend = sum(list(self.reward_history)[-3:]) - sum(list(self.reward_history)[:3])
            self.system_state["drift"] = trend
            if trend < -0.05:
                self.system_state["mode"] = "recovery"
            elif abs(trend) < 0.01:
                self.system_state["mode"] = "exploration"
            else:
                self.system_state["mode"] = "exploitation"

        # 4. PID-Style Control Loop with Dead Zone
        reward_delta = current_reward - self.optimizer_state["last_reward"]
        
        if abs(reward_delta) < 0.01 and self.system_state["mode"] != "exploration":
            logger.info("[*] Reward within dead zone tolerance. Stabilizing system state.")
        else:
            if reward_delta < 0:
                logger.info("[!] Reward decreased. Reversing momentum and annealing step size.")
                for key in self.optimizer_state["momentum"]:
                    self.optimizer_state["param_impact"][key] -= 0.1 # Negative attribution
                    self.optimizer_state["momentum"][key] *= -0.5 
            else:
                for key in self.optimizer_state["momentum"]:
                    self.optimizer_state["param_impact"][key] += 0.1 # Positive attribution
                    self.optimizer_state["momentum"][key] *= 1.1 

            # Apply Exploration vs Exploitation balance (ε-exploration)
            if self.system_state["mode"] == "exploration" or random.random() < 0.1:
                logger.info("[*] Executing ε-exploration step to prevent local maxima lock.")
                self._update_config("frame_min_pred_confidence", random.uniform(-0.1, 0.1), 0.5, 0.95)
                self._update_config("tvg_max_upper_ratio", random.uniform(-0.05, 0.05), 0.2, 0.8)
            else:
                self._update_config("tvg_max_upper_ratio", self.optimizer_state["momentum"]["tvg_max_upper_ratio"], 0.2, 0.8)
                self._update_config("frame_max_arg_tokens", self.optimizer_state["momentum"]["frame_max_arg_tokens"], 10, 40)
                self._update_config("frame_min_pred_confidence", self.optimizer_state["momentum"]["frame_min_pred_confidence"], 0.5, 0.95)
        
        # 5. Ontology Learning Gate (Protected from pollution)
        new_preds = 0
        pred_counts = Counter(log.get("predicate") for log in logs if log.get("reason") == "unknown_predicate" and "predicate" in log)
        for pred, count in pred_counts.items():
            if count > 5: # Strict validation gate
                self.predicate_classes["execution"].add(pred)
                new_preds += 1
                
        if new_preds > 0:
            logger.info(f"    -> Learned {new_preds} new execution predicates from verified ACS feedback.")
                
        self.optimizer_state["last_reward"] = current_reward
        logger.info(f"[*] Active Configuration State: {self.config} | System Mode: {self.system_state['mode']}")
        
        # Reset ephemeral metrics
        self.metrics = {
            "nodes_processed": 0,
            "nodes_accepted": 0,
            "edges_generated": 0,
            "unique_edge_targets": set(),
            "total_confidence": 0.0,
            "total_complexity": 0.0,
            "downstream_rejections": 0
        }
        
        return current_reward

    # =========================
    # TEXT VALIDITY GATE (TVG)
    # =========================

    def _is_index_block(self, text: str) -> bool:
        keywords = ["index", "contents", "volume", "edited by", "bibliography"]
        text_lower = text.lower()
        word_count = len(text.split())
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

        # Reject pure numeric / OCR fragments before they become nodes
        if re.fullmatch(r"[0-9\.\-\s\,\+\[\]\(\)°%]+", text):
            return False

        # Reject structural markers (table headers, figure captions, page numbers)
        if re.search(r"\b(table|fig|figure|page)\b", text, re.I):
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
                    previous_node=previous_node
                ).build()
                
                # Update epistemic metrics for Objective Function
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