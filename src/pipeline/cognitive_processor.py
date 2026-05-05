"""
Aletheia Polyglot Cognitive Processor v8.3 (Core Compiler)
Purpose: Single-Responsibility AST Transformation Engine.
         Ingests raw code slices, applies deterministic canonicalization (Black/SQLGlot),
         resolves structural identity, and outputs normalized node payloads.
         [v8.3]: Added explicit structural_completeness signal for downstream filtering.
"""

# --- Deterministic Seeding for Global Reproducibility ---
import os
import random
import numpy as np
os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)
# If torch is used in the environment, uncomment:
# import torch
# torch.manual_seed(42)

import re
import hashlib
import logging
from typing import Any, Callable, Iterator, List, Dict, Optional

# --- Optimization Dependencies ---
try:
    import sqlglot
    from sqlglot.optimizer import optimize as sql_optimize
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False

try:
    import black
    HAS_BLACK = True
except ImportError:
    HAS_BLACK = False

from src.core.compiler_utils import (
    stable_json_dumps as _stable_json_dumps,
)

# --- Epistemic Enricher Integration ---
try:
    from src.interfaces.epistemic_enricher import EpistemicExtractor
    _ENRICHER = EpistemicExtractor()
except Exception:
    _ENRICHER = None

__all__ = ["CognitiveProcessor", "ASTCanonicalizer", "AgnosticNodeBuilder"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
_PLACEHOLDER_TEXTS = {"", "pending analysis", "unknown", "unspecified", "tbd"}
_TEST_PATH_MARKERS = ("test", "tests", "testenv", "spec", "fixture", "mock")
_VALIDATION_SIGNAL_MARKERS = {
    "assert": "assert",
    "self.assert": "self.assert",
    "pytest": "pytest",
    "raises": "raises",
    "expect": "expect",
    "raise ": "raise",
    " error": "error",
}

# --- Utility Functions & Telemetry ---
def safe_execute(label: str, fn: Callable[..., Any], fallback: Any = None, metadata_target: Optional[Dict[Any, Any]] = None) -> Any:
    """Failure Telemetry Injection: Ensures no silent degradation during complex external operations."""
    try:
        return fn()
    except Exception as e:
        err_msg = str(e)
        logger.warning(f"[{label}] failed: {err_msg} - Falling back to default.")
        if metadata_target is not None:
            failures = metadata_target.setdefault("failure_modes", [])
            failures.append({"step": label, "error": err_msg})
        return fallback

def _is_placeholder_text(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    return value.strip().lower() in _PLACEHOLDER_TEXTS

def _clean_text_field(value: Any) -> str:
    if _is_placeholder_text(value):
        return ""
    return str(value).strip()

def _clean_docstring_artifacts(docstring: Any) -> str:
    if not isinstance(docstring, str):
        return ""
    return re.sub(r"[#>*`]+", "", docstring).strip()

def _is_identity_function(source_code: Any) -> bool:
    """Identity Trap: Filters out semantically empty wrappers and constant returns."""
    if not isinstance(source_code, str):
        return False
    normalized = re.sub(r"#.*", "", source_code)
    normalized = re.sub(r"\s+", "", normalized).strip().lower()
    if re.match(r"^def\w+\((\w+)\)(?:->\w+)?:return\1$", normalized): return True
    if re.match(r"^def\w+\(.*\)(?:->\w+)?:(?:pass|return|returnnone)$", normalized): return True
    if re.match(r"^def\w+\((\w+)\)(?:->\w+)?:return\w+\(\1\)$", normalized): return True
    return False

def _is_trivial_stub_source(source_code: Any) -> bool:
    if not isinstance(source_code, str):
        return False
    normalized = re.sub(r"#.*", " ", source_code)
    normalized = re.sub(r"//.*", " ", normalized)
    normalized = re.sub(r"/\*.*?\*/", " ", normalized, flags=re.DOTALL)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    stub_suffixes = (
        ": pass",
        ": ...",
        ": return none",
        ": raise notimplementederror",
        "{ }",
        "{ return ; }",
        "{ return null ; }",
    )
    return any(normalized.endswith(suffix) for suffix in stub_suffixes)

# --- AST Canonicalization ---
class ASTCanonicalizer:
    """Implements First-Class SQL Semantics & Python AST Canonicalization with Telemetry."""
    @staticmethod
    def canonicalize(code: str, file_path: str = "", metadata: Optional[Dict[Any, Any]] = None) -> str:
        if not code:
            return code
        
        code_lower = code.lower()
        is_sql = file_path.endswith(".sql") or any(kw in code_lower for kw in ["select ", "from ", "where ", "join "])
        
        if HAS_SQLGLOT and is_sql:
            def _opt_sql():
                parsed = sqlglot.parse_one(code, read=None)
                return sql_optimize(parsed).sql(pretty=True)
            code = safe_execute("SQLGlot Canonicalization", _opt_sql, fallback=code, metadata_target=metadata)
        
        is_python = file_path.endswith(".py") or "def " in code_lower or "class " in code_lower
        if HAS_BLACK and is_python:
            def _opt_py():
                return black.format_str(code, mode=black.Mode())
            code = safe_execute("Black Canonicalization", _opt_py, fallback=code, metadata_target=metadata)

        return code

# --- Centralized AST Normalization ---
def _normalize_ast_slice(ast_slice: dict) -> dict:
    ast_slice = dict(ast_slice)
    ast_slice["name"] = str(ast_slice.get("name") or "").strip()
    ast_slice["file"] = str(ast_slice.get("file") or "").strip()
    ast_slice["code_snippet"] = str(ast_slice.get("code_snippet") or "").strip()
    ast_slice["imports"] = [str(i).strip() for i in (ast_slice.get("imports") or [])]
    ast_slice["operator_type"] = str(ast_slice.get("operator_type") or "").strip()
    
    if "topology_cluster" in ast_slice and isinstance(ast_slice["topology_cluster"], dict):
        tc = ast_slice["topology_cluster"]
        for k in ("node_ids", "edges", "parent_ids", "child_ids"):
            if k in tc and tc[k] is not None:
                if k == "edges":
                    tc[k] = sorted(tc[k], key=lambda e: (str(e.get("source", "")).strip(), str(e.get("target", "")).strip()))
                else:
                    tc[k] = sorted([str(i).strip() for i in tc[k]])
    return ast_slice

def _detect_skill_type(slice_data: Dict[str, Any]) -> str:
    file_path = str(slice_data.get("file") or "").lower()
    name = str(slice_data.get("name") or "").lower()
    code_snippet = str(slice_data.get("code_snippet") or "").lower()

    if any(marker in file_path for marker in _TEST_PATH_MARKERS):
        return "validation"
    if any(marker in name for marker in _TEST_PATH_MARKERS):
        return "validation"
    if any(marker in code_snippet for marker in ("assert", "self.assert", "pytest", "raises", "expect")):
        return "validation"
    if "raise" in code_snippet and "error" in code_snippet:
        return "validation"
    return "execution"

def _extract_validation_profile(slice_data: Dict[str, Any]) -> Dict[str, Any]:
    code_snippet = str(slice_data.get("code_snippet") or "")
    lowered_code = code_snippet.lower()
    assertion_markers = sorted(
        {
            marker_name
            for marker_text, marker_name in _VALIDATION_SIGNAL_MARKERS.items()
            if marker_text in lowered_code
        }
    )
    assertion_count = sum(lowered_code.count(marker_text) for marker_text in _VALIDATION_SIGNAL_MARKERS.keys())

    coverage_scope: List[str] = []
    if "raise" in lowered_code or "error" in lowered_code or "exception" in lowered_code:
        coverage_scope.append("error_handling")
    if "edge" in lowered_code or "boundary" in lowered_code:
        coverage_scope.append("edge_case")
    if assertion_count > 0:
        coverage_scope.append("behavioral_contract")

    invariant_hints: List[str] = []
    if "assert" in lowered_code:
        invariant_hints.append("asserted output or state invariant")
    if "raises" in lowered_code or "exception" in lowered_code:
        invariant_hints.append("expected failure path is exercised")

    return {
        "assertion_count": assertion_count,
        "assertion_markers": sorted(set(assertion_markers)),
        "coverage_scope": sorted(set(coverage_scope)),
        "invariant_hints": sorted(set(invariant_hints)),
    }

def _normalize_graph_payload(graph_payload: Any) -> Dict[str, Any]:
    graph_payload = graph_payload if isinstance(graph_payload, dict) else {}
    node_ids = sorted({str(node_id).strip() for node_id in (graph_payload.get("node_ids") or []) if str(node_id).strip()})
    parent_ids = sorted({str(node_id).strip() for node_id in (graph_payload.get("parent_ids") or []) if str(node_id).strip()})
    child_ids = sorted({str(node_id).strip() for node_id in (graph_payload.get("child_ids") or []) if str(node_id).strip()})

    edge_items = []
    seen_edges = set()
    for edge in (graph_payload.get("edges") or []):
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if not source or not target:
            continue
        if source == target:
            continue  
        edge_payload = {"source": source, "target": target}
        edge_key = _stable_json_dumps(edge_payload)
        if edge_key not in seen_edges:
            seen_edges.add(edge_key)
            edge_items.append(edge_payload)
    edge_items = sorted(edge_items, key=_stable_json_dumps)

    dependency_count = int(graph_payload.get("dependency_count") or len(edge_items))
    node_count = int(graph_payload.get("node_count") or len(node_ids))
    edge_count = int(graph_payload.get("edge_count") or len(edge_items))
    
    max_edges = max(node_count * (node_count - 1), 1)
    graph_density = float(graph_payload.get("graph_density") or (edge_count / max_edges))

    return {
        "node_ids": node_ids,
        "edges": edge_items,
        "parent_ids": parent_ids,
        "child_ids": child_ids,
        "node_count": node_count,
        "edge_count": edge_count,
        "dependency_count": dependency_count,
        "graph_density": round(graph_density, 3),
    }

# --- 1. PolyglotOntology (Core Structural Classes Only) ---
class PolyglotOntology:
    """Maps language-specific raw operations to semantic ontologies (Heuristic Fallback Layer)."""
    @classmethod
    def classify(cls, operation_name: str) -> str:
        """Legacy classifier for standard lineage parsing before dense enrichment."""
        if not operation_name: return "compute.generic"
        op_lower = operation_name.lower()
        if any(kw in op_lower for kw in ["cuda", "gpu", "tensor", "cufft", "torch", "jax"]): return "compute.accelerated"
        if any(kw in op_lower for kw in ["sql", "db", "mongo", "redis", "query"]): return "io.database"
        if any(kw in op_lower for kw in ["http", "request", "fetch", "socket"]): return "io.network"
        return "compute.generic"

# --- 2. Skill Identity Resolver ---
class SkillIdentityResolver:
    """Generates deterministic, globally unique identifiers via Dual Identity architecture."""
    @staticmethod
    def resolve(semantics: Dict[str, Any], identity: Dict[str, Any], graph: Dict[str, Any], canonical_code: str = "", raw_code: str = "", validation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        validation = validation or {}
        ops = [str(op).strip() for op in (semantics.get("operator_types") or []) if str(op).strip()]
        side_effects = sorted(str(effect).strip() for effect in (semantics.get("side_effects") or []) if str(effect).strip())
        
        category = "general.compute"
        ops_str = " ".join(ops).lower()
        if any(kw in ops_str for kw in ["network", "request", "http", "socket"]):
            category = "io.network"
        elif any(kw in ops_str for kw in ["db", "sql", "query", "mongo"]):
            category = "io.database"
        elif any(kw in ops_str for kw in ["tensor", "math", "cuda", "gpu"]):
            category = "compute.accelerated"
            
        topology_hash = hashlib.sha256(_stable_json_dumps(graph.get("edges", [])).encode('utf-8')).hexdigest()[:10]
        raw_hash = hashlib.sha256(raw_code.encode('utf-8')).hexdigest()[:8]
        can_hash = hashlib.sha256(canonical_code.encode('utf-8')).hexdigest()[:8]
        composite_code_hash = f"{raw_hash}_{can_hash}"
        
        canonical_payload = {
            "ops_bag": sorted(ops),
            "ops_sequence": ops,
            "topology_hash": topology_hash,
            "code_hash": composite_code_hash,
            "arity": len(ops),
            "node_count": graph.get("node_count", 1),
            "side_effects": side_effects,
            "validation_signature": validation.get("assertion_count", 0),
        }
        canonical_sig = _stable_json_dumps(canonical_payload)
        canonical_hash = hashlib.sha256(canonical_sig.encode('utf-8')).hexdigest()[:10].upper()
        canonical_id = f"SKILL_{category.replace('.', '_').upper()}_{canonical_hash}"

        instance_payload = {
            "canonical_id": canonical_id,
            "file": identity.get("file", ""),
            "name": identity.get("name", "")
        }
        instance_sig = _stable_json_dumps(instance_payload)
        instance_hash = hashlib.sha256(instance_sig.encode('utf-8')).hexdigest()[:8].upper()
        instance_id = f"INST_{instance_hash}"
        
        return {
            "canonical_id": canonical_id,
            "instance_id": instance_id,
            "category": category,
            "aliases": [identity.get("name", "anonymous_operation")]
        }

# --- 3. AgnosticNodeBuilder ---
class AgnosticNodeBuilder:
    """Normalizes AST slices into clean, deduplicated Structural Nodes."""

    @staticmethod
    def _enrich_reasoning_vectors(code_snippet: str, function_name: str = "", docstring: str = "") -> Dict[str, Any]:
        """Derives reasoning vectors via EpistemicExtractor, with safe fallback to stubs."""
        fallback = {"intent": "Derived execution intent", "strategy": "AST Translation", "constraints": ["Implicit environment boundaries apply"], "execution_pattern": ["Execute sequence"], "failure_modes": ["Unhandled Runtime Exception"]}
        if _ENRICHER is None:
            return fallback
        try:
            return _ENRICHER.enrich_node(code_snippet=code_snippet, function_name=function_name, docstring=docstring, ocr_excerpt="")
        except Exception as e:
            logger.warning(f"[EpistemicExtractor] Failed to enrich node: {e}")
            return fallback

    @staticmethod
    def _normalize_code_snippet(code_snippet: Any, file_path: str = "", metadata: Optional[Dict[Any, Any]] = None) -> str:
        """Strips line-number prefixes and applies AST canonicalization."""
        if not isinstance(code_snippet, str):
            return ""

        cleaned_lines = [
            re.sub(r"^\s*\d+\s*\|\s?", "", line.rstrip())
            for line in code_snippet.splitlines()
        ]
        cleaned_code = "\n".join(cleaned_lines).strip()
        return ASTCanonicalizer.canonicalize(cleaned_code, file_path, metadata)

    @staticmethod
    def build_from_ast(ast_slice: Dict[str, Any], lineage: List[Dict[str, Any]], risk_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        risk_meta = risk_meta or {}
        base_id = ast_slice.get("slice_id", f"AST_{hashlib.md5(ast_slice.get('code_snippet', '').encode()).hexdigest()[:8]}")
        
        canonical_meta: Dict[str, Any] = {}
        file_path = str(ast_slice.get("file", "")).strip()
        raw_code = str(ast_slice.get("code_snippet", ""))
        canonical_code = AgnosticNodeBuilder._normalize_code_snippet(raw_code, file_path, canonical_meta)
        ast_slice["code_snippet"] = canonical_code 
        
        skill_type = _detect_skill_type(ast_slice)
        validation_profile = _extract_validation_profile(ast_slice)
        topology_cluster = ast_slice.get("topology_cluster", {}) if isinstance(ast_slice.get("topology_cluster", {}), dict) else {}
        node_graph = _normalize_graph_payload(
            {
                "node_ids": topology_cluster.get("node_ids", [base_id]),
                "edges": topology_cluster.get("edges", []),
                "parent_ids": topology_cluster.get("parent_ids", []),
                "child_ids": topology_cluster.get("child_ids", []),
                "node_count": topology_cluster.get("node_count", len(topology_cluster.get("node_ids", [base_id]) or [])),
                "edge_count": topology_cluster.get("edge_count", len(topology_cluster.get("edges", []) or [])),
                "dependency_count": len(topology_cluster.get("parent_ids", []) or []) + len(topology_cluster.get("child_ids", []) or []),
                "graph_density": topology_cluster.get(
                    "graph_density",
                    len(topology_cluster.get("edges", []) or []) / max(len(topology_cluster.get("node_ids", [base_id]) or []), 1),
                ),
            }
        )
        
        identity_meta = {
            "name": ast_slice.get("name", "unknown"),
            "file": file_path,
            "lines": f"{ast_slice.get('start_line', 0)}-{ast_slice.get('end_line', 0)}",
            "package_dependencies": ast_slice.get("imports", [])
        }
        
        semantics = {
            "operator_types": [step.get('operation_class') for step in lineage] or [ast_slice.get("operator_type", "generic")],
            "return_types": ast_slice.get("inferred_type", "Any"),
            "side_effects": ast_slice.get("side_effects", [])
        }
        
        risk_level = risk_meta.get("risk_level", "LOW")
        
        structural_completeness = all([
            len(semantics.get("operator_types", [])) > 0,
            len(node_graph.get("node_ids", [])) > 0,
            bool(canonical_code.strip())
        ])
        
        # Initial structural constraint tagging for downstream ACS/DAG consumption
        initial_constraints = []
        if not canonical_code.strip():
            initial_constraints.append({
                "type": "structural", "description": "Empty code snippet — no parseable source",
                "valid": False, "severity": "error", "tags": ["empty_source"]
            })
        if not identity_meta.get("package_dependencies"):
            initial_constraints.append({
                "type": "structural", "description": "No imports declared — potential missing dependencies",
                "valid": True, "severity": "info", "tags": ["no_imports"]
            })
        if canonical_code.strip() and "return" not in canonical_code and ast_slice.get("type") == "function":
            initial_constraints.append({
                "type": "structural", "description": "Function lacks explicit return statement",
                "valid": True, "severity": "warning", "tags": ["no_return"]
            })
        if canonical_code.strip():
            try:
                compile(canonical_code, "<ast_check>", "exec")
            except SyntaxError:
                initial_constraints.append({
                    "type": "structural", "description": "Code fails syntax validation",
                    "valid": False, "severity": "error", "tags": ["syntax_error"]
                })
        
        # Build core structural node (Epistemic Reasoning to be injected downstream)
        return {
            "node_id": base_id,
            "name": identity_meta.get("name", "unknown"),
            "file": file_path,
            "imports": identity_meta.get("package_dependencies", []),
            "operator_type": semantics["operator_types"][0] if semantics.get("operator_types") else "unknown",
            "skill_type": skill_type,
            "type": ast_slice.get("type", "function"),
            "source_type": "ast_code",
            "code_snippet": canonical_code,
            "topology_cluster": topology_cluster,
            "identity": identity_meta,
            "source_context": {
                "docstring": _clean_docstring_artifacts(_clean_text_field(ast_slice.get("docstring", "")))
            },
            "graph": node_graph,
            "validation": validation_profile,
            "semantics": semantics,
            "skill_identity": SkillIdentityResolver.resolve(semantics, identity_meta, node_graph, canonical_code, raw_code, validation_profile),
            "dependencies": {
                "downstream_calls": ast_slice.get("calls", []),
                "upstream_callers": ast_slice.get("called_by", [])
            },
            "state_and_execution": {
                "mutation_tracking": ast_slice.get("mutations", []),
                "side_effects": semantics["side_effects"]
            },
            "constraints": {
                "risk_level": risk_level,
                "initial_constraints": initial_constraints
            },
            "metadata": {
                "canonicalization_failures": canonical_meta.get("failure_modes", []),
                "structural_completeness": structural_completeness
            },
            "teaching_layer": {"skill_identity": {}, "method_metadata": {}, "reasoning_vectors": AgnosticNodeBuilder._enrich_reasoning_vectors(canonical_code, identity_meta.get("name", "unknown"), _clean_docstring_artifacts(_clean_text_field(ast_slice.get("docstring", "")))), "implementation_template": {"code": canonical_code}},
            "epistemic_core": {},
            "epistemic": {"state": "unresolved", "c_node": 0.5, "retry_budget": 6, "depth": 0, "branch_id": "root", "evidence_refs": []} # Placeholder for downstream epistemic_enricher.py
        }


# --- 4. Main Compiler Orchestrator ---
class CognitiveProcessor:
    """Core Compiler - Strictly limits scope to building pure, canonicalized AST nodes."""
    def __init__(self, input_data: Dict[str, Any]):
        self.input_data = input_data
        self._validate_input()
        self.raw_slices = self._ingest_slices()
        self.risk_meta = self.input_data.get("layer_2_7_patch_collision_risk", {})

    def _validate_input(self) -> None:
        if not isinstance(self.input_data, dict):
            logger.error("[ERROR] Input data is not a valid dictionary.")
            raise ValueError("Invalid Input Data")
        if "layer_2_intelligence" not in self.input_data and "slices" not in self.input_data:
            logger.warning("[WARNING] Input bundle lacks 'layer_2_intelligence' or 'slices'. Output may be empty.")

    def _ingest_slices(self) -> Dict[str, Any]:
        slices: Dict[str, Any] = {}
        layer_2 = self.input_data.get("layer_2_intelligence", [])
        for file_data in layer_2:
            for slice_obj in file_data.get("slices", []):
                if "slice_id" in slice_obj:
                    slices[slice_obj["slice_id"]] = slice_obj
        return slices if slices else self.input_data.get("slices", {})

    def _extract_data_lineage(self, slice_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        lineage: List[Dict[str, Any]] = []
        calls = slice_data.get("calls") or []
        mutations = slice_data.get("mutations") or []
        for i, call in enumerate(calls):
            semantic_class = PolyglotOntology.classify(str(call))
            output_var = mutations[0] if (i == len(calls) - 1 and mutations) else f"internal_state_{i}"
            input_var = "args" if i == 0 else f"internal_state_{i-1}"
            lineage.append({
                "step": i + 1,
                "operation_class": semantic_class,
                "raw_call": str(call),
                "inputs": [input_var],
                "outputs": [output_var],
                "control": "sequential"
            })
        return lineage

    def _build_nodes(self) -> Iterator[Dict[str, Any]]:
        name_to_id = {str(s_data.get("name", "")).strip(): slice_id for slice_id, s_data in self.raw_slices.items() if s_data.get("name")}

        # Batch intent pre-computation: one model.encode() call for all nodes instead of N.
        # Monkey-patches extract_intent to serve cached results during the build loop.
        _intent_map: Dict[str, str] = {}
        if _ENRICHER is not None and getattr(_ENRICHER, "model", None) is not None:
            try:
                from sentence_transformers import util as _st_util
                _sid_order = list(self.raw_slices.keys())
                _queries = [
                    f"{self.raw_slices[sid].get('name', '')} "
                    f"{_clean_docstring_artifacts(_clean_text_field(self.raw_slices[sid].get('docstring', '')))}"
                    .strip()
                    for sid in _sid_order
                ]
                _batch_embs = _ENRICHER.model.encode(
                    _queries, batch_size=64, convert_to_tensor=True, show_progress_bar=False
                )
                for sid, emb in zip(_sid_order, _batch_embs):
                    best, best_score = "unknown.compute", -1.0
                    for label, ont_emb in _ENRICHER.ontology_embeddings.items():
                        score = _st_util.cos_sim(emb, ont_emb).item()
                        if score > best_score:
                            best_score, best = score, label
                    _intent_map[sid] = best if best_score >= 0.85 else "unknown.compute"
                logger.info(f"[BatchEncode] Pre-computed intents for {len(_intent_map)} nodes in one pass.")
                # Free tensor and query list — intent_map is the only keeper.
                del _sid_order, _queries, _batch_embs
            except Exception as _be:
                logger.warning(f"[BatchEncode] Batch pre-computation failed, falling back to per-node: {_be}")

        _current_sid: List[Optional[str]] = [None]
        _orig_extract_intent = None
        if _ENRICHER is not None and _intent_map:
            _orig_extract_intent = _ENRICHER.extract_intent
            def _lookup_intent(function_name: str, docstring: str, ocr_excerpt: str) -> str:
                return _intent_map.get(_current_sid[0], "unknown.compute")
            _ENRICHER.extract_intent = _lookup_intent

        try:
            for slice_id, s_data in self.raw_slices.items():
                _current_sid[0] = slice_id
                s_data = _normalize_ast_slice(s_data)
                slice_risk = self.risk_meta.get(slice_id, {})
                lineage = self._extract_data_lineage(s_data)

                s_data["slice_id"] = slice_id
                orig_calls = s_data.get("calls") or []
                orig_called_by = s_data.get("called_by") or []

                unresolved = []
                def resolve_call(c):
                    if c is None: return "None"
                    c_str = str(c)
                    if c_str in self.raw_slices: return c_str
                    if c_str in name_to_id: return name_to_id[c_str]
                    unresolved.append(c_str)
                    return c_str

                s_data["calls"] = [resolve_call(c) for c in orig_calls]
                s_data["called_by"] = [resolve_call(c) for c in orig_called_by]

                if unresolved:
                    s_data.setdefault("metadata", {})["unresolved_calls"] = sorted(set(unresolved))

                if (_is_trivial_stub_source(s_data.get("code_snippet", ""))
                        or _is_identity_function(s_data.get("code_snippet", ""))):
                    continue

                node = AgnosticNodeBuilder.build_from_ast(ast_slice=s_data, lineage=lineage, risk_meta=slice_risk)
                yield node
        finally:
            if _orig_extract_intent is not None:
                _ENRICHER.extract_intent = _orig_extract_intent

    def compile(self) -> List[Dict[str, Any]]:
        """
        Executes Phase 1 Compilation.
        Returns a clean list of canonicalized AST nodes ready for Downstream Graph/Enrichment layers.
        """
        logger.info("[*] Cognitive Processor: Compiling AST Slices into Canonical Nodes...")
        semantic_nodes = sorted(
            self._build_nodes(),
            key=lambda n: (
                str(n.get("skill_identity", {}).get("canonical_id", "")),
                str(n.get("node_id", "")),
            ),
        )
        logger.info(f"[OK] Successfully compiled {len(semantic_nodes)} structured AST nodes.")
        return semantic_nodes

if __name__ == "__main__":
    import argparse
    import gc
    import os
    import yaml
    import json
    from pathlib import Path
    from src.pipeline.dag_runtime import DAGRuntime, InMemoryEpistemicGraph
    from src.pipeline.acs_engine import ACSExecutionGovernor

    # ALETHEIA_CHUNK_SIZE: max nodes per DAG segment. Controls peak RSS.
    # Lower = safer RAM, weaker within-chunk topology scores. Default 500 ≈ 200 MB peak.
    MAX_BUFFER = int(os.environ.get("ALETHEIA_CHUNK_SIZE", "500"))

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Raw AST JSON bundle")
    parser.add_argument("-o", "--output", required=True, help="Output YAML dossier")
    parser.add_argument("--agent-prefix", default="LogosAgent")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    processor = CognitiveProcessor(data)
    all_nodes = processor.compile()
    total = len(all_nodes)
    print(f"[INFO] {total} canonical nodes compiled. Chunk size: {MAX_BUFFER}.")
    del data, processor
    gc.collect()

    # Sort roots-first (fewest upstream_callers first) so the ghost pre-rejection pass
    # below catches transitive chains (A→B→C) in a single left-to-right sweep per chunk.
    all_nodes.sort(key=lambda n: (
        len(n.get("dependencies", {}).get("upstream_callers", [])),
        str(n.get("node_id", ""))
    ))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path = out_path.with_suffix(".rejected.yaml")

    total_validated = 0
    total_rejected = 0
    num_chunks = (total + MAX_BUFFER - 1) // MAX_BUFFER

    # node_ids of every node rejected so far (by DAGRuntime OR by ghost pre-rejection).
    # Persists across chunks to implement CASCADE_FROM_PARENT without touching DAGRuntime.
    global_rejected_ids: set = set()

    class _NoAliasDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True

    def _stream_node(f, node: dict) -> None:
        item_yaml = yaml.dump(
            node, Dumper=_NoAliasDumper, default_flow_style=False,
            sort_keys=False, allow_unicode=True
        )
        lines = item_yaml.rstrip("\n").split("\n")
        f.write("- " + lines[0] + "\n")
        for line in lines[1:]:
            f.write("  " + line + "\n")

    with open(out_path, "w", encoding="utf-8") as f_out, \
         open(rejected_path, "w", encoding="utf-8") as f_rej:

        f_out.write("validated_nodes:\n")
        f_rej.write("rejected_nodes:\n")

        for chunk_idx in range(num_chunks):
            start = chunk_idx * MAX_BUFFER
            chunk = all_nodes[start : start + MAX_BUFFER]

            # --- Ghost Pre-Rejection ---
            # Iterates the depth-sorted chunk left-to-right. Because parents always appear
            # before their children in the sorted order, adding a node_id to
            # global_rejected_ids mid-loop automatically catches transitive descendants
            # (A rejected → B added → C caught in the same pass, no second sweep needed).
            surviving = []
            ghost_count = 0
            for node in chunk:
                node_id = node.get("node_id", "")
                callers = node.get("dependencies", {}).get("upstream_callers", [])
                if any(caller_id in global_rejected_ids for caller_id in callers):
                    global_rejected_ids.add(node_id)
                    ghost_node = dict(node)
                    ghost_node["reason"] = "cross-chunk cascade failure"
                    _stream_node(f_rej, ghost_node)
                    ghost_count += 1
                    total_rejected += 1
                else:
                    surviving.append(node)

            print(
                f"  [->] Segment {chunk_idx + 1}/{num_chunks}: "
                f"{len(chunk)} nodes  ({ghost_count} ghost-rejected, "
                f"{len(surviving)} passed to runtime)"
            )

            # --- DAGRuntime Pass ---
            # Guard: skip runtime entirely if all chunk nodes were ghost-pre-rejected.
            if surviving:
                graph = InMemoryEpistemicGraph(surviving)
                acs = ACSExecutionGovernor()
                runtime = DAGRuntime(graph=graph, acs=acs, mode="coding_assistant")
                chunk_result = runtime.run()

                # TODO: identity_continuity — DAGRuntime creates IdentityManager internally
                # with no restore API. Cross-chunk drift accumulation requires a future
                # DAGRuntime constructor extension: initial_identity_state=snapshot.
                identity_snap = chunk_result.get("identity_state")
                if identity_snap:
                    logger.info(
                        f"[Chunk {chunk_idx + 1}] identity_state "
                        f"drift={identity_snap.get('drift_score', 'N/A')}"
                    )

                for node in chunk_result.get("validated_nodes", []):
                    _stream_node(f_out, node)
                total_validated += len(chunk_result.get("validated_nodes", []))

                # Propagate runtime rejections into global_rejected_ids so future chunks
                # ghost-pre-reject their descendants in the next iteration.
                newly_rejected = chunk_result.get("rejected_nodes", [])
                global_rejected_ids.update(
                    rn.get("node_id", "") for rn in newly_rejected if rn.get("node_id")
                )
                for node in newly_rejected:
                    _stream_node(f_rej, node)
                total_rejected += len(newly_rejected)

                del graph, acs, runtime, chunk_result

            # Flush per-chunk so partial results survive an OOM mid-run.
            f_out.flush()
            f_rej.flush()

            del chunk, surviving
            gc.collect()

    print(f"[OK] {total_validated} validated, {total_rejected} rejected "
          f"({len(global_rejected_ids)} unique IDs in rejection set).")
    print(f"[OK] Capability Dossier: {out_path}")
    print(f"[OK] Rejected nodes: {rejected_path}")
