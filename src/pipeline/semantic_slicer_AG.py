"""
Aletheia Workspace Packager (CLI) v6.0 - Universal Polyglot Context Engine
Purpose: Produce verifiable workspace bundles with deterministic hashing, syntax validation,
         safe traversal, PEM/entropy redaction, polyglot summarization, and AST slicing.

Phase 1 Improvements:
    - Bundle Hash (SHA256) for context verification across agents
    - File Fingerprints (SHA1 + mtime) for stale bundle detection
    - Syntax Validity Tracking with warnings

Phase 2 Improvements:
    - Import graph extraction
    - Entry-point detection (if __name__ == "__main__" or main())
    - Execution flow reconstruction

Phase 3 & 4:
    - Multi-agent patch collision risk
    - Batch chunking for large repositories (<97MB limits)

Phase 5 (v6.0) UPGRADE:
    - Universal Tree-sitter parsing engine
    - Polyglot support (Python, C++, Rust)
    - Normalized abstract syntax slices across languages

Phase 6 (Enterprise Scale) UPGRADE:
    - Distributed Compute Grids (Ray)
    - True BPE Tokenization (tiktoken)
    - First-Class SQL Semantics & Lineage (SQLGlot)
    - Semantic AST Diffing
"""

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime
import hashlib
import json
import logging
import math
import mimetypes
import os
import pathlib
import re
import sys
import csv
import subprocess
import networkx as nx
from typing import Any, Callable, Dict, List, Literal, Optional, Set, TypedDict

# ============================================================================
# OPTIONAL ENTERPRISE DEPENDENCIES
# ============================================================================
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

try:
    import fitz  # PyMuPDF — PDF text extraction for Theorist Mode
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

try:
    import sqlglot
    from sqlglot.optimizer import optimize as sql_optimize
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False


# ============================================================================
# TREE-SITTER POLYGLOT SETUP
# ============================================================================
HAS_TREESITTER = False
TS_PYTHON = False
TS_CPP = False
TS_RUST = False
TS_JAVA = False
TS_RUBY = False
TS_CSHARP = False

try:
    from tree_sitter import Language, Parser
    HAS_TREESITTER = True
    
    try:
        import tree_sitter_python
        TS_PYTHON = True
    except ImportError:
        pass

    try:
        import tree_sitter_cpp
        TS_CPP = True
    except ImportError:
        pass

    try:
        import tree_sitter_rust
        TS_RUST = True
    except ImportError: 
        pass

    try: import tree_sitter_java; TS_JAVA = True
    except ImportError: TS_JAVA = False
    
    try: import tree_sitter_ruby; TS_RUBY = True
    except ImportError: TS_RUBY = False
    
    try: import tree_sitter_c_sharp; TS_CSHARP = True
    except ImportError: TS_CSHARP = False

except ImportError:
    pass


# ============================================================================
# PHASE 1 CONSTANTS
# ============================================================================
MAX_FILE_SIZE_BYTES = 1_500_000
ENABLE_DETERMINISTIC_HASH = True  # Omit timestamp for reproducible bundles
MAX_WORKERS_DEFAULT = 8
MAX_WORKERS_LIMIT = 32
AST_CACHE_VERSION = "stage6-v6.0"
BUNDLE_SCHEMA_VERSION = "aletheia_bundle_v6.0"
MAX_CLUSTER_NODES = 25
MAX_CLUSTER_EDGES = 100
SYSTEM_PURPOSE = (
    "This system simulates nonlinear complex field dynamics with geometry feedback "
    "to test the hypothesis that prime-log spectral locking emerges in a conformal "
    "scalar field system."
)

IGNORE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".tiff", ".zip",
    ".gz", ".tar", ".tgz", ".bz2", ".xz", ".exe", ".dll", ".so",
    ".dylib", ".bin", ".class", ".pyc"
}

# PDF files are binary but handled via PyMuPDF extraction — not ignored.
PDF_EXTENSIONS = {".pdf"}

IGNORE_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    ".idea", ".vscode", "dist", "build", ".mypy_cache"
}


# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure structured logging with appropriate verbosity."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    
    logger = logging.getLogger("semantic_slicer")
    logger.setLevel(log_level)
    logger.handlers.clear()  # Remove defaults
    logger.addHandler(handler)
    
    return logger

logger = logging.getLogger("semantic_slicer")


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================
class SkipDetails(TypedDict):
    binary_or_ext: int
    oversize: int
    outside_base: int


class ScanStats(TypedDict):
    bundled: int
    skipped: int
    errors: int
    skipped_details: SkipDetails
    skipped_by_ext: Dict[str, int]


class FileFingerprint(TypedDict):
    """File identity for stale bundle detection."""
    sha1: str
    mtime_iso: str
    size_bytes: int


# ============================================================================
# SECURITY KERNEL
# ============================================================================
class SecurityKernel:
    """Enhanced security with fingerprinting and syntax tracking."""
    
    SENSITIVE_PATTERNS = [
        re.compile(r"-----BEGIN[A-Z0-9 ]+KEY-----.*?-----END[A-Z0-9 ]+KEY-----", re.DOTALL),
        re.compile(r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----", re.DOTALL),
        re.compile(r"(api_key|secret_key|auth_token)\s*[:=]\s*['\"][A-Za-z0-9+/=]{20,}['\"]", re.IGNORECASE),
        re.compile(r"(password|passwd|pwd)\s*[:=]\s*['\"][^'\"]{8,}['\"]", re.IGNORECASE),
    ]

    # Enterprise High-Fidelity Vendor Signatures (TruffleHog / GitLeaks style)
    VENDOR_SIGNATURES = [
        re.compile(r"(?:^|[^A-Z0-9])(AKIA[A-Z0-9]{16})(?:[^A-Z0-9]|$)"),          # AWS Access Key
        re.compile(r"(ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{36}"),                    # GitHub Personal Access Token
        re.compile(r"sk_(test|live)_[0-9a-zA-Z]{24,34}"),                        # Stripe Standard Secret
        re.compile(r"xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24,34}"),  # Slack Token
        re.compile(r"AIza[0-9A-Za-z-_]{35}"),                                    # Google API Key
        re.compile(r"ya29\.[0-9A-Za-z\-_]+"),                                    # Google OAuth
        re.compile(r"sq0csp-[0-9A-Za-z\-_]{43}"),                                # Square Access Token
        re.compile(r"key-[0-9a-zA-Z]{32}"),                                      # Mailgun API Key
        re.compile(r"SG\.[a-zA-Z0-9_-]{22}\.[a-zA-Z0-9_-]{43}"),                 # SendGrid API Key
    ]

    @staticmethod
    def compute_file_fingerprint(filepath: pathlib.Path) -> FileFingerprint:
        try:
            sha1_hash = hashlib.sha1()
            with open(filepath, 'rb') as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    sha1_hash.update(chunk)
            
            sha1_str = sha1_hash.hexdigest()[:8]
            mtime = os.path.getmtime(filepath)
            mtime_iso = datetime.datetime.fromtimestamp(mtime).isoformat()
            size_bytes = filepath.stat().st_size
            
            return {
                "sha1": sha1_str,
                "mtime_iso": mtime_iso,
                "size_bytes": size_bytes,
            }
        except Exception as e:
            logger.warning(f"Cannot compute fingerprint for {filepath}: {e}")
            return {"sha1": "unknown", "mtime_iso": "unknown", "size_bytes": 0}

    @staticmethod
    def is_binary(filepath: str, scan_bytes: int = 2048) -> bool:
        # PDFs are binary but NOT ignored — they are extracted via PyMuPDF
        if any(filepath.lower().endswith(ext) for ext in PDF_EXTENSIONS):
            return False  # Allow PDF files through; text extraction handled downstream
        if any(filepath.lower().endswith(ext) for ext in IGNORE_EXTENSIONS):
            return True
        try:
            with open(filepath, 'rb') as handle:
                chunk = handle.read(scan_bytes)
            if not chunk:
                return False
            if b'\x00' in chunk:
                return True
            guess, _ = mimetypes.guess_type(filepath)
            if guess and not guess.startswith(('text', 'application')):
                return True
            text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x7F)))
            nontext = sum(byte not in text_chars for byte in chunk)
            return (nontext / len(chunk)) > 0.40
        except Exception:
            return True

    @staticmethod
    def calculate_entropy(text: str) -> float:
        if not text: 
            return 0.0
        from collections import Counter
        
        # Step 1.1: Standardize to byte-space
        byte_data = text.encode('utf-8')
        counts = Counter(byte_data)
        
        # Performance tweak: len(byte_data) is faster than sum(counts.values())
        total = len(byte_data) 
        return -sum((c/total) * math.log2(c/total) for c in counts.values())

    @classmethod
    def sanitize_content(cls, content: str) -> str:
        # Redact known sensitive patterns (e.g., PEM, certs, obvious secrets)
        for pattern in cls.SENSITIVE_PATTERNS:
            try:
                content = pattern.sub("[REDACTED_SENSITIVE_PATTERN]", content)
            except Exception as e:
                logger.debug(f"Regex pattern error during sanitization: {e}")

        # Enterprise TruffleHog/GitLeaks-style Vendor Signature Redaction
        for pattern in cls.VENDOR_SIGNATURES:
            try:
                content = pattern.sub("[REDACTED_VENDOR_SECRET]", content)
            except Exception as e:
                logger.debug(f"Regex pattern error during vendor sanitization: {e}")

        sensitive_keywords = [
            'api_key', 'secret_key', 'auth_token', 'access_token', 'password', 
            'bearer', 'client_secret', 'private_key', 'token', 'pwd'
        ]
        sanitized_lines: List[str] = []
        
        # Step 1.2: Refactored regex to isolate the value
        # Group 1: Optional line number + Variable name/whitespace, Group 2: Assignment op, Group 3: Quote, Group 4: Value, Group 5: Remainder
        assignment_regex = re.compile(r"^(\s*(?:\d+\s*\|\s*)?[\w_]+)(\s*[:=]\s*)([\'\"])(.*?)\3(.*)$")

        for line in content.splitlines():
            lower_line = line.lower()
            if any(k in lower_line for k in sensitive_keywords):
                match = assignment_regex.match(line)
                if match:
                    var_part = match.group(1)
                    assign_op = match.group(2)
                    quote = match.group(3)
                    value = match.group(4)
                    remainder = match.group(5)
                    
                    # Only run entropy check on the captured value
                    entropy = cls.calculate_entropy(value)
                    if entropy > 4.8:
                        # Safely reconstruct the line protecting the syntax
                        sanitized_lines.append(f"{var_part}{assign_op}{quote}[REDACTED_HIGH_ENTROPY]{quote}{remainder}")
                        continue
            sanitized_lines.append(line)
        return "\n".join(sanitized_lines)


class SASTScanner:
    """Wrapper for Semgrep to perform enterprise-grade Static Analysis & Taint Flow."""
    
    @staticmethod
    def scan_file(filepath: pathlib.Path) -> List[Dict[str, Any]]:
        """Executes Semgrep on a single file and extracts vulnerability telemetry."""
        try:
            result = subprocess.run(
                ['semgrep', 'scan', '--config=auto', '--json', '--quiet', str(filepath)],
                capture_output=True, text=True, check=False
            )
            if result.returncode in (0, 1) and result.stdout:
                data = json.loads(result.stdout)
                return data.get("results", [])
        except FileNotFoundError:
            logger.debug("Semgrep is not installed or not in PATH. Skipping SAST.")
        except Exception as e:
            logger.debug(f"Semgrep execution failed: {e}")
        return []


class ASTCache:
    """Filesystem cache for Polyglot AST analysis results."""

    def __init__(self, cache_dir: pathlib.Path, enabled: bool = True) -> None:
        self.cache_dir = cache_dir
        self.enabled = enabled
        if self.enabled:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Failed to initialize AST cache directory {cache_dir}: {e}")
                self.enabled = False

    def _cache_key(self, file_path: pathlib.Path, content: str) -> str:
        material = f"{AST_CACHE_VERSION}|{file_path.as_posix()}|{content}"
        return hashlib.sha256(material.encode("utf-8", errors="ignore")).hexdigest()

    def _cache_path(self, key: str) -> pathlib.Path:
        return self.cache_dir / f"{key}.json"

    def get(self, file_path: pathlib.Path, content: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            key = self._cache_key(file_path, content)
            path = self._cache_path(key)
            if not path.exists():
                return None
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError:
            if 'path' in locals() and path is not None:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
            return None
        except Exception:
            return None

    def set(self, file_path: pathlib.Path, content: str, analysis: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        tmp_path: Optional[pathlib.Path] = None
        try:
            key = self._cache_key(file_path, content)
            path = self._cache_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(analysis, handle, ensure_ascii=False)
            try:
                os.replace(tmp_path, path)
            except FileExistsError:
                # Thread Safety - Another thread already cached this, that's OK
                pass
            except PermissionError:
                logger.warning(f"Cannot write cache to {path}: permission denied")
        except Exception:
            pass
        finally:
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception: pass


# ============================================================================
# LEGACY PYTHON AST (Fallback)
# ============================================================================
class CallVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: List[str] = []
    def visit_Call(self, node: ast.Call) -> None:
        try:
            if isinstance(node.func, ast.Name): self.calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute): self.calls.append(node.func.attr)
        except Exception: pass
        self.generic_visit(node)

class ImportVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: List[str] = []
    def visit_Import(self, node: ast.Import) -> None:
        try:
            for alias in node.names:
                if alias.name: self.imports.append(alias.name)
        except Exception: pass
        self.generic_visit(node)
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        try:
            base = node.module or ""
            for alias in node.names:
                if base: self.imports.append(f"{base}.{alias.name}")
                else: self.imports.append(alias.name)
        except Exception: pass
        self.generic_visit(node)

class ASTSliceExtractor(ast.NodeVisitor):
    def __init__(self, source_code: str, filename: str) -> None:
        self.source_lines = source_code.splitlines()
        self.filename = filename.replace("\\", "/")
        self.slices: List[Dict[str, Any]] = []
        self.call_graph: List[str] = []
        self.syntax_valid = True

    def _canonical_slice_id(self, node_name: str, start_line_no: int, end_line_no: int) -> str:
        return f"{self.filename}::{node_name}@{start_line_no}-{end_line_no}"

    def _extract_code(self, node: ast.AST) -> str:
        try:
            start_line_no = int(getattr(node, "lineno", 1))
            end_line_no = int(getattr(node, "end_lineno", start_line_no))
            start = max(0, start_line_no - 1)
            end = min(len(self.source_lines), end_line_no)
            extracted = [f"{start + i + 1:4d} | {line}" for i, line in enumerate(self.source_lines[start:end])]
            return "\n".join(extracted)
        except Exception:
            return "[ERROR: Could not extract code]"

    def _process_slice(self, node: ast.AST, slice_type: str) -> None:
        try:
            cv = CallVisitor()
            cv.visit(node)
            node_name = getattr(node, 'name', '<unknown>')
            start_line_no = int(getattr(node, "lineno", 0))
            end_line_no = int(getattr(node, "end_lineno", start_line_no))
            calls = sorted(set(cv.calls))
            code_snippet = self._extract_code(node)
            
            # Extract plain code without line numbers to prevent Regex extraction bugs
            plain_code = "\n".join(line.split(" | ", 1)[1] if " | " in line else line for line in code_snippet.split("\n"))
            literal_vars = list(set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=', plain_code)))
            imports = list(set(
                re.findall(r'^import\s+([a-zA-Z0-9_.]+)', plain_code, re.MULTILINE) +
                re.findall(r'^from\s+([a-zA-Z0-9_.]+)\s+import', plain_code, re.MULTILINE)
            ))
            
            self.slices.append({
                "slice_id": self._canonical_slice_id(str(node_name), start_line_no, end_line_no),
                "type": slice_type,
                "name": node_name,
                "file": self.filename,
                "start_line": start_line_no,
                "end_line": end_line_no,
                "code_snippet": code_snippet,
                "literal_variables": literal_vars,
                "imports": imports,
                "calls": calls,
                "signature": {"args": [], "returns": "unknown", "calls": calls, "side_effects": ["none"]},
                "mutations": literal_vars,
                "operator_type": "general operator",
                "gpu_profile": {"device": "CPU/Unknown", "backend": "generic"},
            })
        except Exception:
            pass
        finally:
            self.generic_visit(node)
        
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None: self._process_slice(node, "async_function")
    def visit_ClassDef(self, node: ast.ClassDef) -> None: self._process_slice(node, "class")
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None: self._process_slice(node, "function")
    def visit_Call(self, node: ast.Call) -> None:
        try:
            if isinstance(node.func, ast.Name): self.call_graph.append(node.func.id)
            elif isinstance(node.func, ast.Attribute): self.call_graph.append(node.func.attr)
        except Exception: pass
        self.generic_visit(node)


# ============================================================================
# TREE-SITTER EXTRACTOR (Phase 5 / v6.0 Upgrade)
# ============================================================================
class PolyglotASTExtractor:
    """Universal parser utilizing Tree-sitter for multiple languages."""
    
    def __init__(self, file_path: str, ext: str, source_code: str):
        self.file_path = file_path.replace("\\", "/")
        self.ext = ext.lower()
        self.source_code = source_code
        self.source_bytes = source_code.encode('utf-8', errors='ignore')
        self.source_lines = source_code.splitlines()
        
        self.slices = []
        self.call_graph = []
        self.syntax_valid = True
        self.is_entry_point = False
        
        self.langs = {}
        if HAS_TREESITTER:
            if TS_PYTHON: self.langs['.py'] = tree_sitter_python.language()
            if TS_CPP:
                cpp_lang = tree_sitter_cpp.language()
                for e in ['.cpp', '.h', '.hpp', '.c', '.cc']: self.langs[e] = cpp_lang
            if TS_RUST: self.langs['.rs'] = tree_sitter_rust.language()
            if TS_JAVA: self.langs['.java'] = tree_sitter_java.language()
            if TS_RUBY: self.langs['.rb'] = tree_sitter_ruby.language()
            if TS_CSHARP: self.langs['.cs'] = tree_sitter_c_sharp.language()
            
            for cuda_ext in ['.cu', '.cuh']:
                if not self.langs.get(cuda_ext) and TS_CPP:
                    self.langs[cuda_ext] = cpp_lang

        self.parser = Parser() if HAS_TREESITTER else None

    def _extract_treesitter_node(self, node, file_path, source_bytes):
        slice_id = f"{file_path}::{node.type}@{node.start_point[0]}-{node.end_point[0]}"
        code_snippet = source_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='replace')
        literal_vars = list(set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:=|:=)', code_snippet)))
        imports = list(set(
            re.findall(r'^import\s+([a-zA-Z0-9_.]+)', code_snippet, re.MULTILINE) +
            re.findall(r'^from\s+([a-zA-Z0-9_.]+)\s+import', code_snippet, re.MULTILINE) +
            re.findall(r'#include\s*[<"]([^>"]+)[>"]', code_snippet, re.MULTILINE) +
            re.findall(r'use\s+([a-zA-Z0-9_:]+)', code_snippet, re.MULTILINE)
        ))

        return {
            "slice_id": slice_id,
            "type": node.type,
            "name": self._get_node_name(node),
            "file": file_path,
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
            "code_snippet": code_snippet,
            "literal_variables": literal_vars,
            "imports": imports,
            "calls": self._extract_calls(node),
            "mutations": literal_vars,
            "called_by": []
        }

    def _canonical_slice_id(self, node_name: str, start_line_no: int, end_line_no: int) -> str:
        return f"{self.file_path}::{node_name}@{start_line_no}-{end_line_no}"

    def extract_nodes(self) -> bool:
        if not HAS_TREESITTER or self.ext not in self.langs: return False
        try:
            self.parser.set_language(self.langs[self.ext])
            tree = self.parser.parse(self.source_bytes)
            if tree.root_node.has_error: self.syntax_valid = False
            self._traverse_and_map_nodes(tree.root_node)
            self._detect_entry_point(tree.root_node)
            return True
        except Exception as e:
            logger.debug(f"Tree-sitter extraction failed for {self.file_path}: {e}")
            return False

    def _get_node_name(self, node) -> str:
        try:
            if self.ext == '.py':
                name_node = node.child_by_field_name('name')
                if name_node: return self.source_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')
            elif self.ext in ['.cpp', '.h', '.hpp', '.c', '.cc']:
                declarator = node.child_by_field_name('declarator')
                if declarator:
                    if declarator.type == 'function_declarator':
                        name_node = declarator.child_by_field_name('declarator')
                        if name_node: return self.source_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')
                    return self.source_bytes[declarator.start_byte:declarator.end_byte].decode('utf-8')
                name_node = node.child_by_field_name('name')
                if name_node: return self.source_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')
            elif self.ext in ['.rs', '.java', '.cs', '.rb']:
                name_node = node.child_by_field_name('name')
                if name_node: return self.source_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')
        except Exception: pass
        return "<unknown>"

    def _extract_calls(self, node) -> List[str]:
        calls = []
        def walk(n):
            if n.type in ['call', 'call_expression', 'method_invocation']:
                func_node = n.child_by_field_name('function') or n.child_by_field_name('name') or n.child_by_field_name('method')
                if func_node:
                    try:
                        func_name = self.source_bytes[func_node.start_byte:func_node.end_byte].decode('utf-8')
                        if '.' in func_name: func_name = func_name.split('.')[-1]
                        if '::' in func_name: func_name = func_name.split('::')[-1]
                        calls.append(func_name)
                    except Exception: pass
            for c in n.children: walk(c)
        walk(node)
        return calls

    def _extract_mutations(self, node) -> List[str]:
        mutations = []
        def walk(n):
            if n.type in ['assignment', 'assignment_expression', 'augmented_assignment']:
                left_node = n.child_by_field_name('left')
                if left_node:
                    try:
                        mutations.append(self.source_bytes[left_node.start_byte:left_node.end_byte].decode('utf-8'))
                    except Exception: pass
            for c in n.children: walk(c)
        walk(node)
        return mutations

    def _extract_code(self, start_line_no: int, end_line_no: int) -> str:
        start = max(0, start_line_no - 1)
        end = min(len(self.source_lines), end_line_no)
        extracted = []
        for i, line in enumerate(self.source_lines[start:end]):
            line_num = start + i + 1
            extracted.append(f"{line_num:4d} | {line}")
        return "\n".join(extracted)

    def _traverse_and_map_nodes(self, node):
        target_types = {
            '.py': {'function_definition': 'function', 'class_definition': 'class', 'async_function_definition': 'async_function'},
            '.cpp': {'function_definition': 'function', 'class_specifier': 'class', 'struct_specifier': 'struct'},
            '.h': {'function_definition': 'function', 'class_specifier': 'class', 'struct_specifier': 'struct'},
            '.cc': {'function_definition': 'function', 'class_specifier': 'class', 'struct_specifier': 'struct'},
            '.rs': {'function_item': 'function', 'struct_item': 'struct', 'impl_item': 'class'},
            '.java': {'method_declaration': 'function', 'class_declaration': 'class'},
            '.rb': {'method': 'function', 'singleton_method': 'function', 'class': 'class'},
            '.cs': {'method_declaration': 'function', 'class_declaration': 'class'}
        }
        ext_targets = target_types.get(self.ext, {})
        if node.type in ext_targets:
            slice_obj = self._extract_treesitter_node(node, self.file_path, self.source_bytes)
            self.slices.append(slice_obj)
            self.call_graph.extend(slice_obj.get("calls", []))
        for child in node.children:
            self._traverse_and_map_nodes(child)

    def _detect_entry_point(self, root_node):
        if self.ext == '.py':
            self.is_entry_point = b'if __name__ == "__main__":' in self.source_bytes or b"if __name__ == '__main__':" in self.source_bytes
        else:
            for child in root_node.children:
                if child.type in ['function_definition', 'function_item', 'method_declaration', 'method']:
                    name = self._get_node_name(child)
                    if name in ['main', 'Main']:
                        self.is_entry_point = True
                        break


# ============================================================================
# POLYGLOT ANALYZER (Routing)
# ============================================================================
class PolyglotAnalyzer:
    """Analyze files routing to Tree-sitter (primary), SQLGlot, or AST/Regex (fallback)."""
    
    def __init__(self, event_callback: Optional[Callable[..., None]] = None) -> None:
        self.callback = event_callback

    def analyze_sql_ast(self, content: str, filename: str) -> Dict[str, Any]:
        """Enterprise SQL Semantic Parsing & Lineage Canonicalization."""
        if not HAS_SQLGLOT:
            return {"summary": {"syntax_valid": True, "error": "SQLGlot not installed"}, "complexity": 0}
        try:
            # 1. Parse and mathematically canonicalize the SQL AST
            ast_tree = sqlglot.parse_one(content, read=None)
            optimized_ast = sql_optimize(ast_tree, schema=None)
            optimized_sql = optimized_ast.sql(pretty=True)

            # 2. Extract Data Provenance (Tables) and State Mutation (Columns)
            tables = [table.name for table in optimized_ast.find_all(sqlglot.exp.Table)]
            columns = [col.name for col in optimized_ast.find_all(sqlglot.exp.Column)]

            slice_id = f"{filename}::sql_query@1-1"
            slice_meta = {
                "slice_id": slice_id,
                "type": "sql_query",
                "name": "optimized_data_lineage",
                "file": filename,
                "start_line": 1,
                "end_line": len(optimized_sql.splitlines()),
                "code_snippet": optimized_sql, # Storing optimized/canonicalized intent
                "literal_variables": list(set(columns)),
                "calls": list(set(tables)),    # Tables act as downstream graph edges
                "imports": [],
            }
            return {
                "summary": {
                    "slices": [slice_meta],
                    "call_graph": list(set(tables)),
                    "is_entry_point": False,
                    "syntax_valid": True,
                },
                "complexity": len(tables) + len(columns)
            }
        except Exception as e:
            return {"summary": {"syntax_valid": False, "error": str(e)}, "complexity": 0}

    def analyze_with_treesitter(self, content: str, filename: str, ext: str) -> Optional[Dict[str, Any]]:
        extractor = PolyglotASTExtractor(filename, ext, content)
        if extractor.extract_nodes():
            return {
                "summary": {
                    "slices": extractor.slices,
                    "call_graph": sorted(set(extractor.call_graph)),
                    "import_graph": [], 
                    "is_entry_point": extractor.is_entry_point,
                    "syntax_valid": extractor.syntax_valid,
                },
                "complexity": len(extractor.slices)
            }
        return None

    def analyze_python_fallback(self, content: str, filename: str) -> Dict[str, Any]:
        """Legacy AST fallback if Tree-sitter is unavailable."""
        try:
            tree = ast.parse(content)
            extractor = ASTSliceExtractor(content, filename)
            extractor.visit(tree)

            import_visitor = ImportVisitor()
            import_visitor.visit(tree)
            imports = sorted(set(import_visitor.imports))

            is_entry_point = False
            for node in ast.walk(tree):
                if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
                    if isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__":
                        is_entry_point = True

            return {
                "summary": {
                    "slices": extractor.slices,
                    "call_graph": sorted(set(extractor.call_graph)),
                    "import_graph": imports,
                    "is_entry_point": is_entry_point,
                    "syntax_valid": True,
                },
                "complexity": len(extractor.slices)
            }
        except SyntaxError as e:
            return {"summary": {"error": f"Syntax Error: {e.msg}", "syntax_valid": False}, "complexity": 999}
        except Exception:
            return {"summary": {"syntax_valid": True}, "complexity": 0}

    def analyze_json(self, content: str) -> Dict[str, Any]:
        try:
            data = json.loads(content)
            keys = list(data.keys()) if isinstance(data, dict) else ["<array>"]
            return {"summary": {"keys": keys[:10], "syntax_valid": True}, "complexity": 0}
        except json.JSONDecodeError as e:
            return {"summary": {"error": "Invalid JSON", "syntax_valid": False}, "complexity": 0}

    def analyze_yaml(self, content: str) -> Dict[str, Any]:
        keys = []
        try:
            for line in content.splitlines()[:100]:
                match = re.match(r'^([a-zA-Z0-9_-]+):', line)
                if match: keys.append(match.group(1))
            return {"summary": {"keys": keys[:15], "syntax_valid": True}, "complexity": 0}
        except Exception:
            return {"summary": {"syntax_valid": True}, "complexity": 0}

    def analyze(self, filename: str, content: str) -> Dict[str, Any]:
        try:
            ext = pathlib.Path(filename).suffix.lower()
            if ext == '.sql':
                return self.analyze_sql_ast(content, filename)
            if ext in ['.py', '.cpp', '.h', '.hpp', '.c', '.cc', '.rs', '.java', '.rb', '.cs']:
                ts_result = self.analyze_with_treesitter(content, filename, ext)
                if ts_result:
                    return ts_result
                elif ext == '.py':
                    return self.analyze_python_fallback(content, filename)
                    
            if ext == '.json':
                return self.analyze_json(content)
            if ext in {'.yaml', '.yml'}:
                return self.analyze_yaml(content)
                
            return {"summary": {"syntax_valid": True}, "complexity": 0}
        except Exception as e:
            logger.warning(f"Error analyzing {filename}: {e}")
            return {"summary": {"syntax_valid": True}, "complexity": 0}


# ============================================================================
# WORKER PROCESS
# ============================================================================
def _process_file_worker(
    path_obj: pathlib.Path, 
    base_path: pathlib.Path, 
    cache_dir: pathlib.Path, 
    ast_cache_enabled: bool, 
    enable_sast: bool = False,
    semantic_diff: bool = False
) -> Dict[str, Any]:
    """Independent worker function for ProcessPoolExecutor & Ray to avoid GIL overhead."""
    try:
        if not path_obj.exists() or not path_obj.is_file():
            return {"status": "error", "reason": "missing_or_not_file"}

        if SecurityKernel.is_binary(str(path_obj)):
            return {"status": "skipped", "skip_reason": "binary_or_ext", "ext": path_obj.suffix}

        size = path_obj.stat().st_size
        if size > MAX_FILE_SIZE_BYTES:
            return {"status": "skipped", "skip_reason": "oversize", "ext": path_obj.suffix}

        # --- PDF Interceptor: extract text via PyMuPDF instead of utf-8 open ---
        is_pdf = path_obj.suffix.lower() in PDF_EXTENSIONS
        if is_pdf:
            if not HAS_PYMUPDF:
                return {"status": "skipped", "skip_reason": "pdf_no_pymupdf", "ext": ".pdf"}
            try:
                doc = fitz.open(str(path_obj))
                pages = []
                for page_num, page in enumerate(doc):
                    page_text = page.get_text("text")
                    # Fallback to OCR for scanned/image-based pages
                    if not page_text.strip() and HAS_TESSERACT:
                        mat = fitz.Matrix(300/72, 300/72)
                        pix = page.get_pixmap(matrix=mat)
                        img = Image.open(_io.BytesIO(pix.tobytes('png')))
                        page_text = pytesseract.image_to_string(img)
                    if page_text.strip():
                        pages.append(f"[PAGE {page_num + 1}]\n{page_text}")
                doc.close()
                raw = "\n\n".join(pages)
                if not raw.strip():
                    return {"status": "skipped", "skip_reason": "pdf_empty_extraction", "ext": ".pdf"}
                logger.info(f"PDF extracted: {path_obj.name} — {len(pages)} pages, {len(raw)} chars")
            except Exception as e:
                logger.warning(f"PDF extraction failed for {path_obj}: {e}")
                return {"status": "error", "reason": f"pdf_extraction_error: {e}"}
        else:
            try:
                with open(path_obj, "r", encoding="utf-8", errors="ignore") as handle:
                    raw = handle.read()
            except OSError:
                return {"status": "error", "reason": "read_error"}

        try:
            rel_path = path_obj.relative_to(base_path).as_posix()
        except ValueError:
            return {"status": "skipped", "skip_reason": "outside_base", "ext": path_obj.suffix}

        analyzer = PolyglotAnalyzer()
        ast_cache = ASTCache(cache_dir=cache_dir, enabled=ast_cache_enabled)
        
        ext = path_obj.suffix.lower()
        if ext not in ['.py', '.cpp', '.h', '.hpp', '.c', '.cc', '.rs', '.java', '.rb', '.cs', '.sql']:
            analysis = analyzer.analyze(rel_path, raw)
        else:
            cached = ast_cache.get(path_obj, raw)
            if cached is not None:
                analysis = cached
            else:
                analysis = analyzer.analyze(rel_path, raw)
                ast_cache.set(path_obj, raw, analysis)

        fingerprint = SecurityKernel.compute_file_fingerprint(path_obj)
        summary = analysis.get("summary") or {}

        syntax_error = None
        if not summary.get("syntax_valid", True):
            error_msg = summary.get("error", "Unknown syntax error")
            error_line = summary.get("error_line", "?")
            syntax_error = f"{rel_path} (Line {error_line}: {error_msg})"

        sast_findings = []
        if enable_sast:
            sast_findings = SASTScanner.scan_file(path_obj)

        # Mathematical AST diffing for logical evolutions vs superficial diffing
        ast_semantic_diffs = []
        if semantic_diff and ext == '.sql' and HAS_SQLGLOT:
            try:
                # Capture previous HEAD logic from git history
                old_content = subprocess.check_output(
                    ['git', 'show', f'HEAD:{rel_path}'], 
                    cwd=base_path, text=True, stderr=subprocess.DEVNULL
                )
                old_ast = sqlglot.parse_one(old_content, read=None)
                new_ast = sqlglot.parse_one(raw, read=None)
                diff_results = sqlglot.diff(old_ast, new_ast)
                ast_semantic_diffs = [str(d) for d in diff_results]
            except Exception:
                pass

        return {
            "status": "bundled",
            "entry": {
                "path": rel_path,
                "size_bytes": size,
                "content": raw,
                "summary": summary,
                "complexity": analysis.get("complexity", 0),
                "fingerprint": fingerprint,
                "sast_findings": sast_findings,
                "semantic_diffs": ast_semantic_diffs
            },
            "syntax_error": syntax_error,
        }

    except Exception as e:
        return {"status": "error", "reason": "unexpected"}


# ============================================================================
# CORE PACKAGER
# ============================================================================
class WorkspacePackager:

    def _compute_bundle_hash(self, text: str) -> str:
        """Compute deterministic SHA256 hash of bundle content."""
        if ENABLE_DETERMINISTIC_HASH:
            return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        else:
            timestamp = datetime.datetime.now().isoformat()
            material = f"{text}|{timestamp}"
            return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    def __init__(
        self,
        target_files: List[pathlib.Path],
        base_path: pathlib.Path,
        project_name: str = "custom_bundle",
        focus_target: Optional[str] = None,
        depth: int = 0,
        append_rules: bool = False,
        deterministic: bool = False,
        agent_role: Optional[str] = None,
        agent_task: Optional[str] = None,
        agent_target: Optional[str] = None,
        workers: int = MAX_WORKERS_DEFAULT,
        ast_cache_enabled: bool = True,
        enable_sast: bool = False,
        semantic_diff: bool = False,
        use_ray: bool = False,
        event_callback: Optional[Callable[..., None]] = None
    ) -> None:
        self.target_files = target_files
        self.base_path = base_path.resolve()
        self.project_name = project_name
        self.focus_target = focus_target
        self.depth = depth
        self.append_rules = append_rules
        self.deterministic = deterministic
        self.agent_role = agent_role
        self.agent_task = agent_task
        self.agent_target = agent_target
        self.workers = max(1, min(workers, MAX_WORKERS_LIMIT))
        self.ast_cache_enabled = ast_cache_enabled
        self.enable_sast = enable_sast
        self.semantic_diff = semantic_diff
        self.use_ray = use_ray
        self.event_callback = event_callback
        
        self.files_registry: List[Dict[str, Any]] = []
        self.syntax_errors: List[str] = []
        self.bundle_hash: Optional[str] = None
        
        self.scan_stats: ScanStats = {
            "bundled": 0, "skipped": 0, "errors": 0,
            "skipped_details": {"binary_or_ext": 0, "oversize": 0, "outside_base": 0},
            "skipped_by_ext": {},
        }
        
        if deterministic:
            self.timestamp = "DETERMINISTIC_BUILD"
        else:
            self.timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        cache_dir = self.base_path / ".cache" / "aletheia_ast"
        self.ast_cache = ASTCache(cache_dir=cache_dir, enabled=self.ast_cache_enabled)

    def _build_slice_maps(self) -> Dict[str, Any]:
        slice_by_id, function_name_to_slice_ids = {}, {}
        for file_meta in self.files_registry:
            summary = file_meta.get("summary") or {}
            for slice_meta in summary.get("slices", []):
                slice_id = slice_meta.get("slice_id")
                if not slice_id: continue
                slice_by_id[slice_id] = {
                    **slice_meta,
                    "file_path": file_meta.get("path", "unknown"),
                    "is_entry_point_file": summary.get("is_entry_point", False),
                }
                name = str(slice_meta.get("name", "")).lower()
                if name: function_name_to_slice_ids.setdefault(name, []).append(slice_id)
        return {"slice_by_id": slice_by_id, "function_name_to_slice_ids": function_name_to_slice_ids}

    def _build_nx_graph(self) -> nx.DiGraph:
        """Constructs a heavily optimized NetworkX Directed Graph of the AST."""
        G = nx.DiGraph()
        maps = self._build_slice_maps()
        slice_by_id = maps["slice_by_id"]
        name_to_slice_ids = maps["function_name_to_slice_ids"]

        # 1. Populate nodes with metadata for fast metric extraction
        for slice_id, meta in slice_by_id.items():
            G.add_node(
                slice_id,
                file_path=meta.get("file_path"),
                is_entry=meta.get("is_entry_point_file", False),
                name=meta.get("name")
            )

        # 2. Build dependency edges (Caller -> Callee)
        for slice_id, meta in slice_by_id.items():
            for called_name in meta.get("calls", []):
                called_lower = str(called_name).lower()
                for target_slice_id in name_to_slice_ids.get(called_lower, []):
                    if target_slice_id != slice_id:
                        G.add_edge(slice_id, target_slice_id)
        return G

    def _prune_topology_nodes(self, focal_slice_id: str, node_ids: List[str], pagerank_scores: Dict[str, float]) -> List[str]:
        unique_ids = sorted(set(node_ids))
        ranked_ids = sorted(
            unique_ids,
            key=lambda sid: (
                sid != focal_slice_id,
                -pagerank_scores.get(sid, 0.0), # Sort primarily by absolute PageRank centrality
                sid,
            ),
        )
        kept_ids = ranked_ids[:MAX_CLUSTER_NODES]
        if focal_slice_id and focal_slice_id not in kept_ids:
            kept_ids = [focal_slice_id] + kept_ids[: MAX_CLUSTER_NODES - 1]
        return sorted(set(kept_ids))

    def _build_topology_cluster(self, focal_slice_id: str, G: nx.DiGraph, pagerank_scores: Dict[str, float]) -> Dict[str, Any]:
        if not focal_slice_id or focal_slice_id not in G:
            return {"radius": 1, "node_ids": [], "edges": [], "parent_ids": [], "child_ids": []}

        parent_ids = sorted(list(G.predecessors(focal_slice_id)))
        child_ids = sorted(list(G.successors(focal_slice_id)))
        
        cluster_node_ids = [focal_slice_id] + parent_ids + child_ids
        cluster_node_ids = self._prune_topology_nodes(focal_slice_id, cluster_node_ids, pagerank_scores)
        allowed_ids = set(cluster_node_ids)

        # Utilize NetworkX native subgraph views to rapidly resolve topology links
        subgraph = G.subgraph(allowed_ids)
        cluster_edges = []
        for u, v in subgraph.edges():
            if u == focal_slice_id or v == focal_slice_id:
                cluster_edges.append({"source": u, "target": v})

        cluster_edges = sorted(
            {json.dumps(edge, sort_keys=True, separators=(",", ":"), ensure_ascii=True) for edge in cluster_edges}
        )[:MAX_CLUSTER_EDGES]

        return {
            "radius": 1,
            "node_ids": cluster_node_ids,
            "edges": [json.loads(edge) for edge in cluster_edges],
            "parent_ids": [node_id for node_id in parent_ids if node_id in allowed_ids],
            "child_ids": [node_id for node_id in child_ids if node_id in allowed_ids],
        }

    def _compute_slice_risk_scores(self, G: nx.DiGraph, pagerank_scores: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        risk_meta = {}

        for slice_id in G.nodes():
            callers_count = G.in_degree(slice_id)
            callees_count = G.out_degree(slice_id)
            is_entry_point_file = G.nodes[slice_id].get("is_entry", False)
            
            # Incorporate mathematical centrality directly into patch collision logic
            pr_score = pagerank_scores.get(slice_id, 0.0)
            pr_boost = int(pr_score * 500) 

            score = min(100, callers_count * 25 + callees_count * 15 + (20 if is_entry_point_file else 0) + pr_boost)
            level = "CRITICAL" if score >= 80 else "HIGH" if score >= 60 else "MEDIUM" if score >= 40 else "LOW"

            risk_meta[slice_id] = {
                "slice_id": slice_id, 
                "file_path": G.nodes[slice_id].get("file_path", "unknown"),
                "slice_name": G.nodes[slice_id].get("name", "unknown"),
                "callers": sorted(list(G.predecessors(slice_id))), 
                "callees": sorted(list(G.successors(slice_id))),
                "callers_count": callers_count, 
                "callees_count": callees_count,
                "pagerank_score": round(pr_score, 6),
                "entrypoint_context": is_entry_point_file, 
                "risk_score": score, 
                "risk_level": level,
            }
        return risk_meta

    def _agent_header_lines(self) -> List[str]:
        if not (self.agent_role or self.agent_task or self.agent_target): return []
        return [
            "# AGENT_CONTEXT",
            f"# ROLE: {self.agent_role or 'unspecified'}",
            f"# TASK: {self.agent_task or 'unspecified'}",
            f"# TARGET: {self.agent_target or self.focus_target or 'unspecified'}",
            "",
        ]

    def _build_execution_paths(self, G: nx.DiGraph) -> Dict[str, List[str]]:
        file_entry_slices = {}
        for file_meta in self.files_registry:
            summary = file_meta.get("summary") or {}
            if not summary.get("is_entry_point"): continue
            entry_candidates = [s.get("slice_id") for s in summary.get("slices", []) if str(s.get("name", "")).lower() in {"main", "run", "start", "cli", "entrypoint"}]
            if not entry_candidates:
                entry_candidates = [s.get("slice_id") for s in summary.get("slices", []) if s.get("type") in {"function", "async_function"}]
            file_entry_slices[file_meta.get("path", "unknown")] = [sid for sid in entry_candidates if sid]

        execution_paths = {}
        for file_path, roots in file_entry_slices.items():
            discovered = set()
            discovered_list = []
            for root in roots:
                if root not in G: continue
                
                # Replaces manual iterative queue loop with C-accelerated breadth-first tree traversal
                bfs_nodes = list(nx.bfs_tree(G, root, depth_limit=6).nodes())
                for node in bfs_nodes:
                    if node not in discovered:
                        discovered.add(node)
                        discovered_list.append(node)
            execution_paths[file_path] = discovered_list
        return execution_paths

    def _track_skip(self, reason: Literal["binary_or_ext", "oversize", "outside_base"], ext: str) -> None:
        self.scan_stats['skipped'] += 1
        self.scan_stats['skipped_details'][reason] += 1
        safe_ext = (ext or "").lower()
        self.scan_stats['skipped_by_ext'][safe_ext] = self.scan_stats['skipped_by_ext'].get(safe_ext, 0) + 1

    def _generate_tree_map(self) -> str:
        lines = [f"{self.project_name}/"]
        paths = sorted(pathlib.Path(f['path']) for f in self.files_registry)
        seen_dirs = set()
        for p in paths:
            try:
                parts, indent = p.parts, 0
                for i, part in enumerate(parts[:-1]):
                    d = pathlib.Path(*parts[:i + 1])
                    if str(d) not in seen_dirs:
                        lines.append(f"{'  ' * (indent + 1)}|-- {part}/")
                        seen_dirs.add(str(d))
                    indent += 1
                lines.append(f"{'  ' * (indent + 1)}|-- {parts[-1]}")
            except Exception: pass
        return "\n".join(lines)

    def _get_allowed_focus_slices(self) -> Set[str]:
        if not self.focus_target: return set()
        allowed_names = {self.focus_target.lower()}
        all_slices = []
        for f in self.files_registry: all_slices.extend((f.get('summary') or {}).get("slices", []))
        current_focus = set(allowed_names)
        for _ in range(self.depth):
            new_calls = set()
            for s in all_slices:
                if s.get('name', '').lower() in current_focus:
                    for c in s.get('calls', []): new_calls.add(c.lower())
            allowed_names.update(new_calls)
            current_focus = new_calls
        return allowed_names

    def _generate_json_output(self) -> str:
        try:
            G = self._build_nx_graph()
            try:
                # Compute PageRank for Centrality Scoring safely
                pagerank_scores = nx.pagerank(G, max_iter=100)
            except Exception:
                # Safe fallback if PageRank hits extremely rare cyclic convergence anomalies
                pagerank_scores = {n: G.in_degree(n) / max(1, len(G)) for n in G.nodes()}

            # Extract standard dict layouts for backwards JSON API compatibility
            slice_graph = nx.to_dict_of_lists(G)
            reverse_graph = nx.to_dict_of_lists(G.reverse())
            
            risk_meta = self._compute_slice_risk_scores(G, pagerank_scores)
            execution_paths = self._build_execution_paths(G)
            allowed_slices = self._get_allowed_focus_slices()
            
            layer_2_intelligence = []
            for f in self.files_registry:
                summary = f.get('summary')
                if not summary: continue
                raw_slices = summary.get("slices", [])
                if self.focus_target:
                    raw_slices = [s for s in raw_slices if str(s.get('name', '')).lower() in allowed_slices]
                enriched_slices = []
                for s in raw_slices:
                    slice_id = s.get("slice_id", "")
                    
                    # Ensure we grab whichever code key exists
                    raw_code = str(s.get("code_snippet", s.get("code", "")))
                    
                    enriched_slice = {
                        **s,  # Inherit standard slice keys
                        "code": SecurityKernel.sanitize_content(raw_code),
                        "called_by": reverse_graph.get(slice_id, s.get("called_by", [])),
                    }
                    
                    # Step 1.4: Guarantee topology_cluster is attached to every slice
                    enriched_slice["topology_cluster"] = self._build_topology_cluster(
                        slice_id,
                        G,
                        pagerank_scores,
                    )
                    
                    enriched_slices.append(enriched_slice)
                    
                layer_2_intelligence.append({
                    "path": f['path'],
                    "fingerprint": f.get('fingerprint', {}),
                    "call_graph": sorted(set(summary.get("call_graph", []))),
                    "is_entry_point": summary.get("is_entry_point", False),
                    "slices": enriched_slices,
                    "sast_findings": f.get("sast_findings", []),
                    "semantic_diffs": f.get("semantic_diffs", [])
                })

            payload = {
                "meta": {
                    "project": self.project_name,
                    "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
                    "generated_at": self.timestamp if not self.deterministic else None,
                    "deterministic": self.deterministic,
                    "stats": self.scan_stats,
                },
                "layer_2_intelligence": layer_2_intelligence,
                "layer_2_5_slice_dependency_graph": slice_graph,
                "layer_2_6_execution_flow": execution_paths,
                "layer_2_7_patch_collision_risk": risk_meta,
                "layer_3_full_files": [] if self.focus_target else [
                    {
                        "path": f['path'],
                        "size_bytes": f['size_bytes'],
                        "content": SecurityKernel.sanitize_content(f['content']),
                    }
                    for f in self.files_registry
                ],
            }
            interim = json.dumps(payload, indent=2, sort_keys=True, default=str)
            self.bundle_hash = self._compute_bundle_hash(interim)
            
            # Repack the JSON payload with the newly calculated correct deterministic bundle hash
            payload["meta"]["bundle_hash"] = self.bundle_hash
            return json.dumps(payload, indent=2, sort_keys=True, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)

    def _generate_text_output(self) -> str:
        lines = []
        allowed_slices = self._get_allowed_focus_slices()
        
        G = self._build_nx_graph()
        try:
            pagerank_scores = nx.pagerank(G, max_iter=100)
        except Exception:
            pagerank_scores = {n: G.in_degree(n) / max(1, len(G)) for n in G.nodes()}

        slice_graph = nx.to_dict_of_lists(G)
        reverse_graph = nx.to_dict_of_lists(G.reverse())
        risk_meta = self._compute_slice_risk_scores(G, pagerank_scores)
        execution_paths = self._build_execution_paths(G)

        lines.extend(["==================================================================", "LAYER 1: TOPOLOGY MAP", "=================================================================="])
        lines.extend(self._generate_tree_map().splitlines()); lines.append("")

        lines.extend(["==================================================================", "LAYER 2: CODE INTELLIGENCE (SLICES & CALL GRAPHS)", "==================================================================", ""])
        for f in self.files_registry:
            summary = f.get('summary') or {}
            slices = summary.get("slices", [])
            call_graph = summary.get("call_graph", [])

            if self.focus_target:
                slices = [s for s in slices if s.get('name', '').lower() in allowed_slices]
                if not slices: continue

            if slices or call_graph or f.get("sast_findings") or f.get("semantic_diffs"):
                lines.append(f"--- Intelligence: {f['path']} ---")
                if call_graph: lines.append(f"Global Dependencies: {', '.join(sorted(set(call_graph)))}")
                
                sast = f.get("sast_findings", [])
                if sast:
                    lines.append(f"SAST Findings ({len(sast)} Vulnerabilities Detected):")
                    for finding in sast:
                        rule_id = finding.get('check_id', 'unknown_rule')
                        lines.append(f"  [!] {rule_id}: {finding.get('extra', {}).get('message', 'No details')}")

                ast_diffs = f.get("semantic_diffs", [])
                if ast_diffs:
                    lines.append(f"AST Semantic Evolutions ({len(ast_diffs)} Mathematical Diffs):")
                    for d in ast_diffs: lines.append(f"  [>] {d}")

                for s in slices:
                    lines.extend([
                        f"\n[SLICE: {s['name']} | Type: {s['type']} | Lines {s['start_line']}-{s['end_line']}]",
                        f"SLICE_ID: {s.get('slice_id', 'unknown')}",
                        f"Target Calls: {', '.join(s.get('calls', []))}",
                        SecurityKernel.sanitize_content(s.get('code_snippet', s.get('code', '')))
                    ])
                lines.append("\n" + ("-" * 40))

        lines.extend(["", "==================================================================", "LAYER 2.5: SLICE DEPENDENCY GRAPH", "==================================================================", ""])
        for source_slice, targets in sorted(slice_graph.items()):
            if targets:
                lines.append(source_slice)
                for target in targets[:20]: lines.append(f"  ├── depends_on -> {target}")

        lines.extend(["", "==================================================================", "LAYER 2.6: EXECUTION FLOW", "==================================================================", ""])
        for source_file, path_nodes in sorted(execution_paths.items()):
            if path_nodes:
                lines.append(source_file)
                for idx, node in enumerate(path_nodes[:30]):
                    prefix = "  └──" if idx == len(path_nodes[:30]) - 1 else "  ├──"
                    lines.append(f"{prefix} {node}")

        if not self.focus_target:
            lines.extend(["", "==================================================================", "LAYER 3: FULL FILE CACHE", "=================================================================="])
            for f in self.files_registry:
                lines.extend([f"--- FULL FILE: {f['path']} ---", "Content: |"])
                lines.extend(f"  {line}" for line in SecurityKernel.sanitize_content(f['content']).splitlines())
                lines.append("")

        if self.append_rules:
            lines.extend([
                "", "==================================================================", "STRICT OPERATIONAL GUARDRAILS", "==================================================================",
                "1. DO NOT regenerate entire files. Output surgical replacements.",
                "2. All code modifications MUST be output in PATCH format.",
            ])

        final_text = "\n".join(lines)
        self.bundle_hash = self._compute_bundle_hash(final_text)
        
        # Exact BPE Tokenization using OpenAI's cl100k_base (tiktoken) to prevent LLM context overflow
        if HAS_TIKTOKEN:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
                estimated_tokens = len(enc.encode(final_text, allowed_special="all"))
            except Exception:
                estimated_tokens = len(final_text) // 4
        else:
            estimated_tokens = len(final_text) // 4
        
        header = [f"# Workspace Bundle: {self.project_name}"]
        if not self.deterministic:
            header.append(f"# Generated: {self.timestamp}")
            
        header.append(f"# Estimated Tokens (cl100k_base): ~{estimated_tokens:,} tokens")
        header.append(f"# Bundle Hash: {self.bundle_hash}")
        header.append(f"# Bundle Schema Version: {BUNDLE_SCHEMA_VERSION}")
        header.append("# Compliance: v6.0 - Distributed Scalable Architecture (Ray/Pydantic/Semgrep)")
        header.append("")
        header.extend(self._agent_header_lines())
        
        if self.focus_target:
            header.append(f"# FOCUS MODE: Target '{self.focus_target}' (Depth: {self.depth})\n")

        return "\n".join(header) + "\n" + final_text

    def run(self, format_type: str = "text") -> str:
        cache_dir = self.base_path / ".cache" / "aletheia_ast"

        # Distributed compute scale-out execution using RAY cluster manager
        if self.use_ray and HAS_RAY:
            logger.info(f"Scanning {len(self.target_files)} targeted files using RAY Distributed Cluster...")
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)

            @ray.remote
            def ray_worker(path_obj, base_path, cache_dir, ast_cache_enabled, enable_sast, semantic_diff):
                return _process_file_worker(path_obj, base_path, cache_dir, ast_cache_enabled, enable_sast, semantic_diff)

            futures = [
                ray_worker.remote(
                    path_obj, self.base_path, cache_dir, self.ast_cache_enabled, 
                    self.enable_sast, self.semantic_diff
                )
                for path_obj in self.target_files
            ]

            for idx, file_result in enumerate(ray.get(futures), start=1):
                if idx % 100 == 0:
                    logger.info(f"Processed {idx}/{len(self.target_files)} files (Ray)...")
                
                status = file_result.get("status")
                if status == "bundled":
                    self.files_registry.append(file_result["entry"])
                    self.scan_stats["bundled"] += 1
                    syntax_error = file_result.get("syntax_error")
                    if syntax_error and isinstance(syntax_error, str):
                        self.syntax_errors.append(syntax_error)
                elif status == "skipped":
                    skip_reason = file_result.get("skip_reason", "unknown")
                    self._track_skip(skip_reason, file_result.get("ext", ""))
                else:
                    self.scan_stats["errors"] += 1

        # Fallback to local multiprocess compute (Bypassing GIL)
        else:
            logger.info(f"Scanning {len(self.target_files)} targeted files with {self.workers} workers (ProcessPoolExecutor)...")
            if self.enable_sast:
                logger.info("Enterprise SAST (Semgrep) enabled. Tracing vulnerabilities...")

            future_to_path: Dict[Any, pathlib.Path] = {}
            with ProcessPoolExecutor(max_workers=self.workers) as executor:
                for path_obj in self.target_files:
                    future = executor.submit(
                        _process_file_worker,
                        path_obj,
                        self.base_path,
                        cache_dir,
                        self.ast_cache_enabled,
                        self.enable_sast,
                        self.semantic_diff
                    )
                    future_to_path[future] = path_obj

                for idx, future in enumerate(as_completed(future_to_path), start=1):
                    if idx % 100 == 0:
                        logger.info(f"Processed {idx}/{len(self.target_files)} files...")

                    path_obj = future_to_path[future]

                    try:
                        file_result = future.result()
                    except Exception as e:
                        logger.error(f"Error processing {path_obj}: {e}")
                        self.scan_stats["errors"] += 1
                        continue

                    status = file_result.get("status")
                    if status == "bundled":
                        self.files_registry.append(file_result["entry"])
                        self.scan_stats["bundled"] += 1
                        
                        syntax_error = file_result.get("syntax_error")
                        if syntax_error and isinstance(syntax_error, str):
                            self.syntax_errors.append(syntax_error)
                    elif status == "skipped":
                        skip_reason = file_result.get("skip_reason", "unknown")
                        self._track_skip(skip_reason, file_result.get("ext", ""))
                    else:
                        self.scan_stats["errors"] += 1

        self.files_registry.sort(key=lambda x: (x.get('complexity', 0), x.get('path', '')))
        return self._generate_json_output() if format_type == "json" else self._generate_text_output()


# ============================================================================
# MAIN & CLI
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aletheia Workspace Packager v6.0 (Universal Polyglot Engine)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="*", help="Specific files or directories to package")
    parser.add_argument("--manifest", help="Path to CSV or TXT file with file list")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("-o", "--output", help="Output filename")
    parser.add_argument("--base-dir", default=".", help="Base directory for relative paths")
    parser.add_argument("--focus", help="Only output slices for this exact function/class name")
    parser.add_argument("--depth", type=int, default=0, help="Dependency resolution depth")
    parser.add_argument("--append-rules", action="store_true", help="Add LLM modification rules")
    parser.add_argument("--git-diff", action="store_true", help="Scan Git-changed files")
    parser.add_argument("--semantic-diff", action="store_true", help="Calculate Mathematical AST Diffs instead of text diffs (SQL)")
    parser.add_argument("--deterministic", action="store_true", help="Omit timestamp for reproducibility")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS_DEFAULT, help=f"Parallel workers (1-{MAX_WORKERS_LIMIT})")
    parser.add_argument("--use-ray", action="store_true", help="Distribute massive parsing jobs across a Ray cluster")
    parser.add_argument("--no-ast-cache", action="store_true", help="Disable AST cache for Polyglot analysis")
    parser.add_argument("--enable-sast", action="store_true", help="Enable Enterprise SAST scanning via Semgrep")
    parser.add_argument("--agent-role", choices=["review", "patch", "architecture", "optimize", "validate"], help="Agent role")
    parser.add_argument("--agent-task", help="Short task label for the agent")
    parser.add_argument("--agent-target", help="Primary code target for the agent")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    logger.info("=== Semantic Slicer v6.0 (Polyglot Tree-sitter + AST Pipeline) ===")
    
    if not HAS_TREESITTER:
        logger.warning("Tree-sitter not detected. Falling back to native python 'ast' only.")
        logger.warning("To enable polyglot (C++, Rust), run: pip install tree-sitter tree-sitter-cpp tree-sitter-rust tree-sitter-python")

    target_files: List[pathlib.Path] = []
    base_path = pathlib.Path(args.base_dir).resolve()

    if not base_path.exists() or not base_path.is_dir():
        logger.error(f"Invalid base directory: {base_path}")
        sys.exit(1)

    def add_target_file(candidate: pathlib.Path) -> None:
        resolved = candidate.resolve()
        try: resolved.relative_to(base_path)
        except ValueError: return
        target_files.append(resolved)

    if args.paths:
        for p_str in args.paths:
            try:
                p = pathlib.Path(p_str).resolve()
                if p.is_file(): add_target_file(p)
                elif p.is_dir():
                    try: p.relative_to(base_path)
                    except ValueError: continue
                    for root, dirs, files in os.walk(p):
                        dirs[:] = sorted(d for d in dirs if d not in IGNORE_DIRS)
                        for file_name in sorted(files): add_target_file(pathlib.Path(root) / file_name)
            except Exception: pass

    if args.manifest:
        manifest_path = pathlib.Path(args.manifest)
        if manifest_path.exists():
            try:
                if manifest_path.suffix.lower() == '.csv':
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        for row in csv.DictReader(f):
                            if 'abs_path' in row and row['abs_path']: add_target_file(pathlib.Path(row['abs_path']))
                else:
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            clean = line.strip()
                            if clean and not clean.startswith('#'): add_target_file(pathlib.Path(clean))
            except Exception: pass

    if args.git_diff or args.semantic_diff:
        try:
            git_ls = subprocess.check_output(['git', 'ls-files', '--modified', '--others', '--exclude-standard'], cwd=base_path, text=True).splitlines()
            git_diff = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=base_path, text=True).splitlines()
            for changed_file in sorted(set(git_ls + git_diff)):
                full_path = (base_path / changed_file).resolve()
                if full_path.is_file(): add_target_file(full_path)
        except Exception: pass

    target_files = sorted(set(target_files), key=lambda p: p.as_posix().lower())
    if not target_files:
        logger.error("No valid files found to package")
        sys.exit(1)

    project_name = base_path.name or "workspace"

    # --- CHUNKING LOGIC TO LIMIT FILE SIZES (< 97MB) ---
    CHUNK_SIZE = 3500
    target_file_chunks = [target_files[i:i + CHUNK_SIZE] for i in range(0, len(target_files), CHUNK_SIZE)]
    
    if len(target_file_chunks) > 1:
        logger.info(f"Target file list is large. Splitting into {len(target_file_chunks)} separate bundle parts.")

    for chunk_idx, chunk in enumerate(target_file_chunks):
        part_suffix = f"_part{chunk_idx + 1}" if len(target_file_chunks) > 1 else ""
        logger.info(f"\n=== Generating Bundle Part {chunk_idx + 1}/{len(target_file_chunks)} ({len(chunk)} files) ===")
        
        try:
            packager = WorkspacePackager(
                chunk, base_path, project_name=project_name, focus_target=args.focus,
                depth=args.depth, append_rules=args.append_rules, deterministic=args.deterministic,
                agent_role=args.agent_role, agent_task=args.agent_task, agent_target=args.agent_target,
                workers=args.workers, ast_cache_enabled=(not args.no_ast_cache),
                enable_sast=args.enable_sast,
                semantic_diff=args.semantic_diff,
                use_ray=args.use_ray
            )
            bundle_content = packager.run(args.format)
        except Exception as e:
            logger.error(f"Error generating bundle part {chunk_idx + 1}: {e}")
            sys.exit(1)

        if args.output:
            base_out = pathlib.Path(args.output)
            out_path = base_out.with_name(f"{base_out.stem}{part_suffix}{base_out.suffix}")
        else:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = {"text": "txt", "json": "json"}.get(args.format, "txt")
            out_dir = base_path / f"{project_name}_bundle_{ts}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{project_name}_bundle_{ts}{part_suffix}.{ext}"

        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path.resolve(), "w", encoding="utf-8") as handle:
                handle.write(bundle_content)
            
            logger.info(f"Bundle part saved to: {out_path.resolve()}")
            if packager.bundle_hash:
                print(f"[OK] Bundle Part {chunk_idx + 1} saved: {out_path.resolve()}")
                print(f"  Bundle Hash: {packager.bundle_hash}")
                print(f"  Files: {packager.scan_stats['bundled']} bundled, {packager.scan_stats['skipped']} skipped")
        except Exception as e:
            logger.error(f"Error writing output for part {chunk_idx + 1}: {e}")
            sys.exit(1)

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: sys.exit(130)
    except Exception: sys.exit(1)