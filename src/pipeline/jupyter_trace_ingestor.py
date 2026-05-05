"""
jupyter_trace_ingestor.py
Loads Jupyter notebooks and streams cells as structured dicts for the
VeteranStateMachine. Each cell dict carries: type, source, outputs,
and a pre-parsed `traceback_text` if the cell errored.
"""

import json
from pathlib import Path
from typing import Iterator, Dict, Any

try:
    import nbformat
    _HAS_NBFORMAT = True
except ImportError:
    _HAS_NBFORMAT = False


def load_notebook(path: str):
    """Parse a .ipynb file. Returns an nbformat NotebookNode (or a fallback dict)."""
    nb_path = Path(path)
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {path}")

    if _HAS_NBFORMAT:
        with open(nb_path, "r", encoding="utf-8", errors="replace") as f:
            return nbformat.read(f, as_version=4)
    else:
        with open(nb_path, "r", encoding="utf-8", errors="replace") as f:
            return json.load(f)


def _extract_traceback(outputs) -> str:
    """Pull traceback text out of a cell's output list."""
    parts = []
    for out in (outputs or []):
        otype = out.get("output_type", "")
        if otype == "error":
            ename = out.get("ename", "")
            evalue = out.get("evalue", "")
            tb_lines = out.get("traceback", [])
            clean = "\n".join(
                line.replace("\x1b[0;31m", "").replace("\x1b[0m", "").replace("\x1b[1;32m", "")
                for line in tb_lines
            )
            parts.append(f"{ename}: {evalue}\n{clean}".strip())
        elif otype == "stream" and out.get("name") == "stderr":
            text = out.get("text", "")
            if "Traceback" in text or "Error" in text:
                parts.append(text.strip())
    return "\n\n".join(parts)


def stream_cells(nb_node) -> Iterator[Dict[str, Any]]:
    """
    Yield one dict per notebook cell with these keys:
      cell_type   : 'code' | 'markdown' | 'raw'
      source      : str  (the cell's source text)
      outputs     : list (raw output objects)
      traceback   : str  (populated only when a traceback was found in outputs)
      exec_count  : int | None
    """
    cells = nb_node.get("cells", [])
    for idx, cell in enumerate(cells):
        ctype = cell.get("cell_type", "code")
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        outputs = cell.get("outputs", [])
        tb = _extract_traceback(outputs) if ctype == "code" else ""
        yield {
            "cell_index": idx,
            "cell_type": ctype,
            "source": source.strip(),
            "outputs": outputs,
            "traceback": tb,
            "exec_count": cell.get("execution_count"),
        }
