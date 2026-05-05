"""
Aletheia Graph Backend Shim (D9)
Conditional cuGraph GPU backend with NetworkX fallback.
"""

import logging
import networkx as nx

logger = logging.getLogger(__name__)

GPU_AVAILABLE = False
_cugraph = None

try:
    import cugraph as _cugraph_module
    _cugraph = _cugraph_module
    GPU_AVAILABLE = True
    logger.info("[GraphBackend] cuGraph GPU backend detected and available.")
except ImportError:
    GPU_AVAILABLE = False
    logger.info("[GraphBackend] cuGraph not available — using NetworkX CPU backend.")


def is_gpu_available() -> bool:
    """Returns True if cuGraph GPU backend is available."""
    return GPU_AVAILABLE


def get_graph_backend():
    """
    Returns the active graph backend module.
    cuGraph when GPU available, NetworkX otherwise.
    """
    if GPU_AVAILABLE and _cugraph is not None:
        return _cugraph
    return nx
