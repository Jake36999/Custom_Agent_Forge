"""
Aletheia DAG Engine: Epistemic Graph Interface
Defines the abstract contract for all graph operations, agnostic to AST or OCR source.
"""
from abc import ABC, abstractmethod
from typing import Any, List, Dict

class EpistemicGraphInterface(ABC):
    @abstractmethod
    def active(self) -> bool:
        """Return True if the graph is active (not finalized/collapsed)."""
        pass

    @abstractmethod
    def ready(self) -> List[Dict[str, Any]]:
        """Return a list of node dictionaries that are ready for processing."""
        pass

    @abstractmethod
    def add_nodes(self, nodes: List[Dict[str, Any]]) -> None:
        """Add new nodes to the graph."""
        pass

    @abstractmethod
    def get_branch(self, branch_id: str) -> List[Dict[str, Any]]:
        """Retrieve a branch (subgraph) by branch_id."""
        pass

    @abstractmethod
    def remove_branch(self, branch_id: str) -> None:
        """Remove a branch and all its nodes from the graph."""
        pass

    @abstractmethod
    def branch_ids(self) -> List[str]:
        """Return all branch IDs in the graph."""
        pass

    @abstractmethod
    def validated_nodes(self) -> List[Dict[str, Any]]:
        """Return all nodes that have reached a validated state."""
        pass

    @abstractmethod
    def get_evidence(self, node_id: str) -> List[Dict[str, Any]]:
        """Retrieve the evidence refs associated with a specific node."""
        pass
