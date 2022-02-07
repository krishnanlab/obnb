import functools
import itertools
from typing import List
from typing import Optional
from typing import Union

from ..util import idhandler
from .sparse import SparseGraph


class OntologyGraph(SparseGraph):
    """Ontology graph."""

    def __init__(self):
        """Initialize the ontology graph."""
        super().__init__(weighted=False, directed=True)
        self.idmap = idhandler.IDprop()
        self.idmap.new_property("node_attr", default_val=None)

    def __hash__(self):
        """Trivial hash.

        This hash is solely for the sake of enabling LRU cache when calling
        _aggregate_node_attrs recursion.

        """
        return 0

    def get_node_id(self, node: Union[str, int]) -> str:
        """Return the node ID given the node index.

        Args:
            node (Union[str, int]): Node index (int) or node ID (str). If input
                is already node ID, return directly. If input is node index,
                then return the node ID of the corresponding node index.

        Return:
            str: Node ID.

        """
        return node if isinstance(node, str) else self.idmap.lst[node]

    def set_node_attr(self, node: Union[str, int], node_attr: List[str]):
        """Set node attribute of a given node.

        Args:
            node (Union[str, int]): Node index (int) or node ID (str).
            node_attr (:obj:`list` of :obj:`str`): Node attributes to set.

        """
        self.idmap.set_property(self.get_node_id(node), "node_attr", node_attr)

    def get_node_attr(self, node: Union[str, int]) -> Optional[List[str]]:
        """Get node attribute of a given node.

        Args:
            node (Union[str, int]): Node index (int) or node ID (str).

        """
        return self.idmap.get_property(self.get_node_id(node), "node_attr")

    @functools.lru_cache(maxsize=None)
    def _aggregate_node_attrs(self, node_idx: int) -> List[str]:
        if len(self._edge_data[node_idx]) == 0:  # is leaf node
            node_attr = self.get_node_attr(node_idx) or []
        else:
            children_attrs = [
                self._aggregate_node_attrs(nbr_idx)
                for nbr_idx in self._edge_data[node_idx]
            ]
            self_attrs = self.get_node_attr(node_idx) or []
            node_attr = itertools.chain(*children_attrs, self_attrs)
        return sorted(set(node_attr))

    def complete_node_attrs(self):
        for node_idx in range(self.size):
            self.set_node_attr(node_idx, self._aggregate_node_attrs(node_idx))

    def read_obo(self, path: str):
        """Read OBO-formatted ontology.

        Args:
            path (str): Path to the OBO file.

        """
        raise NotImplementedError("To be implemented")

    @classmethod
    def from_obo(cls, path: str):
        """Construct the ontology graph from an obo file."""
        graph = cls()
        graph.read_obo(path)
        return graph
