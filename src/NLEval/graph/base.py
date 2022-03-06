from copy import deepcopy
from typing import Tuple
from typing import Union

from ..util import checkers
from ..util import idhandler
from ..util.logger import get_logger
from ..util.types import LogLevel


class BaseGraph:
    """Base Graph object that contains basic graph operations."""

    def __init__(self, log_level: LogLevel = "WARNING", verbose: bool = False):
        """Initialize BaseGraph object."""
        self.idmap = idhandler.IDmap()
        self.logger = get_logger(
            self.__class__.__name__,
            log_level=log_level,
            verbose=verbose,
        )

    @property
    def node_ids(self) -> Tuple[str, ...]:
        """Return node IDs as a tuple."""
        return tuple(self.idmap.lst)

    @property
    def idmap(self):
        """Map node ID to the corresponding index."""
        return self._idmap

    @idmap.setter
    def idmap(self, idmap):
        checkers.checkType("idmap", idhandler.IDmap, idmap)
        self._idmap = idmap

    def get_node_id(self, node: Union[str, int]) -> str:
        """Return the node ID given the node index or node ID.

        Args:
            node (Union[str, int]): Node index (int) or node ID (str). If input
                is already node ID, return directly. If input is node index,
                then return the node ID of the corresponding node index.

        Return:
            str: Node ID.

        """
        return node if isinstance(node, str) else self.idmap.lst[node]

    def get_node_idx(self, node: Union[str, int]) -> int:
        """Return the node index given the node ID or node index.

        Args:
            node (Union[str, int]): Node index (int) or node ID (str). If input
                is already node index, return directly. If input is node index,
                then return the node index of the corresponding node ID.

        Return:
            int: Node index.

        """
        return node if isinstance(node, int) else self.idmap[node]

    def copy(self):
        return deepcopy(self)

    def __contains__(self, graph):
        """Return true if contains the input graph."""
        # Check if containes all IDs in input graph
        for node_id in graph.idmap:
            if node_id not in self.idmap:
                return False

        for node_id1 in graph.idmap:  # check if all connections match
            for node_id2 in graph.idmap:
                if self.get_edge(node_id1, node_id2) != graph.get_edge(
                    node_id1,
                    node_id2,
                ):
                    return False
        return True

    def __eq__(self, graph):
        """Return true if input graph is identical to self.

        For example, same set of IDs with same connections.

        """
        if self.idmap == graph.idmap:
            return graph in self
        return False

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the graph indicated by the ID map."""
        return self.idmap.size

    @property
    def size(self):
        """int: number of nodes in graph."""
        return self.num_nodes

    def isempty(self):
        """bool: true if graph is empty, indicated by empty idmap."""
        return not self.size
