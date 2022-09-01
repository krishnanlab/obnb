import logging
from copy import deepcopy

from NLEval.exception import IDExistsError
from NLEval.typing import EdgeDir, Iterable, List, LogLevel, Optional, Tuple, Union
from NLEval.util import checkers, idhandler
from NLEval.util.checkers import checkLiteral
from NLEval.util.logger import get_logger


class BaseGraph:
    """Base Graph object that contains basic graph operations."""

    def __init__(
        self,
        log_level: LogLevel = "WARNING",
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize BaseGraph object."""
        self.verbose = verbose
        if logger is None:
            self.log_level = log_level
            self.logger = get_logger(
                self.__class__.__name__,
                log_level=log_level,
                verbose=verbose,
            )
        else:
            self.logger = logger
            self.log_level = logging.getLevelName(logger.getEffectiveLevel())
        self.idmap = idhandler.IDmap()

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

    def add_node(self, node: str, exist_ok: bool = False):
        """Add a new node to the graph.

        Args:
            node: Name (or ID) of the node.
            exist_ok: Do not raise IDExistsError even if the node to be added
                already exist.

        """
        try:
            self.idmap.add_id(node)
            self._new_node_data()
        except IDExistsError as e:
            if not exist_ok:
                raise e

    def _new_node_data(self):
        """Set up data for the newly added node."""
        raise NotImplementedError(
            f"The add_node method has not been set up for {self.__class__!r}, "
            f"need to implement method _new_node_data first.",
        )

    def add_nodes(self, nodes: Iterable[str], exist_ok: bool = False):
        """Add new nodes to the graph.

        Args:
            node: Names (or IDs) of the nodes.
            exist_ok: Do not raise IDExistsError even if the node to be added
                already exist.

        """
        for node in nodes:
            self.add_node(node, exist_ok=exist_ok)

    def add_edge(
        self,
        node_id1: str,
        node_id2: str,
        weight: float = 1.0,
        reduction: Optional[str] = None,
    ):
        """Add or update an edge in the graph.

        Args:
            node_id1: ID of node 1.
            node_id2: ID of node 2.
            weight: Edge weight to use.
            reduction: Type of edge reduction to use if the target edge already
                exist. If not set, warn if old edge exists with different edge
                weight value then the input edge weight, and then overwite it
                with the new value.

        """
        raise NotImplementedError

    def remove_edge(self, node_id1: str, node_id2: str):
        """Remove an edge in the graph.

        Args:
            node_id1: ID of node 1.
            node_id2: ID of node 2.

        """
        raise NotImplementedError

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

    def get_neighbors(
        self,
        node: Union[str, int],
        direction: EdgeDir = "both",
    ) -> List[str]:
        """Get neighboring nodes of the input node.

        Args:
            node: Node index (int) or node ID (str).
            direction: Direction of the edges to be considered
                ["in", "out", "both"], default is "both".

        Return:
            List[str]: List of neighboring node IDs.

        """
        checkLiteral("direction", EdgeDir, direction)
        node_idx = self.get_node_idx(node)
        nbr_idxs = self._get_nbr_idxs(node_idx, direction)
        nbr_ids = self.idmap.get_ids(nbr_idxs)
        return nbr_ids

    def _get_nbr_idxs(self, node_idx: int, direction: EdgeDir) -> List[int]:
        """Return neighboring node indexes given the current node index."""
        raise NotImplementedError

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

    @property
    def num_edges(self) -> int:
        """int: Number of edges."""
        raise NotImplementedError

    def isempty(self):
        """bool: true if graph is empty, indicated by empty idmap."""
        return not self.size

    def induced_subgraph(self, node_ids: List[str]):
        """Return a subgraph induced by a subset of nodes."""
        raise NotImplementedError

    def connected_components(self) -> List[List[str]]:
        """Find connected components."""
        raise NotImplementedError

    def is_connected(self) -> bool:
        """Retrun True if the graph is connected."""
        return len(self.connected_components()) == 1

    def largest_connected_subgraph(self):
        """Return the largest connected subgraph of the graph."""
        comps = self.connected_components()
        self.logger.info(f"Components sizes = {list(map(len, comps))}")

        subgraph = self.induced_subgraph(comps[0])
        self.logger.info(
            f"Number of nodes = {subgraph.num_nodes:,} ({self.num_nodes:,} "
            f"originally), number of edges = {subgraph.num_edges:,} "
            f"({self.num_edges:,} originally)",
        )

        return subgraph
