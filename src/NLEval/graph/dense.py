import logging

import numpy as np

from ..typing import Dict, List, LogLevel, Optional, Union
from ..util import checkers
from ..util.exceptions import IDNotExistError
from ..util.idhandler import IDmap
from .base import BaseGraph
from .sparse import SparseGraph


class DenseGraph(BaseGraph):
    """DenseGraph object storing data using numpy array."""

    def __init__(
        self,
        log_level: LogLevel = "WARNING",
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize DenseGraph object."""
        super().__init__(log_level=log_level, verbose=verbose, logger=logger)
        self._mat = np.array([])

    def __getitem__(self, key):
        """Return slice of graph.

        Args:
            key(str): key of ID
            key(:obj:`list` of :obj:`str`): list of keys of IDs

        """
        if isinstance(key, slice):
            raise NotImplementedError
        idx = self.idmap[key]
        return self.mat[idx]

    @property
    def num_edges(self) -> int:
        """int: Number of edges."""
        return (self.mat != 0).sum()

    @property
    def mat(self):
        """Node information stored as numpy matrix."""
        return self._mat

    @mat.setter
    def mat(self, val):
        """Setter for mat.

        Note:
            need to construct idmap (self.idmap) first before loading matrix
            (self.mat), which should have same number of entires (rows) as size
            of idmap, riases exption other wise>

        Args:
            val(:obj:`numpy.ndarray`): 2D numpy array

        """
        checkers.checkNumpyArrayIsNumeric("val", val)
        if val.size > 0:
            checkers.checkNumpyArrayNDim("val", 2, val)
            if self.idmap.size != val.shape[0]:
                raise ValueError(
                    f"Expecting {self.idmap.size} entries, not {val.shape[0]}",
                )
        self._mat = val.copy()

    def propagate(self, seed: np.ndarray) -> np.ndarray:
        """Propagate label informmation.

        Args:
            seeds: 1-dimensinoal numpy array where each entry is the seed
                information for a specific node.

        Raises:
            ValueError: If ``seed`` is not a 1-dimensional array with the size
                of number of the nodes in the graph.

        """
        checkers.checkNumpyArrayShape("seed", self.size, seed)
        return np.matmul(self.mat, seed)

    def get_edge(self, node_id1, node_id2):
        """Return edge weight between node_id1 and node_id2.

        Args:
            node_id1(str): ID of first node
            node_id2(str): ID of second node

        """
        return self.mat[self.idmap[node_id1], self.idmap[node_id2]]

    def induced_subgraph(self, node_ids: List[str]):
        """Return a subgraph induced by a subset of nodes.

        Args:
            node_ids (List[str]): List of nodes of interest.

        """
        # Add nodes to new graph and make sure all nodes are present
        for node in node_ids:
            if node not in self.idmap:
                raise IDNotExistError(f"{node!r} is not in the graph")

        # Find index of the corresponding nodes and usge to subset adjmat
        idx = self.idmap[node_ids]

        return self.from_mat(
            self.mat[idx][:, idx],
            node_ids,
            log_level=self.log_level,
            verbose=self.verbose,
        )

    def connected_components(self) -> List[List[str]]:
        """Find connected components via Breadth First Search.

        Returns a list of connected components sorted by the number of nodes,
        each of which is a list of node ids within a connected component.

        Note:
            This BFS approach assumes the graph is undirected.

        """
        unvisited = np.arange(self.num_nodes)
        connected_components = []

        while unvisited.size > 0:
            visited = np.zeros(0)
            tovisit = unvisited[0:1]

            while tovisit.size > 0:
                visited = np.union1d(visited, tovisit)
                tovisit_next = np.where(self.mat[tovisit].sum(0) > 0)[0]
                tovisit = np.setdiff1d(tovisit_next, visited)

            unvisited = np.setdiff1d(unvisited, visited)
            connected_components.append(
                [self.idmap.lst[int(i)] for i in visited],
            )

        return sorted(connected_components, key=len, reverse=True)

    @classmethod
    def from_mat(
        cls,
        mat: np.ndarray,
        ids: Optional[Union[List[str], IDmap]] = None,
        **kwargs,
    ):
        """Construct DenseGraph using ids and adjcency matrix.

        Args:
            mat(:obj:`numpy.ndarray`): 2D numpy array of adjacency matrix
            ids(list or :obj:`IDmap`): list of IDs or idmap of the
                adjacency matrix, if None, use input ordering of nodes as IDs
                (default: :obj:`None`).

        """
        if ids is None:
            ids = list(map(str, range(mat.shape[0])))
        idmap = ids if isinstance(ids, IDmap) else IDmap.from_list(ids)
        if idmap.size != mat.shape[0]:
            raise ValueError(
                f"Inconsistent dimension between IDs ({idmap.size}) and the "
                f"matrix ({mat.shape[0]})",
            )
        graph = cls(**kwargs)
        graph.idmap = idmap
        graph.mat = mat
        return graph

    @classmethod
    def from_npy(cls, path_to_npy, **kwargs):
        """Read numpy array from .npy file and construct BaseGraph."""
        mat = np.load(path_to_npy, **kwargs)
        return cls.from_mat(mat)

    @classmethod
    def from_edgelist(cls, path_to_edgelist, weighted, directed, **kwargs):
        """Read from edgelist and construct BaseGraph."""
        graph = SparseGraph.from_edgelist(
            path_to_edgelist,
            weighted,
            directed,
            **kwargs,
        )
        return cls.from_mat(graph.to_adjmat(), graph.idmap)

    @classmethod
    def from_cx_stream_file(cls, *args, **kwargs):
        """Construct DenseGraph from CX stream files."""
        graph = SparseGraph.from_cx_stream_file(*args, **kwargs)
        return cls.from_mat(graph.to_adjmat(), graph.idmap)

    def to_sparse_graph(self):
        """Convert DenseGraphh to a SparseGraph."""
        return SparseGraph.from_mat(self.mat, self.idmap)

    def save_npz(self, out_path: str, key_map: Optional[Dict[str, str]] = None):
        """Save the graph as dense array npz file.

        The npz file contains two fields, including "adj" and "node_ids". The
        two keys can be replaced using the key_map argument.

        Args:
            out_path (str): path to the output file.
            key_map: Dictionary mapping the default keys to new keys.

        """
        default_key_map = {"adj": "adj", "node_ids": "node_ids"}
        default_key_map.update(key_map or {})
        adj_key, ids_key = default_key_map["adj"], default_key_map["node_ids"]
        np.savez(out_path, **{adj_key: self.mat, ids_key: self.node_ids})
