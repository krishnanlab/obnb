from typing import List
from typing import Optional
from typing import Union

import numpy as np

from ..util import checkers
from ..util.idhandler import IDmap
from .base import BaseGraph
from .sparse import SparseGraph


class DenseGraph(BaseGraph):
    """DenseGraph object storing data using numpy array."""

    def __init__(self):
        """Initialize DenseGraph object."""
        super().__init__()
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

    @classmethod
    def from_mat(
        cls,
        mat: np.ndarray,
        ids: Optional[Union[List[str], IDmap]] = None,
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
        graph = cls()
        graph.idmap = idmap
        graph.mat = mat
        return graph

    @classmethod
    def from_npy(cls, path_to_npy, **kwargs):
        """Read numpy array from .npy file and construct BaseGraph."""
        mat = np.load(path_to_npy, **kwargs)
        return cls.from_mat(mat)

    @classmethod
    def from_edglst(cls, path_to_edglst, weighted, directed, **kwargs):
        """Read from edgelist and construct BaseGraph."""
        graph = SparseGraph.from_edglst(
            path_to_edglst,
            weighted,
            directed,
            **kwargs,
        )
        return cls.from_mat(graph.to_adjmat(), graph.idmap)
