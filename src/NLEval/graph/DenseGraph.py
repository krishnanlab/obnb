from NLEval.graph.BaseGraph import BaseGraph
from NLEval.graph.SparseGraph import SparseGraph
from NLEval.util import checkers, IDHandler
from scipy.spatial import distance
import numpy as np

__all__ = ["DenseGraph", "FeatureVec"]


class DenseGraph(BaseGraph):
    """Base Graph object that stores data using numpy array"""

    def __init__(self):
        super().__init__()
        self._mat = np.array([])

    def __getitem__(self, key):
        """Return slice of graph

        Args:
            key(str): key of ID
            key(:obj:`list` of :obj:`str`): list of keys of IDs
        """
        if isinstance(key, slice):
            raise NotImplementedError
        idx = self.IDmap[key]
        return self.mat[idx]

    @property
    def mat(self):
        """Node information stored as numpy matrix"""
        return self._mat

    @mat.setter
    def mat(self, val):
        """Setter for DenseGraph.mat
        Note: need to construct IDmap (self.IDmap) first before
        loading matrix (self.mat), which should have same number of
        entires (rows) as size of IDmap, riases exption other wise

        Args:
            val(:obj:`numpy.ndarray`): 2D numpy array
        """
        checkers.checkNumpyArrayIsNumeric("val", val)
        if val.size > 0:
            checkers.checkNumpyArrayNDim("val", 2, val)
            if self.IDmap.size != val.shape[0]:
                raise ValueError(
                    f"Expecting {self.IDmap.size} entries, not {val.shape[0]}",
                )
        self._mat = val.copy()

    def get_edge(self, ID1, ID2):
        """Return edge weight between ID1 and ID2

        Args:
            ID1(str): ID of first node
            ID2(str): ID of second node
        """
        return self.mat[self.IDmap[ID1], self.IDmap[ID2]]

    @classmethod
    def construct_graph(cls, ids, mat):
        """Construct DenseGraph using ids and adjcency matrix

        Args:
            ids(list or :obj:`IDHandler.IDmap`): list of IDs or IDmap of the
                adjacency matrix
            mat(:obj:`numpy.ndarray`): 2D numpy array of adjacency matrix

        """
        idmap = (
            ids
            if isinstance(ids, IDHandler.IDmap)
            else IDHandler.IDmap.from_list(ids)
        )
        if idmap.size != mat.shape[0]:
            raise ValueError(
                f"Inconsistent dimension between IDs ({idmap.size}) and the "
                f"matrix ({mat.shape[0]})",
            )
        graph = cls()
        graph.IDmap = idmap
        graph.mat = mat
        return graph

    @classmethod
    def from_mat(cls, mat):
        """Construct DenseGraph object from numpy array

        Note:
            First column of mat encodes ID, must be integers

        """
        idmap = IDHandler.IDmap()
        for ID in mat[:, 0]:
            if int(ID) != ID:
                raise ValueError("ID must be int type")
            idmap.addID(str(int(ID)))
        return cls.construct_graph(idmap, mat[:, 1:].astype(float))

    @classmethod
    def from_npy(cls, path_to_npy, **kwargs):
        """Read numpy array from .npy file and construct BaseGraph"""
        mat = np.load(path_to_npy, **kwargs)
        return cls.from_mat(mat)

    @classmethod
    def from_edglst(cls, path_to_edglst, weighted, directed, **kwargs):
        """Read from edgelist and construct BaseGraph"""
        graph = SparseGraph.from_edglst(
            path_to_edglst, weighted, directed, **kwargs
        )
        return cls.construct_graph(graph.IDmap, graph.to_adjmat())


class FeatureVec(DenseGraph):
    """Feature vectors with ID maps"""

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    @property
    def dim(self):
        """int: dimension of feature vectors"""
        return self._dim

    @dim.setter
    def dim(self, d):
        checkers.checkTypeAllowNone("d", checkers.INT_TYPE, d)
        if d is not None:
            if d < 1:
                raise ValueError(
                    f"Feature dimension must be greater than 1, got {d}",
                )
        if not self.isempty():
            if d != self.mat.shape[1]:
                if self.dim != self.mat.shape[1]:
                    # self.dim should always in sync with actual dim of feature vec
                    print("CRITICAL: This should never happen!")
                raise ValueError(
                    f"Inconsistent dimension between input ({d}) and data "
                    f"({self.mat.shape[1]})",
                )
        self._dim = d

    @DenseGraph.mat.setter
    def mat(self, val):
        """Setter for FeatureVec.mat
        Note: matrix must match dimension of both self.IDmap and self.dim
        """
        mat_bkp = self.mat  # create backup copy
        DenseGraph.mat.fset(self, val)
        if val.size > 0:
            if self.dim is None:  # set dim
                self.dim = val.shape[1]
            elif self.mat.shape[1] != self.dim:  # check dim of input
                self._mat = mat_bkp
                raise ValueError(
                    f"Inconsistent dimension between input ({val.shape[1]}) "
                    f"and specified dimension ({self.dim})",
                )

    def get_edge(self, ID1, ID2, dist_fun=distance.cosine):
        """Return pairwise similarity of two features as 'edge'

        Args:
            ID1(str): ID of first node
            ID2(str): ID of second node
            dist_fun: function to calculate distance between two vectors
                        default as cosine similarity
        """
        return dist_fun(self[ID1], self[ID2])

    def addVec(self, ID, vec):
        """Add a new feature vector"""
        checkers.checkNumpyArrayNDim("vec", 1, vec)
        checkers.checkNumpyArrayIsNumeric("vec", vec)

        # check size consistency between IDmap and mat
        if self.size != self.mat.shape[0]:
            raise ValueError(
                f"Inconsistent number of IDs ({self.IDmap.size}) and matrix "
                f"entries ({self.mat.shape[0]})",
            )

        if self.isempty():
            if self.dim is not None:
                checkers.checkNumpyArrayShape("vec", self.dim, vec)
            else:
                self.dim = vec.shape[0]
            new_mat = vec.copy().reshape((1, vec.size))
        else:
            new_mat = np.vstack([self.mat, vec])
        self.IDmap.addID(ID)
        self.mat = new_mat

    @classmethod
    def from_emd(cls, path_to_emd, **kwargs):
        fvec_lst = []
        idmap = IDHandler.IDmap()
        with open(path_to_emd, "r") as f:
            f.readline()  # skip header
            for line in f:
                terms = line.split(" ")
                ID = terms[0].strip()
                idmap.addID(ID)
                fvec_lst.append(np.array(terms[1:], dtype=float))
        mat = np.asarray(fvec_lst)
        return cls.construct_graph(idmap, mat)


class MultiFeatureVec(BaseGraph):
    """Multi feature vectors with ID maps

    Note: experimenting feature

    """

    def __init__(self):
        self.mat_list = []
        self.name_list = []

    def add_feature(self, val, name):
        self.mat_list.append(val)
        self.name_list.append(name)
