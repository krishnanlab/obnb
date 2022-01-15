from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
from scipy.spatial import distance

from ..util import checkers
from ..util.idhandler import IDmap
from .dense import DenseGraph


class FeatureVec(DenseGraph):
    """Feature vectors object."""

    def __init__(self, dim=None):
        """Initialize FeatureVec object."""
        # TODO: create from dict
        super().__init__()
        self.dim = dim

    @property
    def dim(self):
        """int: dimension of feature vectors."""
        return self._dim

    @dim.setter
    def dim(self, d):
        checkers.checkNullableType("d", checkers.INT_TYPE, d)
        if d is not None:
            if d < 1:
                raise ValueError(
                    f"Feature dimension must be greater than 1, got {d}",
                )
        if not self.isempty():
            if d != self.mat.shape[1]:
                # self.dim should always in sync with actual dim of feature vec
                if self.dim != self.mat.shape[1]:
                    print("CRITICAL: This should never happen!")
                raise ValueError(
                    f"Inconsistent dimension between input ({d}) and data "
                    f"({self.mat.shape[1]})",
                )
        self._dim = d

    @DenseGraph.mat.setter
    def mat(self, val):
        """Setter for mat.

        Note:
            Matrix must match the dim of both ``self.idmap`` and ``self.dim``.

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

    def propagate(self, seed):
        raise NotImplementedError("Feature vectors do can not propagate")

    def get_edge(self, node_id1, node_id2, dist_fun=distance.cosine):
        """Return pairwise similarity of two features as 'edge'.

        Args:
            node_id1(str): ID of the first node.
            node_id2(str): ID of the second node.
            dist_fun: function to calculate distance between two vectors
                default as cosine similarity.

        """
        return dist_fun(self[node_id1], self[node_id2])

    def add_vec(self, node_id, vec):
        """Add a new feature vector."""
        # TODO: allow list
        checkers.checkNumpyArrayNDim("vec", 1, vec)
        checkers.checkNumpyArrayIsNumeric("vec", vec)

        # check size consistency between idmap and mat
        if self.size != self.mat.shape[0]:
            raise ValueError(
                f"Inconsistent number of IDs ({self.idmap.size}) and matrix "
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
        self.idmap.add_id(node_id)
        self.mat = new_mat

    def to_pyg_x(self):
        """Return a copy of the feature matrix."""
        return self.mat.copy()

    @classmethod
    def from_emd(cls, path_to_emd, **kwargs):
        fvec_lst = []
        idmap = IDmap()
        with open(path_to_emd, "r") as f:
            f.readline()  # skip header
            for line in f:
                terms = line.split(" ")
                node_id = terms[0].strip()
                idmap.add_id(node_id)
                fvec_lst.append(np.array(terms[1:], dtype=float))
        mat = np.asarray(fvec_lst)
        return cls.from_mat(mat, idmap)


class MultiFeatureVec(FeatureVec):
    """Multiple feature vectors."""

    def __init__(self):
        """Initialize MultiFeatureVec."""
        super().__init__()
        self.indptr = None
        self.fset_idmap = IDmap()

    def get_features_from_idx(
        self,
        idx: Sequence[int],
        fset_id: str,
    ) -> np.ndarray:
        """Return features given node index and the selected feature set ID.

        Args:
            idx (sequence of int): node index of interest.
            fset_id (str): feature set ID.

        """
        fset_idx = self.fset_idmap[fset_id]
        fset_slice = slice(self.indptr[fset_idx], self.indptr[fset_idx + 1])
        return self.mat[idx, fset_slice]

    def get_features(
        self,
        ids: Union[str, List[str]],
        fset_id: str,
    ) -> np.ndarray:
        """Return features given node IDs and the selected feature set ID.

        Args:
            ids (str or list of str): node ID(s) of interest, return a 1-d
                array if input a single id, otherwise return a 2-d array
                where each row is the feature vector with the corresponding
                node ID.
            fset_id (str): feature set ID.

        """
        idx = self.idmap[ids]
        return self.get_features_from_idx(idx, fset_id)

    @classmethod
    def from_mat(
        cls,
        mat: np.ndarray,
        indptr: np.ndarray,
        ids: Optional[Union[List[str], IDmap]] = None,
        fset_ids: Optional[Union[List[str], IDmap]] = None,
    ):
        """Construct MultiFeatureVec object.

        Args:
            mat (:obj:`numpy.ndarray`): concatenated feature vector matrix.
            indptr (:obj:`numpy.ndarray`): index pointers indicating the start
                and the end of each feature set (columns).
            ids (list of str or :obj:`IDmap`, optional): node IDs, if not
                specified, use the default ordering as node IDs.
            fset_ids (list of str or :obj:`IDmap`, optional): feature set IDs,
                if not specified, use the default ordering as feature set IDs.

        """
        # TODO: refactor the following block(s)
        if ids is None:
            ids = list(map(str, range(mat.shape[0])))
        if isinstance(ids, IDmap):
            idmap = ids
        else:
            idmap = IDmap.from_list(ids)

        if fset_ids is None:
            fset_ids = list(map(str, range(indptr.size - 1)))
        if isinstance(fset_ids, IDmap):
            fset_idmap = fset_ids
        else:
            fset_idmap = IDmap.from_list(fset_ids)

        graph = super().from_mat(mat, ids)
        graph.indptr = indptr  # TODO: check indptr
        graph.idmap = idmap
        graph.fset_idmap = fset_idmap

        return graph

    @classmethod
    def from_mats(
        cls,
        mats: List[np.ndarray],
        ids: Optional[Union[List[str], IDmap]] = None,
        fset_ids: Optional[Union[List[str], IDmap]] = None,
    ):
        """Construct MultiFeatureVec object from list of matrices.

        Args:
            mats (list of :obj:`numpy.ndarray`): list of feature vecotr
                matrices.
            ids (list of str or :obj:`IDmap`, optional): node IDs, if not
                specified, use the default ordering as node IDs.
            fset_ids (list of str or :obj:`IDmap`, optional): feature set IDs,
                if not specified, use the default ordering as feature set IDs.

        """
        dims = [mat.shape[1] for mat in mats]
        indptr = np.zeros(len(mats) + 1, dtype=np.uint32)
        indptr[1:] = np.cumsum(dims)
        return cls.from_mat(np.hstack(mats), indptr, ids, fset_ids)
