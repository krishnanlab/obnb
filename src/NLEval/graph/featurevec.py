from __future__ import annotations

import logging
from itertools import chain

import numpy as np
from scipy.spatial import distance

from NLEval.typing import INT_TYPE, LogLevel, Optional, Sequence
from NLEval.util import checkers
from NLEval.util.idhandler import IDmap
from NLEval.graph.dense import DenseGraph


class FeatureVec(DenseGraph):
    """Feature vectors object."""

    def __init__(
        self,
        dim=None,
        log_level: LogLevel = "WARNING",
        verbose: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize FeatureVec object."""
        # TODO: create from dict
        super().__init__(log_level=log_level, verbose=verbose, logger=logger)
        self.dim = dim

    @property
    def dim(self):
        """int: dimension of feature vectors."""
        return self._dim

    @dim.setter
    def dim(self, d):
        checkers.checkNullableType("d", INT_TYPE, d)
        if d is not None:
            if d < 1:
                raise ValueError(
                    f"Feature dimension must be greater than 1, got {d}",
                )
        if not self.isempty():
            if d != self.mat.shape[1]:
                # self.dim should always in sync with actual dim of feature vec
                if self.dim != self.mat.shape[1]:
                    self.logger.critical(
                        "Mismatching dimensions. This should never happen!",
                    )
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

    def align(
        self,
        new_fvec: FeatureVec,
        join: str = "right",
        update: bool = False,
    ):
        """Align FeatureVec object with another FeatureVec.

        Utilizes the ``align`` method of ``IDmap`` to align, then update the
        feature vector matrix based on the returned left and right index.

        """
        checkers.checkType("Feature vectors", FeatureVec, new_fvec)
        new_idmap = new_fvec.idmap
        l_idx, r_idx = self.idmap.align(new_idmap, join=join, update=update)

        if join == "right":
            new_mat = np.zeros((len(new_idmap), self.mat.shape[1]))
            new_mat[r_idx] = self.mat[l_idx]
            self._mat = new_mat
        elif join == "left":
            if update:
                new_mat = np.zeros((len(self.idmap), new_fvec.mat.shape[1]))
                new_mat[l_idx] = new_fvec.mat[r_idx]
                new_fvec._mat = new_mat
        elif join == "intersection":
            self._mat = self._mat[l_idx]
            if update:
                new_fvec._mat = new_fvec._mat[r_idx]
        elif join == "union":
            new_mat = np.zeros((len(self.idmap), self.mat.shape[1]))
            new_mat[l_idx] = self._mat
            self._mat = new_mat

            if update:
                new_mat = np.zeros((len(self.idmap), new_fvec.mat.shape[1]))
                new_mat[r_idx] = new_fvec._mat
                new_fvec._mat = new_mat
        else:
            raise ValueError(f"Unrecognized join type {join!r}")

    def align_to_idmap(self, new_idmap):
        """Align FeatureVec to a given idmap.

        This is essentially right align with update = False, i.e. reorder the
        current FeatureVec using the new_idmap.

        """
        checkers.checkType("IDmap", IDmap, new_idmap)
        l_idx, r_idx = self.idmap.align(new_idmap, join="right", update=False)

        new_mat = np.zeros((len(new_idmap), self.mat.shape[1]))
        new_mat[r_idx] = self.mat[l_idx]
        self._mat = new_mat

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

    def __init__(
        self,
        log_level: LogLevel = "WARNING",
        verbose: bool = True,
    ):
        """Initialize MultiFeatureVec."""
        super().__init__(log_level=log_level, verbose=verbose)
        self.indptr = None
        self.fset_idmap = IDmap()

    @property
    def feature_ids(self) -> tuple[str, ...]:
        """Return feature IDs as a tuple."""
        return tuple(self.fset_idmap.lst)

    def get_features_from_idx(
        self,
        idx: Sequence[int] | int,
        fset_idx: Sequence[int] | int,
    ) -> np.ndarray:
        """Return features given node index and feature set inde.

        Args:
            idx (int or sequence of int): node index of interest.
            fset_id (int or sequence of int): feature set index of interest.

        """
        if isinstance(idx, int):  # return as one 2-d array with one row
            idx = [idx]

        indptr = self.indptr
        if isinstance(fset_idx, int):
            fset_slice = slice(indptr[fset_idx], indptr[fset_idx + 1])
        else:
            fset_slices = [list(range(indptr[i], indptr[i + 1])) for i in fset_idx]
            fset_slice = list(chain(*fset_slices))

        return self.mat[idx][:, fset_slice]

    def get_features(
        self,
        ids: str | list[str] | None = None,
        fset_ids: str | list[str] | None = None,
    ) -> np.ndarray:
        """Return features given node IDs and the selected feature set ID.

        Args:
            ids (str or list of str, optional): node ID(s) of interest, return
                a 1-d array if input a single id, otherwise return a 2-d array
                where each row is the feature vector with the corresponding
                node ID. If not specified, use all rows.
            fset_ids (str or list of str, optional): feature set ID(s) of
                interest. If not specified, use all columns.

        """
        if ids is None:
            idx = list(range(len(self.idmap)))
        else:
            idx = self.idmap[ids]

        if fset_ids is None:
            fset_idx = list(range(len(self.fset_idmap)))
        else:
            fset_idx = self.fset_idmap[fset_ids]

        return self.get_features_from_idx(idx, fset_idx)

    @classmethod
    def from_mat(
        cls,
        mat: np.ndarray,
        indptr: np.ndarray | None = None,
        ids: list[str] | IDmap | None = None,
        fset_ids: list[str] | IDmap | None = None,
    ):
        """Construct MultiFeatureVec object.

        Args:
            mat (:obj:`numpy.ndarray`): concatenated feature vector matrix.
            indptr (:obj:`numpy.ndarray`, optional): index pointers indicating
                the start and the end of each feature set (columns). If set to
                None, and the dimension of fset_ids matches the number of
                columns in the input matrix, then automatically set indptr
                to corresponding to all ones.
            ids (list of str or :obj:`IDmap`, optional): node IDs, if not
                specified, use the default ordering as node IDs.
            fset_ids (list of str or :obj:`IDmap`, optional): feature set IDs,
                if not specified, use the default ordering as feature set IDs.

        """
        if indptr is None:
            if fset_ids is None:
                raise ValueError("Cannot set both indptr and fset_ids to None.")
            if len(fset_ids) != mat.shape[1]:
                raise ValueError(
                    "Cannot asign indptr automatically because the  dimension "
                    f"of fset_ids ({len(fset_ids)}) does not match the number "
                    f"of columsn in the input matrix ({mat.shape[1]}). Please "
                    "specify fset_ids",
                )
            indptr = np.arange(mat.shape[1] + 1)

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
        mats: list[np.ndarray],
        ids: list[str] | IDmap | None = None,
        fset_ids: list[str] | IDmap | None = None,
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
