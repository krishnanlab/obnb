from __future__ import annotations

from copy import deepcopy

import numpy as np

from nleval.typing import INT_TYPE, Iterable, LogLevel, Optional, Union
from nleval.util import checkers
from nleval.util.idhandler import IDmap
from nleval.util.logger import get_logger


class BaseFeature:
    """BaseFeature object."""

    def __init__(
        self,
        dim: Optional[int] = None,
        log_level: LogLevel = "INFO",
        verbose: bool = False,
    ):
        # TODO: create from dict
        self.idmap = IDmap()
        self._mat = np.array([])

        self.dim = dim
        self.log_level = log_level
        self.verbose = verbose

        self.logger = get_logger(
            self.__class__.__name__,
            log_level=log_level,
            verbose=verbose,
        )

    def copy(self):
        return deepcopy(self)

    @property
    def idmap(self) -> IDmap:
        """Map ID to index."""
        return self._idmap

    @idmap.setter
    def idmap(self, idmap: IDmap):
        checkers.checkType("idmap", IDmap, idmap)
        self._idmap = idmap

    @property
    def size(self) -> int:
        """Number of entities."""
        return self.idmap.size

    def isempty(self) -> bool:
        """Check if the object is empty."""
        return self.idmap.size == 0

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
        if not self.isempty() and self.mat.size > 0:
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

    @property
    def mat(self) -> np.ndarray:
        return self._mat

    @mat.setter
    def mat(self, mat: np.ndarray):
        """Setter for mat.

        Note:
            Matrix must match the dim of both ``self.idmap`` and ``self.dim``.

        """
        checkers.checkType("mat", np.ndarray, mat)
        if mat.size == 0:
            raise ValueError

        if self.dim is None:  # set dim
            self.dim = mat.shape[1]
        elif mat.shape[1] != self.dim:  # check dim of input
            raise ValueError(
                f"Inconsistent dimension between input ({mat.shape[1]}) "
                f"and specified dimension ({self.dim})",
            )

        self._mat = mat

    def add_featvec(self, node_id, vec):
        """Add a new feature vector."""
        # TODO: allow list
        checkers.checkNumpyArrayNDim("vec", 1, vec)
        checkers.checkNumpyArrayIsNumeric("vec", vec)

        # Check size consistency between idmap and mat
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

    def get_featvec(self, ids: Optional[Union[Iterable[str], str]]) -> np.ndarray:
        """Obtain features given entity IDs."""
        # XXX:
        raise NotImplementedError

    def get_featvec_from_idx(
        self,
        idxs: Optional[Union[Iterable[int], int]],
    ) -> np.ndarray:
        """Obtain features given entity indexes."""
        # XXX:
        raise NotImplementedError

    def align(
        self,
        new_fvec: BaseFeature,
        join: str = "right",
        update: bool = False,
    ):
        """Align FeatureVec object with another FeatureVec.

        Utilizes the ``align`` method of ``IDmap`` to align, then update the feature
        vector matrix based on the returned left and right index.

        """
        checkers.checkType("Feature vectors", BaseFeature, new_fvec)
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

    def read_anndata(self, adata, obs_id_name: str = "_index_"):
        """Read feature data from AnnData object.

        Notes:
            This will overwrite existing data in the object.

        Args:
            adata: The AnnData object to be loaded.
            obs_id_name: Name of the observation dataframe column to be used
                as entity IDs. If set to '_index_' (default), then use the
                index column.

        """
        # TODO: add feature ids?
        if obs_id_name == "_index_":
            ids = adata.obs.index.tolist()
        else:
            ids = adata.obs[obs_id_name].tolist()
        self.idmap = IDmap.from_list(ids)
        self.mat = adata.X.toarray()

    @classmethod
    def from_anndata(cls, adata, obs_id_name: str = "_index_", **kwargs):
        """Construct FeatureVec from AnnData.

        Args:
            adata: The AnnData object to be loaded.
            obs_id_name: Name of the observation dataframe column to be used
                as entity IDs. If set to '_index_' (default), then use the
                index column.

        """
        graph = cls(**kwargs)
        graph.read_anndata(adata, obs_id_name)
        return graph

    @classmethod
    def from_mat(
        cls,
        mat: np.ndarray,
        ids: Optional[Union[Iterable[str], IDmap]] = None,
        **kwargs,
    ):
        """Construct feature object using IDs and feature matrix.

        Args:
            mat: 2D numpy array of the feature matrix
            ids: List like object of the entity IDs, or an IDmap object.

        """
        # TODO: refactor the following two lines to a method of idmap
        ids = ids or list(map(str, range(mat.shape[0])))
        idmap = ids if isinstance(ids, IDmap) else IDmap.from_list(ids)
        if idmap.size != mat.shape[0]:
            raise ValueError(
                f"Inconsistent dimension between IDs ({idmap.size}) and the "
                f"matrix ({mat.shape[0]})",
            )

        feat = cls(**kwargs)
        feat.idmap = idmap
        feat.mat = mat

        return feat

    @classmethod
    def from_emd(cls, path_to_emd, **kwargs):
        fvec_lst = []
        idmap = IDmap()
        with open(path_to_emd) as f:
            f.readline()  # skip header
            for line in f:
                terms = line.split(" ")
                node_id = terms[0].strip()
                idmap.add_id(node_id)
                fvec_lst.append(np.array(terms[1:], dtype=float))
        mat = np.asarray(fvec_lst)
        return cls.from_mat(mat, idmap, **kwargs)
