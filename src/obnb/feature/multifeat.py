"""Multi modality feature objects."""
from itertools import chain

import numpy as np

from obnb.feature.base import BaseFeature
from obnb.typing import Iterable, List, LogLevel, Optional, Tuple, Union
from obnb.util.idhandler import IDmap


class MultiFeatureVec(BaseFeature):
    """MultiFeatureVec object."""

    def __init__(
        self,
        log_level: LogLevel = "INFO",
        verbose: bool = False,
    ):
        """Initialize MultiFeatureVec."""
        super().__init__(log_level=log_level, verbose=verbose)
        self.indptr = np.array([], dtype=np.uint32)
        self.fset_idmap = IDmap()

    @property
    def feature_ids(self) -> Tuple[str, ...]:
        """Return feature IDs."""
        return tuple(self.fset_idmap.lst)

    def get_features_from_idx(
        self,
        idx: Union[Iterable[int], int],
        fset_idx: Union[Iterable[int], int],
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
            fset_slice = list(chain(*fset_slices))  # type: ignore

        return self.mat[idx][:, fset_slice]

    def get_features(
        self,
        ids: Optional[Union[List[str], str]] = None,
        fset_ids: Optional[Union[List[str], str]] = None,
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
        ids: Optional[Union[Iterable[str], IDmap]] = None,
        *,
        indptr: Optional[np.ndarray] = None,
        fset_ids: Optional[Union[Iterable[str], IDmap]] = None,
        **kwargs,
    ):
        """Construct MultiFeatureVec object.

        Args:
            mat (:obj:`numpy.ndarray`): concatenated feature vector matrix.
            ids (list of str or :obj:`IDmap`, optional): node IDs, if not
                specified, use the default ordering as node IDs.
            indptr (:obj:`numpy.ndarray`, optional): index pointers indicating
                the start and the end of each feature set (columns). If set to
                None, and the dimension of fset_ids matches the number of
                columns in the input matrix, then automatically set indptr
                to corresponding to all ones.
            fset_ids (list of str or :obj:`IDmap`, optional): feature set IDs,
                if not specified, use the default ordering as feature set IDs.

        """
        if indptr is None:
            if fset_ids is None:
                raise ValueError("Cannot set both indptr and fset_ids to None.")
            if (num_fsets := len(fset_ids)) != mat.shape[1]:  # type: ignore
                raise ValueError(
                    "Cannot assign indptr automatically because the  dimension "
                    f"of fset_ids ({num_fsets}) does not match the number "
                    f"of columns in the input matrix ({mat.shape[1]}). Please "
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

        feat = super().from_mat(mat, ids, **kwargs)
        feat.indptr = indptr  # TODO: check indptr
        feat.idmap = idmap
        feat.fset_idmap = fset_idmap

        return feat

    @classmethod
    def from_mats(
        cls,
        mats: List[np.ndarray],
        ids: Optional[Union[List[str], IDmap]] = None,
        *,
        fset_ids: Optional[Union[List[str], IDmap]] = None,
        **kwargs,
    ):
        """Construct MultiFeatureVec object from list of matrices.

        Args:
            mats (list of :obj:`numpy.ndarray`): list of feature vector
                matrices.
            ids (list of str or :obj:`IDmap`, optional): node IDs, if not
                specified, use the default ordering as node IDs.
            fset_ids (list of str or :obj:`IDmap`, optional): feature set IDs,
                if not specified, use the default ordering as feature set IDs.

        """
        dims = [mat.shape[1] for mat in mats]
        indptr = np.zeros(len(mats) + 1, dtype=np.uint32)
        indptr[1:] = np.cumsum(dims)
        mat = np.hstack(mats)
        return cls.from_mat(mat, ids, indptr=indptr, fset_ids=fset_ids, **kwargs)
