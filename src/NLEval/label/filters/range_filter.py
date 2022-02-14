from typing import Optional

import numpy as np

from ..collection import Splitter
from .base import BaseFilter


class BaseRangeFilter(BaseFilter):
    """Filter entities in labelset collection by range of values.

    Notes:
        If ``min_val`` or ``max_val`` is not specified, no filtering
        will be done on upper/lower bound.

    """

    def __init__(
        self,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> None:
        """Initialize BaseRangeFilter object.

        Args:
            min_val: minimum below which entities are removed
            max_val: maximum beyound which entiteis are removed

        """
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def __repr__(self):
        min_val, max_val = self.min_val, self.max_val
        return f"{super().__repr__()}({min_val=}, {max_val=})"

    def criterion(self, val):
        if self.min_val is not None:
            if val < self.min_val:
                return True
        if self.max_val is not None:
            if val > self.max_val:
                return True
        return False


class EntityRangeFilterNoccur(BaseRangeFilter):
    """Filter entities based on number of occurance.

    Example:
        The following example removes any entity that occurs to be positive
        in more than 10 labelsets.

        >>> labelset_collection.apply(EntityRangeFilterNoccur(max_val=10),
        >>>                           inplace=True)

    """

    def __init__(
        self,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ):
        """Initialize EntityRangeFilterNoccur object."""
        super().__init__(min_val, max_val)

    @staticmethod
    def get_val_getter(lsc):
        return lsc.get_noccur

    @staticmethod
    def get_ids(lsc):
        return lsc.entity.lst

    @staticmethod
    def get_mod_fun(lsc):
        return lsc.pop_entity


class LabelsetRangeFilterSize(BaseRangeFilter):
    """Filter labelsets based on size.

    Example:
        The following example removes any labelset that has more less than 10
        or more than 100 number of positives.

        >>> labelset_collection.apply(
        >>>     LabelsetRangeFilterSize(min_val=10, max_val=100), inplace=True)

    """

    def __init__(
        self,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ):
        """Initialize LabelsetRangeFilterSize object."""
        super().__init__(min_val, max_val)

    @staticmethod
    def get_val_getter(lsc):
        return lambda entity_id: len(lsc.get_labelset(entity_id))

    @staticmethod
    def get_ids(lsc):
        return lsc.label_ids

    @staticmethod
    def get_mod_fun(lsc):
        return lsc.pop_labelset


class LabelsetRangeFilterJaccard(BaseRangeFilter):
    """Filter labelsets based on Jaccard index.

    For each labelset, compare against all other labelsets and record the
    largest Jaccard index computed in the case that the current labelset has
    more positives. If the largest recorded jaccard index is larger than the
    specified ``max_val``, then the current labelset is removed.

    Note:
        Only support filtering with ``max_val``.

    Example:
        >>> labelset_collection.iapply(LabelsetRangeFilterJaccard(max_val=0.7))

    """

    def __init__(self, max_val: float):
        """Initialize LabelsetRangeFilterJaccard object."""
        super().__init__(None, max_val)

    @staticmethod
    def get_val_getter(lsc):
        # if jaccard index greater than threshold, determin whether or not to
        # discard depending on the size, only discard the larger one
        def val_getter(label_id):
            val = 0
            labelset = lsc.get_labelset(label_id)
            for label_id2 in lsc.label_ids:
                if label_id2 == label_id:  # skip self
                    continue
                labelset2 = lsc.get_labelset(label_id2)
                if len(labelset) >= len(labelset2):
                    unionsize = len(labelset | labelset2)
                    jidx = len(labelset & labelset2) / unionsize
                    val = max(val, jidx)
            return val

        return val_getter

    @staticmethod
    def get_ids(lsc):
        return lsc.label_ids

    @staticmethod
    def get_mod_fun(lsc):
        return lsc.pop_labelset


class LabelsetRangeFilterOverlap(BaseRangeFilter):
    """Filter labelsets based on the overlap coefficient.

    Similar to LabelsetRangeFilterJaccard, but using Overlap coefficient, which
    is computed as the size of the intersection over the minimum size of the
    two sets.

    Note:
        Only support filtering with ``max_val``.

    Example:
        >>> labelset_collection.iapply(LabelsetRangeFilterOverlap(max_val=0.7))

    """

    def __init__(self, max_val: float):
        """Initialize LabelsetRangeFilterJaccard object."""
        super().__init__(None, max_val)

    @staticmethod
    def get_val_getter(lsc):
        def val_getter(label_id):
            val = 0
            labelset = lsc.get_labelset(label_id)
            for label_id2 in lsc.label_ids:
                if label_id2 == label_id:  # skip self
                    continue
                labelset2 = lsc.get_labelset(label_id2)
                if len(labelset) >= len(labelset2):
                    ovlpt = len(labelset & labelset2) / len(labelset2)
                    val = max(val, ovlpt)
            return val

        return val_getter

    @staticmethod
    def get_ids(lsc):
        return lsc.label_ids

    @staticmethod
    def get_mod_fun(lsc):
        return lsc.pop_labelset


class LabelsetRangeFilterSplit(BaseRangeFilter):
    """Filter labelsets based on number of positives in each dataset split."""

    def __init__(
        self,
        min_val: float,
        splitter: Splitter,
        verbose: bool = False,
        **kwargs,
    ):
        """Initialize LabelsetRangeFilterTrainTestPos object.

        Args:
            verbose (bool): If set to True, print the relevant information at
                the end of each iteration.

        """
        super().__init__(min_val=min_val)
        self.splitter = splitter
        self.kwargs = kwargs
        self.verbose = verbose

    def get_val_getter(self, lsc):
        """Return the value getter.

        The value getter finds the minimum number of positives for a labelset
        across all the dataset splits.

        """

        def val_getter(label_id):
            y_all, masks = lsc.split(self.splitter, **self.kwargs)
            y = y_all[:, lsc.label_ids.index(label_id)]
            min_num_pos = np.inf
            for mask in masks.values():
                for i in range(mask.shape[1]):
                    num_pos = y[mask[:, i]].sum()
                    min_num_pos = min(min_num_pos, num_pos)
            if self.verbose:
                print(f"{label_id}, {min_num_pos=}")
            return min_num_pos

        return val_getter

    @staticmethod
    def get_ids(lsc):
        return lsc.label_ids

    @staticmethod
    def get_mod_fun(lsc):
        return lsc.pop_labelset
