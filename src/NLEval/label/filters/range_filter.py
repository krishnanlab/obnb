from typing import Optional

import numpy as np

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
        >>> labelset_collection.apply(LabelsetRangeFilterSize(max_val=0.7),
        >>>                           inplace=True)

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
                jidx = len(labelset & labelset2) / len(labelset | labelset2)
                if (jidx > val) & (len(labelset) >= len(labelset2)):
                    val = jidx
            return val

        return val_getter

    @staticmethod
    def get_ids(lsc):
        return lsc.label_ids

    @staticmethod
    def get_mod_fun(lsc):
        return lsc.pop_labelset


class LabelsetRangeFilterTrainTestPos(BaseRangeFilter):
    """Filter labelsets based on number of positives in train/test sets.

    Note:
        Only intended to be work with Holdout split type for now. Would not
            raise error for other split types, but only will check the first
            split. If validation set is available, will also check the
            validation split.

    """

    def __init__(self, min_val: float):
        """Initialize LabelsetRangeFilterTrainTestPos object."""
        super().__init__(min_val=min_val)

    @staticmethod
    def get_val_getter(lsc):
        return lambda label_id: min(
            idx_ary.size
            for idx_ary in next(
                lsc.valsplit.get_split_idx_ary(
                    np.array(list(lsc.get_labelset(label_id))),
                    valid=lsc.valsplit.valid_index is not None,
                ),
            )
        )

    @staticmethod
    def get_ids(lsc):
        return lsc.label_ids

    @staticmethod
    def get_mod_fun(lsc):
        return lsc.pop_labelset  # replace with soft filter
