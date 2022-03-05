import logging
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
        **kwargs,
    ) -> None:
        """Initialize BaseRangeFilter object.

        Args:
            min_val: minimum below which entities are removed
            max_val: maximum beyound which entiteis are removed

        """
        super().__init__(**kwargs)
        self.min_val = min_val
        self.max_val = max_val

    def __repr__(self):
        """Return name of the RangeFilter and its parameters."""
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
        **kwargs,
    ):
        """Initialize EntityRangeFilterNoccur object."""
        super().__init__(min_val, max_val, **kwargs)

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
        **kwargs,
    ):
        """Initialize LabelsetRangeFilterSize object."""
        super().__init__(min_val, max_val, **kwargs)

    @staticmethod
    def get_val_getter(lsc):
        return lambda entity_id: len(lsc.get_labelset(entity_id))

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
        **kwargs,
    ):
        """Initialize LabelsetRangeFilterTrainTestPos object."""
        super().__init__(min_val=min_val, **kwargs)
        self.splitter = splitter
        self.kwargs = kwargs

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
            self.logger.info(f"{label_id}, {min_num_pos=}")
            return min_num_pos

        return val_getter

    @staticmethod
    def get_ids(lsc):
        return lsc.label_ids

    @staticmethod
    def get_mod_fun(lsc):
        return lsc.pop_labelset
