from collections import Counter
from numbers import Real
from typing import Iterator
from typing import List
from typing import Tuple

import numpy as np

from ...util.checkers import checkTypesInIterableErrEmpty
from .base import BaseSortedSplit


class BasePartition(BaseSortedSplit):
    """BasePartition object for splitting by partitioning the dataset."""

    def __call__(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Iterator[Tuple[np.ndarray, ...]]:
        """Partition the dataset.

        First sort the entity based on their 1-dimensional properties (x),
        then find the list of index used to split the dataset based on the
        sorted entities. Finally, yield the splits.

        Note:
            The use of yield instead of return is to make it compatible with
            the sklearn split methods.

        """
        x_sorted_idx, x_sorted_val = self.sort(x)
        idx = self.get_split_idx(x_sorted_val)
        yield self.split_by_idx(idx, x_sorted_idx)

    @staticmethod
    def split_by_idx(
        idx: List[int],
        x_sorted_idx: np.ndarray,
    ) -> Tuple[np.ndarray, ...]:
        """Return the splits given the split index.

        Args:
            idx: Index indicating to split intervals the sorted entities.
            x_sorted_idx: Sorted index of the entities (data points) in the
                dataset.

        """
        slices = [slice(idx[i], idx[i + 1]) for i in range(len(idx) - 1)]
        splits = (*(x_sorted_idx[i] for i in slices),)
        return splits

    def get_split_idx(self, x_sorted_val):
        raise NotImplementedError


class RatioPartition(BasePartition):
    """Split the dataset into parts of size proportional to some ratio.

    First sort the dataset entities (data points) based on a 1-dimensional
    entity property parsed in as ``x``, either ascendingly or descendingly.
    Then split the dataset based on the defined ratios.

    """

    def __init__(self, *ratios: Real, ascending: bool = True) -> None:
        """Initialize the RatioPartition object.

        Ags:
            ratios: Ratio of each split.

        """
        super().__init__(ascending)
        self.ratios = ratios

    @property
    def ratios(self) -> Tuple[Real, ...]:
        """Ratio of each split."""
        return self._ratios

    @ratios.setter
    def ratios(self, vals: Tuple[Real, ...]) -> None:
        """Setter for ratios.

        Raises:
            ValueError: If no ratio value is specified, or the ratios are not
                strictly positive, or the ratios do not add up to 1.

        """
        checkTypesInIterableErrEmpty("ratios", Real, vals)
        if min(vals) <= 0:
            raise ValueError(f"Ratios must be strictly positive: got {vals}")
        if sum(vals) != 1:
            raise ValueError(
                f"Ratios must sum up to 1, specified ratios {vals} sum up to "
                f"{sum(vals)} instead",
            )
        self._ratios = vals

    def get_split_idx(self, x_sorted_val: np.ndarray) -> List[int]:
        """Return the split index based on the split ratios."""
        x_size = x_sorted_val.size
        ratio_cum_sum = np.cumsum((0,) + self.ratios)
        return [np.floor(x_size * r).astype(int) for r in ratio_cum_sum]


class ThresholdPartition(BasePartition):
    """Split the dataset according to some threshold values.

    First sort the dataset entities (data points) based on a 1-dimensional
    entity property parsed in as ``x``, either ascendingly or descendingly.
    When sorted ascendingly, the first partition would be entities that have
    properties with values up to but not including the first (smallest)
    threshold value, and the second partition would be the entities that have
    properties with values starting (inclusive) from the first threshold value
    up to the second threshold value (not inclusive).

    Example:
        Suppose we have some dataset with properties x, then given the
        specified thresholds, we would split the dataset as follows

        >>> x = [0, 1, 1, 1, 2, 3, 4]
        >>> thresholds = (1, 3)
        >>>
        >>> split1 = [0]
        >>> split2 = [1, 2, 3, 4]
        >>> split3 = [5, 6]

    """

    def __init__(self, *thresholds: Real, ascending: bool = True) -> None:
        """Initialize the ThresholdPartition object.

        Args:
            thresholds: Thresholds used to determine the splits.

        """
        super().__init__(ascending)
        self.thresholds = thresholds

    @property
    def thresholds(self) -> Tuple[Real, ...]:
        """Thresholds for splitting."""
        return self._thresholds

    @thresholds.setter
    def thresholds(self, vals: Tuple[Real]) -> None:
        """Setter for thresholds.

        Raises:
            ValueError: If there are any duplicated threshold values, or
                not threshold value is being specified.

        """
        checkTypesInIterableErrEmpty("thresholds", Real, vals)
        for item, count in Counter(vals).items():
            if count > 1:
                raise ValueError(
                    f"Cannot have duplicated thresholds: {item} occured "
                    f"{count} times from the input {vals}",
                )
        self._thresholds = (*sorted(vals, reverse=not self.ascending),)

    def get_split_idx(self, x_sorted_val: np.ndarray) -> List[int]:
        """Return the split index based on the cut thresholds."""
        x_size = x_sorted_val.size
        idx = [0] * (len(self.thresholds) + 2)
        idx[-1] = x_size
        for i, threshold in enumerate(self.thresholds):
            threshold = threshold if self.ascending else -threshold
            where = np.where(x_sorted_val >= threshold)[0]
            idx[i + 1] = x_size if where.size == 0 else where[0]
        return idx
