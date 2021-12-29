from collections import Counter
from numbers import Real
from typing import Iterator
from typing import List
from typing import Tuple

import numpy as np
from NLEval.util.checkers import checkTypeErrNone
from NLEval.util.checkers import checkTypesInIterableErrEmpty


class BaseSplit:
    """BaseSplit object for splitting dataset.

    This is an abstract class for objects used for splitting the dataset
    based on either the labels y and / or some properties of each entity,
    passed in as an 1-dimensional array x. This abstract class only defines a
    __repr__ magic method used for printing.

    """

    def __repr__(self) -> str:
        """Representation of the labelset split object."""
        name = self.__class__.__name__
        attrs = [f"{i.lstrip('_')}={j!r}" for i, j in self.__dict__.items()]
        attrstr = ", ".join(attrs)
        return f"{name}({attrstr})"


class BaseHoldout(BaseSplit):
    """BaseHoldout object for splitting via holdout."""

    def __init__(self, ascending: bool = True) -> None:
        """Initialize BaseHoldout object.

        Args:
            ascending: Sort the entities in the dataset ascendingly based on
                a property, parsed in a x. Consequently, entities with smaller
                valued properties are used for training and etc. If set to
                False, on the other hand, then sort descendingly.

        """
        self.ascending = ascending

    def __call__(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Iterator[Tuple[np.ndarray, ...]]:
        """Split the dataset.

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

    @property
    def ascending(self) -> bool:
        """Sort entities in the dataset ascendingly if set to True."""
        return self._ascending

    @ascending.setter
    def ascending(self, val: bool) -> None:
        """Setter for ascending.

        Raises:
            TypeError: If the input value of ascending is no bool type.
            ValueError: If the input value of ascending is None.

        """
        checkTypeErrNone("ascending", bool, val)
        self._ascending = val

    def sort(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return the sorted index and value of the entity properties.

        Note:
            Return the negative sorted values if ``self.ascending`` is set to
            False, which is effectively the same as sorting them descendingly.

        Args:
            x: properties of the entities as an 1-dimensional array.

        """
        x_val = x if self.ascending else -x
        x_sorted_idx = x_val.argsort()
        x_sorted_val = x_val[x_sorted_idx]
        return x_sorted_idx, x_sorted_val

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


class RatioHoldout(BaseHoldout):
    """Split the dataset into parts of size proportional to some ratio.

    First sort the dataset entities (data points) based on a 1-dimensional
    entity property parsed in as ``x``, either ascendingly or descendingly.
    Then split the dataset based on the defined ratios.

    """

    def __init__(self, *ratios: Real, ascending: bool = True) -> None:
        """Initialize the RatioHoldout object.

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


class ThresholdHoldout(BaseHoldout):
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
        """Initialize the ThresholdHoldout object.

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
