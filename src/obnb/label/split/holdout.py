import numpy as np

from obnb.label.split.base import BaseRandomSplit, BaseSortedSplit
from obnb.alltypes import Tuple
from obnb.util.checkers import checkType


class BaseHoldout(BaseSortedSplit):
    """BaseHoldout object for holding out some portion of the dataset."""

    @staticmethod
    def split_by_idx(
        idx: int,
        x_sorted_idx: np.ndarray,
    ) -> Tuple[np.ndarray, ...]:
        """Return the splits given the split index.

        Args:
            idx: Index indicating to split intervals of the sorted entities.
            x_sorted_idx: Sorted index of the entities (data points) in the
                dataset.

        """
        return (x_sorted_idx[0:idx],)


class RatioHoldout(BaseHoldout):
    """Holdout a portion of the dataset.

    First sort the dataset entities (data points) based on a 1-dimensional
    entity property parsed in as ``x``, either ascendingly or descendingly. Then
    take the top datapoints with portion defined by the ratio input.

    """

    def __init__(
        self,
        ratio: float,
        *,
        property_converter,
        ascending: bool = True,
    ) -> None:
        """Initialize the RatioHoldout object.

        Ags:
            ratio: Ratio of holdout.

        """
        super().__init__(property_converter=property_converter, ascending=ascending)
        self.ratio = ratio

    @property
    def ratio(self) -> float:
        """Ratio of each split."""
        return self._ratio

    @ratio.setter
    def ratio(self, ratio) -> None:
        """Setter for ratio.

        Riases:
            TypeError: If the input value is not float type.
            ValueError: If the input value  is not between 0 (not including
                zero) and 1 (including 1).

        """
        checkType("ratio", float, ratio)
        if not 0 < ratio <= 1:
            raise ValueError(
                f"ratio must be between 0 (exclusive) and 1 (inclusive), "
                f"got {ratio}",
            )
        self._ratio = ratio

    def get_split_idx(self, x_sorted_val: np.ndarray) -> int:
        """Return the split index based on the split ratio."""
        x_size = x_sorted_val.size
        idx = np.floor(x_size * self.ratio).astype(int)
        return idx


class ThresholdHoldout(BaseHoldout):
    """Split the dataset according to some threshold values.

    First sort the dataset entities (data points) based on a 1-dimensional
    entity property parsed in as ``x``, either ascendingly or descendingly.
    When sorted ascendingly, the holdout split would be entities that have
    properties with values up to but not including the first (smallest)
    threshold value.

    Example:
        Suppose we have some dataset with properties x, then given the
        specified threshold, we would split the dataset as follows

        >>> x = [0, 1, 1, 1, 2, 3, 4]
        >>> threshold = 2
        >>>
        >>> holdout = [0, 1, 1, 1]

    """

    def __init__(
        self,
        threshold: float,
        *,
        property_converter,
        ascending: bool = True,
    ) -> None:
        """Initialize the ThresholdHoldout object.

        Args:
            threshold: Threshold used to determine the splits.

        """
        super().__init__(property_converter=property_converter, ascending=ascending)
        self.threshold = threshold

    @property
    def threshold(self) -> float:
        """Threshold for splitting."""
        return self._threshold

    @threshold.setter
    def threshold(self, threshold: float) -> None:
        """Setter for threshold.

        Raises:
            TypeError: If the input value not float type.

        """
        checkType("threshold", (int, float), threshold)
        self._threshold = threshold

    def get_split_idx(self, x_sorted_val: np.ndarray) -> int:
        """Return the split index based on the cut threshold."""
        x_size = x_sorted_val.size
        where = (
            np.where(x_sorted_val >= self.threshold)[0]
            if self.ascending
            else np.where(x_sorted_val <= self.threshold)[0]
        )
        idx = x_size if where.size == 0 else where[0]
        return idx


class RandomRatioHoldout(BaseRandomSplit, RatioHoldout):
    """Randomly holdout some ratio of the dataset."""

    def __init__(self, ratio, *, shuffle=True, random_state=None):
        """Initialize RandomRatioHoldout."""
        super().__init__(ratio, shuffle=shuffle, random_state=random_state)


class AllHoldout(RandomRatioHoldout):
    """Holdout all available data points."""

    def __init__(self, *, shuffle=False, random_state=None):
        """Initialize the AllHoldout object."""
        super().__init__(1.0, shuffle=shuffle, random_state=random_state)
