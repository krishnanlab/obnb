import numpy as np

from obnb.label.filters.range_filter import BaseRangeFilter
from obnb.typing import Set


class BaseLabelsetPairwiseFilter(BaseRangeFilter):
    """Filter labelsets in a collection based on pairwise scores.

    For each labelset in the labelset collection, compute its pairwise score
    against all other labelsets and remove the labelset if the pairwise score
    exceeds certain threshold. One can constraint the comparison so that, for
    example, the labelset is only compared against labelsets that are larger
    in size (by setting ``size_constraint`` = 'larger').

    """

    def __init__(
        self,
        max_val: float,
        size_constraint: str = "smaller",
        inclusive: bool = True,
        **kwargs,
    ):
        """Initialize the pairwise labelset filter.

        Args:
            max_val (float):
            size_constraint (str): If set to 'larger' (or 'smaller'), then only
                make the pairwise comparison if the current labelset if larger
                (or smaller) than the target labelset. Finally, 'none' is the
                same as setting to both 'larger' and 'smaller'
                (default; 'larger').
            inclusive (bool): Whether or not to make the comparison if the two
                labelsets have the same size (default: :obj:`True`)

        """
        super().__init__(None, max_val, **kwargs)
        self.size_constraint = size_constraint
        self.inclusive = inclusive

    @property
    def mod_name(self):
        return "DROP LABELSET"

    @staticmethod
    def get_ids(lsc):
        return lsc.label_ids

    @staticmethod
    def get_mod_fun(lsc):
        return lsc.pop_labelset

    @property
    def size_constraint(self):
        return self._size_constraint

    @size_constraint.setter
    def size_constraint(self, size_constraint):
        if size_constraint not in ["larger", "smaller", "none"]:
            raise ValueError(
                f"Unknown value for size_constraint: {size_constraint!r}, "
                "accepted values are 'larger' or 'smaller'.",
            )
        self._size_constraint = size_constraint
        self._size_gt = size_constraint in ["larger", "none"]
        self._size_lt = size_constraint in ["smaller", "none"]

    def comparable(self, labelset1: Set[str], labelset2: Set[str]) -> bool:
        len1, len2 = len(labelset1), len(labelset2)
        return (
            (self._size_gt and len1 > len2)
            or (self._size_lt and len1 < len2)
            or (self.inclusive and len1 == len2)
        )

    def compute_pairwise_score(
        self,
        labelset1: Set[str],
        labelset2: Set[str],
    ) -> float:
        raise NotImplementedError

    def get_val_getter(self, lsc):
        def val_getter(label_id):
            labelset = lsc.get_labelset(label_id)
            for label_id2 in lsc.label_ids:
                if label_id2 == label_id:  # skip self
                    continue
                labelset2 = lsc.get_labelset(label_id2)
                if self.comparable(labelset, labelset2):
                    score = self.compute_pairwise_score(labelset, labelset2)
                    # Immediately return if the score exists the threshold
                    if score > self.max_val:
                        return score
            return -np.inf

        return val_getter


class LabelsetPairwiseFilterJaccard(BaseLabelsetPairwiseFilter):
    """Filter labelsets based on Jaccard index.

    The Jaccard index is computed as the size of the intersection divided by
    the size of the union of two sets.

    Example:
        >>> labelset_collection.iapply(LabelsetPairwiseFilterJaccard(0.7))

    """

    def compute_pairwise_score(self, labelset1, labelset2):
        return len(labelset1 & labelset2) / len(labelset1 | labelset2)


class LabelsetPairwiseFilterOverlap(BaseLabelsetPairwiseFilter):
    """Filter labelsets based on the Overlap coefficient.

    The Overlap coefficient is computed as the size of the intersection divided
    by the minimum size of the two sets.

    Example:
        >>> labelset_collection.iapply(LabelsetPairwiseFilterOverlap(0.8))

    """

    def compute_pairwise_score(self, labelset1, labelset2):
        minsize = min(map(len, (labelset1, labelset2)))
        return len(labelset1 & labelset2) / minsize
