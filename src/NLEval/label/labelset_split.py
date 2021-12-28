from collections import Counter
from numbers import Real
from typing import Iterator
from typing import Tuple

import numpy as np
from NLEval.util.checkers import checkTypeErrNone
from NLEval.util.checkers import checkTypesInIterable


class BaseSplit:
    def __repr__(self):
        """Representation of the labelset split object."""
        name = self.__class__.__name__
        attrs = [f"{i.lstrip('_')}={j!r}" for i, j in self.__dict__.items()]
        attrstr = ", ".join(attrs)
        return f"{name}({attrstr})"


class BaseHoldout(BaseSplit):
    def __init__(self, ascending: bool = True):
        self.ascending = ascending

    def __call__(self, x, y):
        x_sorted_idx, x_sorted_val = self.sort(x)
        idx = self.get_split_idx(x_sorted_val)
        yield self.split_by_idx(idx, x_sorted_idx)

    @property
    def ascending(self):
        return self._ascending

    @ascending.setter
    def ascending(self, val):
        checkTypeErrNone("ascending", bool, val)
        self._ascending = val

    def sort(self, x):
        x_val = x if self.ascending else -x
        x_sorted_idx = x_val.argsort()
        x_sorted_val = x_val[x_sorted_idx]
        return x_sorted_idx, x_sorted_val

    @staticmethod
    def split_by_idx(idx, x_sorted_idx):
        slices = [slice(idx[i], idx[i + 1]) for i in range(len(idx) - 1)]
        splits = (*(x_sorted_idx[i] for i in slices),)
        return splits


class RatioHoldout(BaseHoldout):
    def __init__(self, *ratios: float, ascending: bool = True):
        super().__init__(ascending)
        self.ratios = ratios

    @property
    def ratios(self):
        return self._ratios

    @ratios.setter
    def ratios(self, vals):
        if not vals:
            raise ValueError("No ratios specified")
        checkTypesInIterable("ratios", Real, vals)
        if min(vals) <= 0:
            raise ValueError(f"Ratios must be strictly positive: got {vals}")
        if sum(vals) != 1:
            raise ValueError(
                f"Ratios must sum up to 1, specified ratios {vals} sum up to "
                f"{sum(vals)} instead",
            )
        self._ratios = vals

    def get_split_idx(self, x_sorted_val):
        x_size = x_sorted_val.size
        ratio_cum_sum = np.cumsum((0,) + self.ratios)
        return [np.floor(x_size * r).astype(int) for r in ratio_cum_sum]


class ThresholdHoldout(BaseHoldout):
    def __init__(self, *thresholds: float, ascending: bool = True):
        super().__init__(ascending)
        self.thresholds = thresholds

    @property
    def thresholds(self):
        return self._thresholds

    @thresholds.setter
    def thresholds(self, vals: Tuple[float]):
        if not vals:
            raise ValueError("No thresholds specified")
        checkTypesInIterable("thresholds", Real, vals)
        for item, count in Counter(vals).items():
            if count > 1:
                raise ValueError(
                    f"Cannot have duplicated thresholds: {item} occured "
                    f"{count} times from the input {vals}",
                )
        self._thresholds = (*sorted(vals, reverse=not self.ascending),)

    def get_split_idx(self, x_sorted_val):
        cut_idx = [None] * len(self.thresholds)
        x_size = x_sorted_val.size
        for i, threshold in enumerate(self.thresholds):
            threshold = threshold if self.ascending else -threshold
            where = np.where(x_sorted_val >= threshold)[0]
            cut_idx[i] = x_size if where.size == 0 else where[0]
        return [0] + cut_idx + [x_size]
