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
    def split_by_cut_idx(cut_idx, x_sorted_idx):
        idx = [0] + cut_idx + [len(x_sorted_idx)]
        slices = [slice(idx[i], idx[i + 1]) for i in range(len(idx) - 1)]
        splits = (*(x_sorted_idx[i] for i in slices),)
        return splits


class RatioHoldout(BaseHoldout):
    def __init__(self, *ratios: float, ascending: bool = True):
        super().__init__(ascending)
        # TODO: check if add up to 1, use -1 for completion
        self.ratios = ratios

    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        x_sorted_idx = self.sort(x)[0]
        cut_idx = [np.floor(x.shape[0] * r).astype(int) for r in self.ratios]

        yield self.split_by_cut_idx(cut_idx, x_sorted_idx)


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
            raise ValueError(f"No thresholds specified")
        checkTypesInIterable("thresholds", Real, vals)
        for item, count in Counter(vals).items():
            if count > 1:
                raise ValueError(
                    f"Cannot have duplicated thresholds: {item} occured "
                    f"{count} times from the input {vals}",
                )
        self._thresholds = (*sorted(vals, reverse=not self.ascending),)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        x_sorted_idx, x_sorted_val = self.sort(x)
        cut_idx = [None] * len(self.thresholds)
        for i, threshold in enumerate(self.thresholds):
            threshold = threshold if self.ascending else -threshold
            where = np.where(x_sorted_val >= threshold)[0]
            cut_idx[i] = len(x) if where.size == 0 else where[0]

        yield self.split_by_cut_idx(cut_idx, x_sorted_idx)
