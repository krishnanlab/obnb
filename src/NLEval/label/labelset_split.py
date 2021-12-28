from typing import Iterator
from typing import Tuple

import numpy as np


class BaseSplit:
    def __repr__(self):
        """Representation of the labelset split object."""
        name = self.__class__.__name__
        attrstr = ", ".join([f"{i}={j!r}" for i, j in self.__dict__.items()])
        return f"{name}({attrstr})"


class BaseHoldout(BaseSplit):
    def __init__(self, ascending: bool = True):
        self.ascending = ascending  # TODO: check type

    def sort(self, x):
        x_sorted_idx = x.argsort() if self.ascending else (-x).argsort()
        x_sorted_val = x[x_sorted_idx]
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
        # TODO: need to sort accordingly ascending
        self.thresholds = thresholds

    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        x_sorted_idx, x_sorted_val = self.sort(x)
        cut_idx = [(x_sorted_val >= t).argmax() for t in self.thresholds]

        yield self.split_by_cut_idx(cut_idx, x_sorted_idx)
