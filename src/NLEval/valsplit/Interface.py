from NLEval.valsplit.Base import *
from sklearn.model_selection import (
    LeaveOneOut,
    LeavePOut,
    StratifiedKFold,
    StratifiedShuffleSplit,
)

__all__ = ["SklSKF", "SklSSS", "SklLOO", "SklLPO"]


def pass_kwargs(func, kwargs):
    """Check if kwargs is None and do not pass if so."""
    if kwargs is None:
        return func()
    else:
        return func(**kwargs)


class SklInterface(BaseInterface):
    """Dedicated interface for Scikit Learn splitter."""

    def __init__(self, SklSplitter, skl_kws=None, shuffle=False):
        super(SklInterface, self).__init__(shuffle=shuffle)
        splitter = pass_kwargs(SklSplitter, skl_kws)
        self.setup_split_func(splitter.split)


class SklSKF(SklInterface):
    """Dedicated interface for Stratified K-Fold in SKLearn."""

    def __init__(self, skl_kws=None, shuffle=False):
        super(SklSKF, self).__init__(StratifiedKFold, skl_kws, shuffle)


class SklSSS(SklInterface):
    """Dedicated interface for Stratified Shuffle Split in SKLearn."""

    def __init__(self, skl_kws=None, shuffle=False):
        super(SklSSS, self).__init__(
            StratifiedShuffleSplit,
            skl_kws,
            shuffle,
        )


class SklLOO(SklInterface):
    """Dedicated interface for Leave One Out in SKLearn."""

    def __init__(self, skl_kws=None, shuffle=False):
        super(SklLOO, self).__init__(LeavePOut, skl_kws, shuffle)


class SklLPO(SklInterface):
    """Dedicated interface for Leave P Out in SKLearn."""

    def __init__(self, skl_kws=None, shuffle=False):
        super(SklLPO, self).__init__(LeaveOneOut, skl_kws, shuffle)
