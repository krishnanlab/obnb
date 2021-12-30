from NLEval.valsplit.Base import BaseInterface
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

__all__ = ["SklSKF", "SklSSS", "SklLOO", "SklLPO"]


def pass_kwargs(func, kwargs):
    """Check if kwargs is None and do not pass if so."""
    if kwargs is None:
        return func()
    else:
        return func(**kwargs)


class SklInterface(BaseInterface):
    """Dedicated interface for Scikit Learn splitter."""

    def __init__(self, skl_splitter, skl_kws=None, shuffle=False):
        """Initialize SklInterface object."""
        super().__init__(shuffle=shuffle)
        splitter = pass_kwargs(skl_splitter, skl_kws)
        self.setup_split_func(splitter.split)


class SklSKF(SklInterface):
    """Dedicated interface for Stratified K-Fold in SKLearn."""

    def __init__(self, skl_kws=None, shuffle=False):
        """Initialize SkeSKF object."""
        super().__init__(StratifiedKFold, skl_kws, shuffle)


class SklSSS(SklInterface):
    """Dedicated interface for Stratified Shuffle Split in SKLearn."""

    def __init__(self, skl_kws=None, shuffle=False):
        """Initialize SklSSS object."""
        super().__init__(StratifiedShuffleSplit, skl_kws, shuffle)


class SklLOO(SklInterface):
    """Dedicated interface for Leave One Out in SKLearn."""

    def __init__(self, skl_kws=None, shuffle=False):
        """Initialize SklLOO object."""
        super().__init__(LeavePOut, skl_kws, shuffle)


class SklLPO(SklInterface):
    """Dedicated interface for Leave P Out in SKLearn."""

    def __init__(self, skl_kws=None, shuffle=False):
        """Initialize SklLPO object."""
        super().__init__(LeaveOneOut, skl_kws, shuffle)
