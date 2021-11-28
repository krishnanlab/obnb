from NLEval.util import checkers, IDHandler
import numpy as np

__all__ = ["BaseValSplit", "BaseHoldout", "BaseInterface"]


class BaseValSplit:
    def __init__(self, shuffle=False):
        super(BaseValSplit, self).__init__()
        self.shuffle = shuffle

    @property
    def shuffle(self):
        """bool: shufle train/test indices when splitting."""
        return self._shuffle

    @shuffle.setter
    def shuffle(self, val):
        checkers.checkType("shuffle", bool, val)
        self._shuffle = val

    def split(self, ID_ary, label_ary, valid=False):
        """Split labelset into training, testing (and validation) sets.

        Given matching arrays of node IDs and label (currently only support
        binary labels), this function yields the IDs and label for training,
        testing (and validation) splits, by calling ``get_split_idx_ary``.

        Note:
            ID_ary and label_ary are coulpled, such that a particular entry
            in label_ary corresponds to the label of the ID in the same entry
            in ID_ary.

        Args:
            ID_ary(:obj:`numpy.ndarray` of :obj:`str`): array of entity IDs
            label_ary(:obj:`numpy.ndarray` of :obj:`bool`):
                boolean/binary array of indicating positive samples
            valid(bool): whether or not to generate validation split, default
                is False

        Yields:
            train_ID_ary(:obj:`numpy.ndarray` of :obj:`str`):
                numpy array of training IDs
            train_label_ary(:obj:`numpy.ndarray` of :obj:`bool`):
                numpy array of training labels
            test_ID_ary(:obj:`numpy.ndarray` of :obj:`str`):
                numpy array of testing IDs
            test_label_ary(:obj:`numpy.ndarray` of :obj:`bool`):
                numpy array of testing labels
            valid_ID_ary(:obj:`numpy.ndarray` of :obj:`str`):
                for `valid=True`, numpy array of validation IDs
            valid_label_ary(:obj:`numpy.ndarray` of :obj:`bool`):
                for `valid=True`, numpy array of validation labels

        """
        # train, test (and validation) index arrays
        for idx_arys in self.get_split_idx_ary(ID_ary, label_ary, valid):
            out = ()
            for idx_ary in idx_arys:
                if self.shuffle:
                    np.random.shuffle(idx_ary)
                out += (ID_ary[idx_ary], label_ary[idx_ary])
            yield out


class BaseHoldout(BaseValSplit):
    def __init__(self, train_on="top", shuffle=False):
        """Initialize base holdout.

        Split based on some numeric properties of the samples, train on
        either top or bottom set and test on the other, depending on
        user specification of `train_on`. If 'top' specified, those samples
        with properties of larger values are used for training, and those
        with smaller values are used for testing, and vice versa. The
        ``train_ID_ary`` and ``test_ID_ary`` will be constructed by more
        specific hold-out class.

        """
        super(BaseHoldout, self).__init__(shuffle=shuffle)
        self.train_on = train_on
        self._test_ID_ary = None
        self._valid_ID_ary = None
        self._train_ID_ary = None

    @property
    def train_ID_ary(self):
        return None if self._train_ID_ary is None else self._train_ID_ary.copy()

    @property
    def valid_ID_ary(self):
        return None if self._valid_ID_ary is None else self._valid_ID_ary.copy()

    @property
    def test_ID_ary(self):
        return None if self._test_ID_ary is None else self._test_ID_ary.copy()

    @property
    def train_on(self):
        """Train on top ('top') or bottom ('bot') sample sets."""
        return self._train_on

    @train_on.setter
    def train_on(self, val):
        checkers.checkTypeErrNone("train_on", str, val)
        if val not in ["top", "bot"]:
            raise ValueError(f"Train on must be 'top' or 'bot', not {val!r}")
        self._train_on = val

    def get_common_ID_list(self, lscIDs, nodeIDs):
        """Get list of common IDs between labelset collection and graph.

        Note:
            Only included IDs that are part of at least one labelset

        Args:
            lscIDs(:obj:`NLEval.util.IDHandler.IDprop`): ID list of labelset
                collection
            nodeIDs(:obj:`NLEval.util.IDHandler.IDmap`): ID list of graph

        """
        checkers.checkType(
            "ID for labelset collection entities", IDHandler.IDprop, lscIDs,
        )
        checkers.checkType("ID for graph entities", IDHandler.IDmap, nodeIDs)
        common_ID_list = []
        for ID in nodeIDs.lst:
            if ID in lscIDs:
                # make sure entity is part of at least one labelset
                if lscIDs.getProp(ID, "Noccur") > 0:
                    common_ID_list.append(ID)
        return common_ID_list

    def check_split_setup(self, valid):
        assert (self._test_ID_ary is not None) & (
            self._train_ID_ary is not None
        ), "Training or testing sets not available, run `train_test_setup` first"
        if valid:
            assert self._valid_ID_ary is not None, (
                f"Validation set is only available for TrainValTest split "
                f"type, current split type is {type(self)!r}"
            )

    def get_split_idx_ary(self, ID_ary, label_ary=None, valid=False):
        self.check_split_setup(valid)
        train_idx_ary = np.where(np.in1d(ID_ary, self.train_ID_ary))[0]
        test_idx_ary = np.where(np.in1d(ID_ary, self.test_ID_ary))[0]
        if valid:
            valid_idx_ary = np.where(np.in1d(ID_ary, self.valid_ID_ary))[0]
            yield train_idx_ary, valid_idx_ary, test_idx_ary
        else:
            yield train_idx_ary, test_idx_ary


class BaseInterface(BaseValSplit):
    """Base interface with user defined validation split generator."""

    def __init__(self, shuffle=False):
        super(BaseInterface, self).__init__(shuffle=shuffle)

    def setup_split_func(self, split_func):
        self.get_split_idx_ary = lambda ID_ary, label_ary: split_func(
            ID_ary, label_ary,
        )
