import numpy as np
from NLEval.util import IDHandler, checkers

__all__ = ["BaseValSplit", "BaseHoldout", "BaseInterface"]


class BaseValSplit:
    def __init__(self, shuffle=False):
        """Initialize BaseValSplit object."""
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

    def split(self, id_ary, label_ary, valid=False):
        """Split labelset into training, testing (and validation) sets.

        Given matching arrays of node IDs and label (currently only support
        binary labels), this function yields the IDs and label for training,
        testing (and validation) splits, by calling ``get_split_idx_ary``.

        Note:
            id_ary and label_ary are coulpled, such that a particular entry
            in label_ary corresponds to the label of the ID in the same entry
            in id_ary.

        Args:
            id_ary(:obj:`numpy.ndarray` of :obj:`str`): array of entity IDs
            label_ary(:obj:`numpy.ndarray` of :obj:`bool`):
                boolean/binary array of indicating positive samples
            valid(bool): whether or not to generate validation split, default
                is False

        Yields:
            train_index(:obj:`numpy.ndarray` of :obj:`str`):
                numpy array of training IDs
            train_label_ary(:obj:`numpy.ndarray` of :obj:`bool`):
                numpy array of training labels
            test_index(:obj:`numpy.ndarray` of :obj:`str`):
                numpy array of testing IDs
            test_label_ary(:obj:`numpy.ndarray` of :obj:`bool`):
                numpy array of testing labels
            valid_index(:obj:`numpy.ndarray` of :obj:`str`):
                for `valid=True`, numpy array of validation IDs
            valid_label_ary(:obj:`numpy.ndarray` of :obj:`bool`):
                for `valid=True`, numpy array of validation labels

        """
        # train, test (and validation) index arrays
        for idx_arys in self.get_split_idx_ary(id_ary, label_ary, valid):
            out = ()
            for idx_ary in idx_arys:
                if self.shuffle:
                    np.random.shuffle(idx_ary)
                out += (id_ary[idx_ary], label_ary[idx_ary])
            yield out


class BaseHoldout(BaseValSplit):
    def __init__(self, train_on="top", shuffle=False):
        """Initialize BaseHoldout object.

        Split based on some numeric properties of the samples, train on
        either top or bottom set and test on the other, depending on
        user specification of `train_on`. If 'top' specified, those samples
        with properties of larger values are used for training, and those
        with smaller values are used for testing, and vice versa. The
        ``train_index`` and ``test_index`` will be constructed by more
        specific hold-out class.

        """
        super(BaseHoldout, self).__init__(shuffle=shuffle)
        self.train_on = train_on
        self._test_index = None
        self._valid_index = None
        self._train_index = None

    @property
    def train_index(self):
        return None if self._train_index is None else self._train_index.copy()

    @property
    def valid_index(self):
        return None if self._valid_index is None else self._valid_index.copy()

    @property
    def test_index(self):
        return None if self._test_index is None else self._test_index.copy()

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

    def get_common_ids(self, lsc_ids, node_ids):
        """Get list of common IDs between labelset collection and graph.

        Note:
            Only included IDs that are part of at least one labelset

        Args:
            lsc_ids(:obj:`NLEval.util.IDHandler.IDprop`): ID list of labelset
                collection
            node_ids(:obj:`NLEval.util.IDHandler.IDmap`): ID list of graph

        """
        checkers.checkType(
            "ID for labelset collection entities",
            IDHandler.IDprop,
            lsc_ids,
        )
        checkers.checkType("ID for graph entities", IDHandler.IDmap, node_ids)
        common_ids = []
        for node_id in node_ids.lst:
            if node_id in lsc_ids:
                # make sure entity is part of at least one labelset
                if lsc_ids.getProp(node_id, "Noccur") > 0:
                    common_ids.append(node_id)
        return common_ids

    def check_split_setup(self, valid):
        if self._test_index is None or self._train_index is None:
            raise ValueError(
                "Training or testing sets not available, "
                "run `train_test_setup` first",
            )

        if valid:
            if self.valid_index is None:
                raise ValueError(
                    f"Validation set is only available for TrainValTest "
                    f"split type, current split type is {type(self)!r}",
                )

    def get_split_idx_ary(self, id_ary, label_ary=None, valid=False):
        self.check_split_setup(valid)
        train_idx_ary = np.where(np.in1d(id_ary, self.train_index))[0]
        test_idx_ary = np.where(np.in1d(id_ary, self.test_index))[0]
        if valid:
            valid_idx_ary = np.where(np.in1d(id_ary, self.valid_index))[0]
            yield train_idx_ary, valid_idx_ary, test_idx_ary
        else:
            yield train_idx_ary, test_idx_ary


class BaseInterface(BaseValSplit):
    """Base interface with user defined validation split generator."""

    def __init__(self, shuffle=False):
        """Initialize BaseInterface object."""
        super(BaseInterface, self).__init__(shuffle=shuffle)

    def setup_split_func(self, split_func):
        self.get_split_idx_ary = lambda id_ary, label_ary: split_func(
            id_ary,
            label_ary,
        )
