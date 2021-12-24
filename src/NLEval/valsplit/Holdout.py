import numpy as np
from NLEval.util import checkers
from NLEval.valsplit.Base import BaseHoldout

__all__ = [
    "TrainValTest",
    "BinHold",
    "ThreshHold",
    "CustomHold",
    "TrainTestAll",
]


class TrainValTest(BaseHoldout):
    """Split into train-val-test sets by ratios.

    Sort the entities based on the desired properties and then prepare the
    splits according to the train-val-test ratio.

    """

    def __init__(self, train_ratio, test_ratio, train_on="top", shuffle=False):
        super(TrainValTest, self).__init__(train_on=train_on, shuffle=shuffle)
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

    def __repr__(self):
        # TODO: make repr a super magic fun, automatically generate for children.
        return (
            f"TrainValTest(train_ratio={self.train_ratio!r}, "
            f"test_ratio={self.test_ratio!r}, "
            f"train_on={self.train_on!r})"
        )

    @property
    def train_ratio(self):
        return self._train_ratio

    @property
    def test_ratio(self):
        return self._test_ratio

    @train_ratio.setter
    def train_ratio(self, val):
        checkers.checkTypeErrNone("Training ratio", checkers.FLOAT_TYPE, val)
        if (val <= 0) | (val > 1):
            raise ValueError(
                f"Training ratio must be between 0 and 1, received {val}",
            )
        self._train_ratio = val

    @test_ratio.setter
    def test_ratio(self, val):
        checkers.checkTypeErrNone("Testing ratio", checkers.FLOAT_TYPE, val)
        if (val <= 0) | (val > 1):
            raise ValueError(
                f"Testing ratio must be between 0 and 1, received {val}",
            )
        if self.train_ratio + val >= 1:
            raise ValueError(
                f"Sum of training and testing ratio must be "
                f"less than 1, received train_raio = "
                f"{self.train_ratio}, and test_ratio = {val}",
            )
        self._test_ratio = val

    def train_test_setup(self, lsc_ids, node_ids, prop_name, **kwargs):
        lsc_ids._check_prop_existence(prop_name, True)
        common_ids = self.get_common_ids(lsc_ids, node_ids)
        sorted_node_ids = sorted(
            common_ids,
            reverse=self.train_on == "bot",
            key=lambda node_id: lsc_ids.getProp(node_id, prop_name),
        )

        n = len(sorted_node_ids)
        train_size = np.floor(n * self.train_ratio).astype(int)
        test_size = np.floor(n * self.test_ratio).astype(int)

        self._test_index = np.array(sorted_node_ids[:test_size])
        self._valid_index = np.array(sorted_node_ids[test_size:-train_size])
        self._train_index = np.array(sorted_node_ids[-train_size:])


class BinHold(BaseHoldout):
    def __init__(self, bin_num, train_on="top", shuffle=False):
        """Initialize bining holdout object.

        Args:
            bin_num(int): num of bins for bin_num mode (see mode)

        """
        super(BinHold, self).__init__(train_on=train_on, shuffle=shuffle)
        self.bin_num = bin_num

    def __repr__(self):
        """Representation of BinHoldout."""
        return f"BinHold(bin_num={self.bin_num!r}, train_on={self.train_on!r})"

    @property
    def bin_num(self):
        return self._bin_num

    @bin_num.setter
    def bin_num(self, val):
        checkers.checkTypeErrNone("Number of bins", checkers.INT_TYPE, val)
        if val < 1:
            raise ValueError(
                f"Number of bins must be greater than 1, received {val}",
            )
        self._bin_num = val

    def train_test_setup(self, lsc_ids, node_ids, prop_name, **kwargs):
        """Set up training and testing.

        Args:
            lsc_ids(:obj:`NLEval.util.IDHandler.IDprop`)
            node_ids(:obj:`NLEval.util.IDHandler.IDmap`)
            prop_name(str): name of property to be used for splitting

        """
        lsc_ids._check_prop_existence(prop_name, True)
        common_ids = self.get_common_ids(lsc_ids, node_ids)
        sorted_node_ids = sorted(
            common_ids,
            reverse=self.train_on == "bot",
            key=lambda node_id: lsc_ids.getProp(node_id, prop_name),
        )
        bin_size = np.floor(len(sorted_node_ids) / self.bin_num).astype(int)
        self._test_index = np.array(sorted_node_ids[:bin_size])
        self._train_index = np.array(sorted_node_ids[-bin_size:])


class ThreshHold(BaseHoldout):
    def __init__(self, cut_off, train_on="top", shuffle=False):
        """Initialize threshold holdout split object.

        Args:
            cut_off: cut-off point for the 'cut' mode, or number of bins for
                the 'bin' mode (see mode).

        """
        super(ThreshHold, self).__init__(train_on=train_on, shuffle=shuffle)
        self.cut_off = cut_off

    def __repr__(self):
        """Representation of ThreshHold."""
        return (
            f"ThreshHold(cut_off={self.cut_off!r}, train_on={self.train_on!r})"
        )

    @property
    def cut_off(self):
        return self._cut_off

    @cut_off.setter
    def cut_off(self, val):
        checkers.checkTypeErrNone("Cut off", checkers.NUMERIC_TYPE, val)
        self._cut_off = val

    def train_test_setup(self, lsc_ids, node_ids, prop_name, **kwargs):
        """Set up training and testing.

        Args:
            lsc_ids(:obj:`NLEval.util.IDHandler.IDprop`)
            node_ids(:obj:`NLEval.util.IDHandler.IDmap`)
            prop_name(str): name of property to be used for splitting

        """
        lsc_ids._check_prop_existence(prop_name, True)
        top_list = []
        bot_list = []
        for node_id in node_ids.lst:
            if node_id in lsc_ids:
                if lsc_ids.getProp(node_id, "Noccur") > 0:
                    if lsc_ids.getProp(node_id, prop_name) >= self.cut_off:
                        top_list.append(node_id)
                    else:
                        bot_list.append(node_id)

        if self.train_on == "top":
            self._train_index, self._test_index = top_list, bot_list
        else:
            self._train_index, self._test_index = bot_list, top_list


class CustomHold(BaseHoldout):
    def __init__(
        self,
        custom_train_index,
        custom_test_index,
        custom_valid_index=None,
        shuffle=False,
    ):
        """User defined training and testing samples."""
        super(CustomHold, self).__init__(shuffle=shuffle)
        self.custom_train_index = custom_train_index
        self.custom_test_index = custom_test_index
        self.custom_valid_index = custom_valid_index

    def __repr__(self):
        """Representation of CustomHold."""
        return f"CustomHold(min_pos={self.min_pos!r})"

    @property
    def custom_train_index(self):
        return self._custom_train_index

    @custom_train_index.setter
    def custom_train_index(self, id_ary):
        checkers.checkTypesInNumpyArray("Training data ID list", str, id_ary)
        self._custom_train_index = id_ary

    @property
    def custom_test_index(self):
        return self._custom_test_index

    @custom_test_index.setter
    def custom_test_index(self, id_ary):
        checkers.checkTypesInNumpyArray("Testing data ID list", str, id_ary)
        self._custom_test_index = id_ary

    @property
    def custom_valid_index(self):
        return self._custom_valid_index

    @custom_valid_index.setter
    def custom_valid_index(self, id_ary):
        if id_ary is None:
            self._custom_valid_index = None
        else:
            checkers.checkTypesInNumpyArray(
                "Validation data ID list",
                str,
                id_ary,
            )
            self._custom_valid_index = id_ary

    def train_test_setup(self, lsc_ids, node_ids, **kwargs):
        common_ids = self.get_common_ids(lsc_ids, node_ids)
        self._train_index = np.intersect1d(
            self.custom_train_index,
            common_ids,
        )
        self._test_index = np.intersect1d(
            self.custom_test_index,
            common_ids,
        )
        self._valid_index = (
            None
            if self.custom_test_index is None
            else np.intersect1d(self.custom_valid_index, common_ids)
        )


class TrainTestAll(BaseHoldout):
    def __init__(self, shuffle=False):
        """Train and test on all data."""
        super(TrainTestAll, self).__init__(shuffle=shuffle)

    def train_test_setup(self, lsc_ids, node_ids, **kwargs):
        common_ids = self.get_common_ids(lsc_ids, node_ids)
        self._train_index = self._test_index = np.array(common_ids)


'''
class HoldoutChildTemplate(BaseHoldout):
    """
    This is a template for BaseHoldout children class
    """
    def __init__(self, **args, min_pos=10, **kwargs):
        super().__init__(min_pos=min_pos)

    def __repr__(self):
        return (
            f'HoldoutChildTemplate(min_pos={self.min_pose!r}, '
            f'train_on={self.train_on!r})'
        )

    @property
    def foo(self):
        return self._foo

    @foo.setter
    def foo(self, val):
        self._foo = val

    def train_test_setup(self, lsc_ids, node_ids, prop_name, **kwargs):
        #setup train_index and test_index
'''
