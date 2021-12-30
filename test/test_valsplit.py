import unittest

import numpy as np
from NLEval.valsplit.Holdout import BinHold
from NLEval.valsplit.Holdout import CustomHold
from NLEval.valsplit.Holdout import ThreshHold
from NLEval.valsplit.Holdout import TrainValTest


class TestRepr(unittest.TestCase):
    def test_train_val_test(self):
        train_val_test = TrainValTest(train_ratio=0.6, test_ratio=0.2)
        self.assertEqual(
            repr(train_val_test),
            "TrainValTest(train_ratio=0.6, test_ratio=0.2, train_on='top', "
            "shuffle=False)",
        )

    def test_bind_hold(self):
        bin_hold = BinHold(3)
        self.assertEqual(
            repr(bin_hold),
            "BinHold(bin_num=3, train_on='top', shuffle=False)",
        )

    def test_thresh_hold(self):
        thresh_hold = ThreshHold(0.5)
        self.assertEqual(
            repr(thresh_hold),
            "ThreshHold(cut_off=0.5, train_on='top', shuffle=False)",
        )

    def test_custom_hold(self):
        custom_hold = CustomHold(np.array([]), np.array([]))
        self.assertEqual(repr(custom_hold), "CustomHold()")


if __name__ == "__main__":
    unittest.main()
