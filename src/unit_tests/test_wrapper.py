import os
import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from NLEval.wrapper.ParWrap import ParDat


class TestParDat(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n_samples = 1000
        cls.n_features = 20
        cls.n_classes = 100
        cls.job_list = list(range(cls.n_classes))

    def setUp(self):
        n_samples = self.n_samples
        n_features = self.n_features
        n_classes = self.n_classes
        self.X = np.random.random((n_samples, n_features))
        self.Y = np.random.randint(2, size=(n_samples, n_classes))
        self.mdl = LogisticRegression()

    @staticmethod
    def train_mdl(i, X, Y, mdl):
        mdl.fit(X, Y[:, i])

    def parallel_logreg_training(self, n_workers, verbose):
        wrapper = ParDat(self.job_list, n_workers=n_workers, verbose=verbose)
        train_mdl_parallel = wrapper(self.train_mdl)

        for _ in train_mdl_parallel(self.X, self.Y, self.mdl):
            pass

    def test_parallel_logreg_training(self):
        params = [
            (1, False),
            (4, True),
            (1, False),
            (4, True),
        ]

        for param in params:
            n_workers, verbose = param
            with self.subTest(n_workers=n_workers, verbose=verbose):
                self.parallel_logreg_training(n_workers, verbose)


if __name__ == "__main__":
    unittest.main()
