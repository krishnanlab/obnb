import os
import unittest

import numpy as np
from NLEval.wrapper.ParWrap import ParDat
from sklearn.linear_model import LogisticRegression


class TestParDatLogReg(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_samples = 1000
        cls.n_features = 20
        cls.n_classes = 10
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


class TestParDat(unittest.TestCase):
    def setUp(self):
        self.num_list = num_list = list(range(100))
        self.sqrt_list = np.sqrt(num_list).tolist()
        self.job_list = list(range(len(num_list)))
        self.n_workers = 4

    def test_parallel_sqrt_decoratedfunc(self):
        @ParDat(job_list=self.job_list, n_workers=self.n_workers)
        def func(i):
            return i, np.sqrt(self.num_list[i])

        out_list = [None for _ in self.job_list]
        for i, j in func():
            out_list[i] = j

        self.assertEqual(self.sqrt_list, out_list)

    def test_parallel_sqrt_wrapedfunc(self):
        def func(i):
            return i, np.sqrt(self.num_list[i])

        wrapped_func = ParDat(
            job_list=self.job_list,
            n_workers=self.n_workers,
        )(func)

        out_list = [None for _ in self.job_list]
        for i, j in wrapped_func():
            out_list[i] = j

        self.assertEqual(self.sqrt_list, out_list)


if __name__ == "__main__":
    unittest.main()
