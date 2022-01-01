import unittest

import numpy as np
from NLEval.graph import DenseGraph
from NLEval.graph import SparseGraph
from NLEval.model import label_propagation


class TestLabelPropagation(unittest.TestCase):
    def setUp(self):
        self.mat = np.array(
            [
                [0, 1, 1, 0, 0],
                [1, 0, 1, 0, 1],
                [1, 1, 0, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 1, 1, 0, 0],
            ],
        )
        self.ids = ["a", "b", "c", "e", "f"]
        self.seed = np.array([0, 0.5, 0, 0.5, 0])

        # Use lazy random walk propagation
        self.lazyrw = 0.5 * (np.eye(5) + self.mat / self.mat.sum(0))
        self.dense_graph = DenseGraph.construct_graph(self.ids, self.lazyrw)
        self.sparse_graph = SparseGraph.construct_graph(self.ids, self.lazyrw)

    def tearDown(self):
        del self.mat, self.ids, self.dense_graph

    def test_construction(self):
        # Only allow non-negative tolerance
        self.assertRaises(
            ValueError,
            label_propagation.IterativePropagation,
            tol=-1e-3,
        )

        # Only allow positive max iteration
        self.assertRaises(
            ValueError,
            label_propagation.IterativePropagation,
            max_iter=0,
        )
        self.assertRaises(
            ValueError,
            label_propagation.IterativePropagation,
            max_iter=-2,
        )

    def test_iterative_propagation_dense(self):
        mdl = label_propagation.IterativePropagation(tol=1e-12)
        # Should converge to the stationary distribution, which is proportional
        # to the node degrees, since the graph is connected.
        for i, j in zip(
            mdl(self.dense_graph, self.seed),
            (self.mat.sum(0) / self.mat.sum()),
        ):
            self.assertAlmostEqual(i, j)

    @unittest.expectedFailure
    def test_iterative_propagation_sparse(self):
        mdl = label_propagation.IterativePropagation()
        mdl(self.sparse_graph, self.seed)

    def test_khop_propagation_dense(self):
        for k in range(2, 7):
            with self.subTest(k=k):
                mdl = label_propagation.KHopPropagation(k=k)
                y_pred = self.seed.copy()
                for _ in range(k):
                    y_pred = np.matmul(self.lazyrw, y_pred)
                self.assertEqual(
                    mdl(self.dense_graph, self.seed).tolist(),
                    y_pred.tolist(),
                )

    def test_onehop_propagation_dense(self):
        mdl = label_propagation.OneHopPropagation()
        self.assertEqual(
            mdl(self.dense_graph, self.seed).tolist(),
            np.matmul(self.lazyrw, self.seed).tolist(),
        )


if __name__ == "__main__":
    unittest.main()
