import unittest

import numpy as np

from NLEval.exception import IDNotExistError
from NLEval.feature import MultiFeatureVec
from NLEval.graph import DenseGraph
from NLEval.model_trainer.base import BaseTrainer


class TestBaseTrainer(unittest.TestCase):
    def setUp(self):
        """Setup toy multi-feature vector object.

           f1  f2  f3
        a   1   2   3
        b   2   3   4
        c   3   4   5
        d   4   5   6
        e   5   6   7
        """
        self.raw_data = raw_data = {
            "a": [1, 2, 3],
            "b": [2, 3, 4],
            "c": [3, 4, 5],
            "d": [4, 5, 6],
            "e": [5, 6, 7],
        }
        self.ids = ids = sorted(raw_data)
        self.fset_ids = fset_ids = ["f1", "f2", "f3"]
        self.raw_data_list = list(map(raw_data.get, ids))

        mat = np.vstack(self.raw_data_list)
        indptr = np.array([0, 1, 2, 3])
        self.features = MultiFeatureVec.from_mat(
            mat,
            ids,
            indptr=indptr,
            fset_ids=fset_ids,
        )

        self.graph = DenseGraph()
        for i in raw_data:
            self.graph.idmap.add_id(i)

        self.toy_metrics = {"NULL": lambda x, y: 0.0}

    def test_set_idmap(self):
        # Test normal construction
        trainer = BaseTrainer(self.toy_metrics, self.graph, self.features)
        self.assertEqual(trainer.idmap.lst, self.graph.idmap.lst)
        self.assertEqual(trainer.idmap.lst, self.features.idmap.lst)

        trainer = BaseTrainer(self.toy_metrics, graph=self.graph)
        self.assertEqual(trainer.idmap.lst, self.graph.idmap.lst)

        trainer = BaseTrainer(self.toy_metrics, features=self.graph)
        self.assertEqual(trainer.idmap.lst, self.graph.idmap.lst)

        trainer = BaseTrainer(self.toy_metrics, features=self.features)
        self.assertEqual(trainer.idmap.lst, self.features.idmap.lst)

        # Remove "d"
        self.graph.idmap.pop_id("d")
        self.assertEqual(self.graph.idmap.lst, ["a", "b", "c", "e"])
        with self.assertRaises(ValueError) as context:
            BaseTrainer(self.toy_metrics, self.graph, self.features)
        self.assertEqual(
            str(context.exception),
            "Misaligned IDs between graph and features",
        )

        # Reorder ids to ["a", "b", "c", "e", "d"]
        self.graph.idmap.add_id("d")
        self.assertEqual(self.graph.idmap.lst, ["a", "b", "c", "e", "d"])
        with self.assertRaises(ValueError) as context:
            BaseTrainer(self.toy_metrics, self.graph, self.features)
        self.assertEqual(
            str(context.exception),
            "Misaligned IDs between graph and features",
        )

    def test_get_x(self):
        trainer = BaseTrainer(self.toy_metrics, features=self.features)
        test_list = [[0, 2], [3], [0, 1, 2, 3, 4]]
        for idx in test_list:
            with self.subTest(idx=idx):
                self.assertEqual(
                    trainer.get_x(idx).tolist(),
                    [self.raw_data_list[i] for i in idx],
                )

        # Index out of range
        self.assertRaises(IndexError, trainer.get_x, [3, 5])

        trainer = BaseTrainer(self.toy_metrics, graph=self.graph)
        with self.assertRaises(ValueError) as context:
            x = trainer.get_x([0, 1])
        self.assertEqual(str(context.exception), "Features not set")

    def test_get_x_dual(self):
        trainer = BaseTrainer(
            self.toy_metrics,
            features=self.features,
            dual=True,
        )
        test_list = [[0, 2], [1], [0, 1, 2]]
        for idx in test_list:
            with self.subTest(idx=idx):
                self.assertEqual(
                    trainer.get_x(idx).tolist(),
                    [self.features.mat[:, i].tolist() for i in idx],
                )

        # Index out of range
        self.assertRaises(IndexError, trainer.get_x, [1, 3])

        # Dual mode should only work when with MultiFeatureVec
        with self.assertRaises(TypeError) as context:
            trainer = BaseTrainer(
                self.toy_metrics,
                features=self.graph,
                dual=True,
            )
        self.assertEqual(
            str(context.exception),
            "'dual' mode only works when the features is of type "
            "MultiFeatureVec, but received type "
            "<class 'NLEval.graph.dense.DenseGraph'>",
        )

        # Dual mode should only work when all feature sets are one-dimensioanl
        fvec = MultiFeatureVec.from_mats(
            [np.random.random((10, 1)), np.random.random((10, 1))],
        )
        trainer = BaseTrainer(self.toy_metrics, features=fvec, dual=True)
        with self.assertRaises(ValueError) as context:
            fvec = MultiFeatureVec.from_mats(
                [np.random.random((10, 1)), np.random.random((10, 2))],
            )
            trainer = BaseTrainer(self.toy_metrics, features=fvec, dual=True)
        self.assertEqual(
            str(context.exception),
            "'dual' mode only works when the MultiFeatureVec only contains "
            "one-dimensional feature sets.",
        )

    def test_get_x_from_ids(self):
        trainer = BaseTrainer(self.toy_metrics, features=self.features)
        test_list = [["a", "c"], ["d"], ["a", "b", "c", "d", "e"]]
        for ids in test_list:
            with self.subTest(ids=ids):
                self.assertEqual(
                    trainer.get_x_from_ids(ids).tolist(),
                    [self.raw_data[i] for i in ids],
                )

        # Unkown node id "f"
        self.assertRaises(IDNotExistError, trainer.get_x_from_ids, ["a", "f"])

    def test_get_x_from_ids_dual(self):
        trainer = BaseTrainer(
            self.toy_metrics,
            features=self.features,
            dual=True,
        )
        test_ids_list = [["f1", "f2"], ["f1"], ["f1", "f2", "f3"]]
        test_idx_list = [[0, 1], [0], [0, 1, 2]]
        for ids, idx in zip(test_ids_list, test_idx_list):
            with self.subTest(ids=ids):
                self.assertEqual(
                    trainer.get_x_from_ids(ids).tolist(),
                    [self.features.mat[:, i].tolist() for i in idx],
                )

        # Unkown node id "f4"
        self.assertRaises(IDNotExistError, trainer.get_x_from_ids, ["f2", "f4"])

    def test_get_x_from_mask(self):
        trainer = BaseTrainer(self.toy_metrics, features=self.features)
        test_list = [[1, 0, 1, 0, 0], [0, 0, 0, 1, 0], [1, 1, 1, 1, 1]]
        for mask in test_list:
            with self.subTest(mask=mask):
                self.assertEqual(
                    trainer.get_x_from_mask(np.array(mask)).tolist(),
                    [self.raw_data_list[i] for i in np.where(mask)[0]],
                )

        # Incorrect mask size
        self.assertRaises(
            ValueError,
            trainer.get_x_from_mask,
            np.array([1, 0, 1, 0, 0, 0]),
        )

    def test_get_x_from_mask_dual(self):
        trainer = BaseTrainer(
            self.toy_metrics,
            features=self.features,
            dual=True,
        )
        test_list = [[1, 0, 0], [1, 0, 1], [1, 1, 1]]
        fmat = self.features.mat
        for mask in test_list:
            with self.subTest(mask=mask):
                self.assertEqual(
                    trainer.get_x_from_mask(np.array(mask)).tolist(),
                    [fmat[:, i].tolist() for i in np.where(mask)[0]],
                )

        # Incorrect mask size
        self.assertRaises(
            ValueError,
            trainer.get_x_from_mask,
            np.array([1, 0, 1, 0]),
        )


class TestSupervisedLearningTrainer(unittest.TestCase):
    def test_train(self):
        pass


class TestLabelPropagationTrainer(unittest.TestCase):
    def test_train(self):
        pass


if __name__ == "__main__":
    unittest.main()
