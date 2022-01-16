import unittest

import NLEval.model_trainer
import numpy as np
from NLEval import model_trainer
from NLEval.graph import DenseGraph
from NLEval.graph import FeatureVec
from NLEval.model_trainer.base import BaseTrainer
from NLEval.util.exceptions import IDNotExistError


class TestBaseTrainer(unittest.TestCase):
    def setUp(self):
        self.raw_data = raw_data = {
            "a": [1, 2, 3],
            "b": [2, 3, 4],
            "c": [3, 4, 5],
            "d": [4, 5, 6],
            "e": [5, 6, 7],
        }
        self.ids = sorted(raw_data)
        self.raw_data_list = list(map(raw_data.get, self.ids))
        self.mat = np.vstack(self.raw_data_list)
        self.features = FeatureVec.from_mat(self.mat, self.ids)

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


class TestSupervisedLearningTrainer(unittest.TestCase):
    def test_train(self):
        pass


class TestLabelPropagationTrainer(unittest.TestCase):
    def test_train(self):
        pass


if __name__ == "__main__":
    unittest.main()
