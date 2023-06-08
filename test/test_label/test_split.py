import unittest

import numpy as np
from sklearn.model_selection import KFold

from obnb.exception import IDNotExistError
from obnb.label import LabelsetCollection, split
from obnb.label.split.base import BaseSplit
from obnb.util.converter import GenePropertyConverter


def test_base_split_repr(mocker):
    a = BaseSplit()
    assert repr(a) == "BaseSplit()"

    a.property_converter = {}
    assert repr(a) == "BaseSplit(property_converter=CustomConverter)"

    mocker.patch("obnb.util.converter.GenePropertyConverter._load_cache")
    a.property_converter = GenePropertyConverter(name="PubMedCount")
    assert repr(a) == (
        "BaseSplit(property_converter=GenePropertyConverter(name='PubMedCount'))"
    )

    a.some_attr = "abc"
    assert repr(a) == (
        "BaseSplit(property_converter=GenePropertyConverter(name='PubMedCount'), "
        "some_attr='abc')"
    )


class TestSplit(unittest.TestCase):
    def setUp(self):
        self.lsc = LabelsetCollection()
        self.lsc.add_labelset(["a", "b", "c"], "Labelset1", "Description1")
        self.lsc.add_labelset(["b", "d"], "Labelset2", "Description2")

    def test_raise_mask_names(self):
        self.assertRaises(
            ValueError,
            self.lsc.split,
            KFold(n_splits=2).split,
            mask_names=("train", "val", "test"),
        )

    def test_raise_label_name(self):
        self.assertRaises(
            IDNotExistError,
            self.lsc.split,
            KFold(n_splits=2).split,
            labelset_name="Labelset3",
        )

    def test_reorder(self):
        y, _ = self.lsc.split(KFold(n_splits=2).split)
        self.assertEqual(y.T.tolist(), [[1, 1, 1, 0], [0, 1, 0, 1]])

        y, _ = self.lsc.split(
            KFold(n_splits=2).split,
            target_ids=("a", "c", "b", "d"),
        )
        self.assertEqual(y.T.tolist(), [[1, 1, 1, 0], [0, 0, 1, 1]])

        y, _ = self.lsc.split(
            KFold(n_splits=2).split,
            target_ids=("a", "e", "c", "b", "d", "f"),
        )
        self.assertEqual(y.T.tolist(), [[1, 0, 1, 1, 0, 0], [0, 0, 0, 1, 1, 0]])

    def test_two_fold(self):
        train_mask = [[False, False, True, True], [True, True, False, False]]
        test_mask = [[True, True, False, False], [False, False, True, True]]

        y, masks = self.lsc.split(KFold(n_splits=2).split)
        self.assertEqual(y.T.tolist(), [[1, 1, 1, 0], [0, 1, 0, 1]])
        self.assertEqual(list(masks), ["train", "test"])
        self.assertEqual(masks["train"].T.tolist(), train_mask)
        self.assertEqual(masks["test"].T.tolist(), test_mask)

        y, masks = self.lsc.split(KFold(n_splits=2).split, labelset_name="Labelset1")
        self.assertEqual(y.T.tolist(), [1, 1, 1, 0])
        self.assertEqual(list(masks), ["train", "test"])
        self.assertEqual(masks["train"].T.tolist(), train_mask)
        self.assertEqual(masks["test"].T.tolist(), test_mask)

        y, masks = self.lsc.split(
            KFold(n_splits=2).split,
            labelset_name="Labelset1",
            target_ids=("a", "e", "c", "b", "d", "f"),
        )
        self.assertEqual(y.T.tolist(), [1, 0, 1, 1, 0, 0])
        self.assertEqual(list(masks), ["train", "test"])
        self.assertEqual(
            masks["train"].T.tolist(),
            [
                [False, False, True, False, True, False],
                [True, False, False, True, False, False],
            ],
        )
        self.assertEqual(
            masks["test"].T.tolist(),
            [
                [True, False, False, True, False, False],
                [False, False, True, False, True, False],
            ],
        )

    def test_three_fold(self):
        train_mask = [
            [False, False, True, True],
            [True, True, False, True],
            [True, True, True, False],
        ]
        test_mask = [
            [True, True, False, False],
            [False, False, True, False],
            [False, False, False, True],
        ]

        y, masks = self.lsc.split(KFold(n_splits=3).split)
        self.assertEqual(y.T.tolist(), [[1, 1, 1, 0], [0, 1, 0, 1]])
        self.assertEqual(list(masks), ["train", "test"])
        self.assertEqual(masks["train"].T.tolist(), train_mask)
        self.assertEqual(masks["test"].T.tolist(), test_mask)

        y, masks = self.lsc.split(
            KFold(n_splits=3).split,
            labelset_name="Labelset1",
        )
        self.assertEqual(y.T.tolist(), [1, 1, 1, 0])
        self.assertEqual(list(masks), ["train", "test"])
        self.assertEqual(masks["train"].T.tolist(), train_mask)
        self.assertEqual(masks["test"].T.tolist(), test_mask)


class TestLabelsetSplit(unittest.TestCase):
    def setUp(self):
        self.lsc = LabelsetCollection()
        self.lsc.add_labelset(["a", "b", "c"], "Labelset1", "Description1")
        self.lsc.add_labelset(["b", "d"], "Labelset2", "Description2")
        self.lsc.add_labelset(["e", "f", "g", "h"], "Labelset3")
        self.lsc.entity.new_property("test_property", 0, int)
        self.split_opts = {
            "property_converter": {
                j: i for i, j in enumerate(sorted(self.lsc.entity_ids))
            },
        }

        self.y_t_list = [
            [1, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ]

    def test_threshold_holdout_repr(self):
        for threshold in [-4, 0.1, 4, 10.31]:
            with self.subTest(threshold):
                splitter = split.ThresholdHoldout(threshold, **self.split_opts)
                self.assertEqual(
                    repr(splitter),
                    "ThresholdHoldout(property_converter=CustomConverter, "
                    f"ascending=True, threshold={threshold})",
                )

    def test_threshold_holdout_raises(self):
        self.assertRaises(TypeError, split.ThresholdHoldout, "6", **self.split_opts)

    def test_threshold_holdout(self):
        with self.subTest(threshold=4):
            y, masks = self.lsc.split(split.ThresholdHoldout(4, **self.split_opts))
            self.assertEqual(y.T.tolist(), self.y_t_list)
            self.assertEqual(list(masks), ["test"])
            self.assertEqual(
                masks["test"].T.tolist(),
                [[1, 1, 1, 1, 0, 0, 0, 0]],
            )

        with self.subTest(threshold=2, ascending=False):
            y, masks = self.lsc.split(
                split.ThresholdHoldout(2, ascending=False, **self.split_opts),
            )
            self.assertEqual(y.T.tolist(), self.y_t_list)
            self.assertEqual(list(masks), ["test"])
            self.assertEqual(
                masks["test"].T.tolist(),
                [[0, 0, 0, 1, 1, 1, 1, 1]],
            )

    def test_threshold_holdout_with_negatives(self):
        self.lsc.set_negative(["e"], "Labelset1")
        self.lsc.set_negative(["a", "e"], "Labelset2")

        # Only works when labelset_name is specified explicitly
        self.assertRaises(
            ValueError,
            self.lsc.split,
            split.ThresholdHoldout(4, **self.split_opts),
            consider_negative=True,
        )

        with self.subTest(threshold=4, labelset_name="Labelset1"):
            y, masks = self.lsc.split(
                split.ThresholdHoldout(4, **self.split_opts),
                labelset_name="Labelset1",
                consider_negative=True,
            )
            self.assertEqual(y.T.tolist(), self.y_t_list[0])
            self.assertEqual(
                masks["test"].T.tolist(),
                [[1, 1, 1, 0, 0, 0, 0, 0]],
            )

        with self.subTest(threshold=4, labelset_name="Labelset2"):
            y, masks = self.lsc.split(
                split.ThresholdHoldout(4, **self.split_opts),
                labelset_name="Labelset2",
                consider_negative=True,
            )
            self.assertEqual(y.T.tolist(), self.y_t_list[1])
            self.assertEqual(
                masks["test"].T.tolist(),
                [[1, 1, 0, 1, 0, 0, 0, 0]],
            )

        # If negatives are not set explicitly, use what's not positives
        with self.subTest(threshold=4, labelset_name="Labelset3"):
            y, masks = self.lsc.split(
                split.ThresholdHoldout(4, **self.split_opts),
                labelset_name="Labelset3",
                consider_negative=True,
            )
            self.assertEqual(y.T.tolist(), self.y_t_list[2])
            self.assertEqual(
                masks["test"].T.tolist(),
                [[1, 1, 1, 1, 0, 0, 0, 0]],
            )

    def test_threshold_holdout_raises_target_ids(self):
        splitter = split.ThresholdHoldout(4, **self.split_opts)

        # target_ids contains all ids in the labelset, should work
        self.lsc.split(splitter, target_ids=("a", "b", "c", "d", "e", "f", "g", "h"))
        self.lsc.split(
            splitter,
            target_ids=("a", "c", "b", "o", "k", "d", "e", "f", "g", "h"),
        )

        # Missting "g"
        with self.assertRaises(ValueError) as context:
            self.lsc.split(splitter, target_ids=("a", "b", "c", "d", "e", "f", "h"))
        self.assertEqual(
            str(context.exception),
            "target_ids must contain all of entity_ids, but 'g' is missing",
        )

    def test_ratio_holdout_repr(self):
        for ratio in [0.1, 0.5, 0.9]:
            with self.subTest(ratio=ratio):
                splitter = split.RatioHoldout(ratio, **self.split_opts)
                self.assertEqual(
                    repr(splitter),
                    "RatioHoldout(property_converter=CustomConverter, "
                    f"ascending=True, ratio={ratio})",
                )

    def test_ratio_holdout(self):
        with self.subTest(ratio=0.2):
            y, masks = self.lsc.split(split.RatioHoldout(0.2, **self.split_opts))
            self.assertEqual(y.T.tolist(), self.y_t_list)
            self.assertEqual(list(masks), ["test"])
            self.assertEqual(
                masks["test"].T.tolist(),
                [[1, 0, 0, 0, 0, 0, 0, 0]],
            )

        with self.subTest(ratio=0.2, ascending=False):
            y, masks = self.lsc.split(
                split.RatioHoldout(0.2, ascending=False, **self.split_opts),
            )
            self.assertEqual(y.T.tolist(), self.y_t_list)
            self.assertEqual(list(masks), ["test"])
            self.assertEqual(
                masks["test"].T.tolist(),
                [[0, 0, 0, 0, 0, 0, 0, 1]],
            )

        with self.subTest(ratio=0.5):
            y, masks = self.lsc.split(split.RatioHoldout(0.5, **self.split_opts))
            self.assertEqual(
                masks["test"].T.tolist(),
                [[1, 1, 1, 1, 0, 0, 0, 0]],
            )

    def test_threshold_partition_repr(self):
        with self.subTest(thresholds=(4,)):
            splitter = split.ThresholdPartition(4, **self.split_opts)
            self.assertEqual(
                repr(splitter),
                "ThresholdPartition(property_converter=CustomConverter, "
                "ascending=True, thresholds=(4,))",
            )

        with self.subTest(thresholds=(2, 7)):
            splitter = split.ThresholdPartition(2, 7, **self.split_opts)
            self.assertEqual(
                repr(splitter),
                "ThresholdPartition(property_converter=CustomConverter, "
                "ascending=True, thresholds=(2, 7))",
            )

        with self.subTest(thresholds=(6, 1, 2)):
            splitter = split.ThresholdPartition(6, 1, 2, **self.split_opts)
            self.assertEqual(
                repr(splitter),
                "ThresholdPartition(property_converter=CustomConverter, "
                "ascending=True, thresholds=(1, 2, 6))",
            )

        with self.subTest(thresholds=(6, 1, 2), ascending=False):
            splitter = split.ThresholdPartition(
                *(6, 1, 2),
                **self.split_opts,
                ascending=False,
            )
            self.assertEqual(
                repr(splitter),
                "ThresholdPartition(property_converter=CustomConverter, "
                "ascending=False, thresholds=(6, 2, 1))",
            )

    def test_threshold_partition_raises(self):
        with self.assertRaises(ValueError) as context:
            split.ThresholdPartition(5, 4, 5, **self.split_opts)
        self.assertEqual(
            str(context.exception),
            "Cannot have duplicated thresholds: 5 occurred 2 times from "
            "the input (5, 4, 5)",
        )

        with self.assertRaises(ValueError) as context:
            split.ThresholdPartition(**self.split_opts)
        self.assertEqual(str(context.exception), "No thresholds specified")

        self.assertRaises(TypeError, split.ThresholdPartition, "6", **self.split_opts)

    def test_threshold_partition(self):
        with self.subTest(thresholds=(4,)):
            y, masks = self.lsc.split(split.ThresholdPartition(4, **self.split_opts))
            self.assertEqual(y.T.tolist(), self.y_t_list)
            self.assertEqual(
                masks["train"].T.tolist(),
                [[1, 1, 1, 1, 0, 0, 0, 0]],
            )
            self.assertEqual(
                masks["test"].T.tolist(),
                [[0, 0, 0, 0, 1, 1, 1, 1]],
            )

        with self.subTest(thresholds=(2, 7)):
            y, masks = self.lsc.split(
                split.ThresholdPartition(2, 7, **self.split_opts),
            )
            self.assertEqual(y.T.tolist(), self.y_t_list)
            self.assertEqual(
                masks["train"].T.tolist(),
                [[1, 1, 0, 0, 0, 0, 0, 0]],
            )
            self.assertEqual(
                masks["val"].T.tolist(),
                [[0, 0, 1, 1, 1, 1, 1, 0]],
            )
            self.assertEqual(
                masks["test"].T.tolist(),
                [[0, 0, 0, 0, 0, 0, 0, 1]],
            )

        with self.subTest(thresholds=(6, 1, 2)):
            y, masks = self.lsc.split(
                split.ThresholdPartition(6, 1, 2, **self.split_opts),
                mask_names=("mask1", "mask2", "mask3", "mask4"),
            )
            self.assertEqual(y.T.tolist(), self.y_t_list)
            self.assertEqual(
                masks["mask1"].T.tolist(),
                [[1, 0, 0, 0, 0, 0, 0, 0]],
            )
            self.assertEqual(
                masks["mask2"].T.tolist(),
                [[0, 1, 0, 0, 0, 0, 0, 0]],
            )
            self.assertEqual(
                masks["mask3"].T.tolist(),
                [[0, 0, 1, 1, 1, 1, 0, 0]],
            )
            self.assertEqual(
                masks["mask4"].T.tolist(),
                [[0, 0, 0, 0, 0, 0, 1, 1]],
            )

        with self.subTest(thresholds=(5, 10, 20)):
            y, masks = self.lsc.split(
                split.ThresholdPartition(5, 10, 20, **self.split_opts),
                mask_names=("mask1", "mask2", "mask3", "mask4"),
            )
            self.assertEqual(y.T.tolist(), self.y_t_list)
            self.assertEqual(
                masks["mask1"].T.tolist(),
                [[1, 1, 1, 1, 1, 0, 0, 0]],
            )
            self.assertEqual(
                masks["mask2"].T.tolist(),
                [[0, 0, 0, 0, 0, 1, 1, 1]],
            )
            self.assertEqual(
                masks["mask3"].T.tolist(),
                [[0, 0, 0, 0, 0, 0, 0, 0]],
            )
            self.assertEqual(
                masks["mask4"].T.tolist(),
                [[0, 0, 0, 0, 0, 0, 0, 0]],
            )

        with self.subTest(thresholds=(-1)):
            y, masks = self.lsc.split(
                split.ThresholdPartition(-1, **self.split_opts),
            )
            self.assertEqual(y.T.tolist(), self.y_t_list)
            self.assertEqual(
                masks["train"].T.tolist(),
                [[0, 0, 0, 0, 0, 0, 0, 0]],
            )
            self.assertEqual(
                masks["test"].T.tolist(),
                [[1, 1, 1, 1, 1, 1, 1, 1]],
            )

        with self.subTest(thresholds=(2, 7)):
            y, masks = self.lsc.split(
                split.ThresholdPartition(2, 7, **self.split_opts),
            )
            self.assertEqual(y.T.tolist(), self.y_t_list)
            self.assertEqual(
                masks["train"].T.tolist(),
                [[1, 1, 0, 0, 0, 0, 0, 0]],
            )
            self.assertEqual(
                masks["val"].T.tolist(),
                [[0, 0, 1, 1, 1, 1, 1, 0]],
            )
            self.assertEqual(
                masks["test"].T.tolist(),
                [[0, 0, 0, 0, 0, 0, 0, 1]],
            )

        with self.subTest(thresholds=(5, 10, 20), ascending=False):
            y, masks = self.lsc.split(
                split.ThresholdPartition(5, 10, 20, ascending=False, **self.split_opts),
                mask_names=("mask1", "mask2", "mask3", "mask4"),
            )
            self.assertEqual(y.T.tolist(), self.y_t_list)
            self.assertEqual(
                masks["mask1"].T.tolist(),
                [[0, 0, 0, 0, 0, 0, 0, 0]],
            )
            self.assertEqual(
                masks["mask2"].T.tolist(),
                [[0, 0, 0, 0, 0, 0, 0, 0]],
            )
            self.assertEqual(
                masks["mask3"].T.tolist(),
                [[0, 0, 0, 0, 0, 0, 1, 1]],
            )
            self.assertEqual(
                masks["mask4"].T.tolist(),
                [[1, 1, 1, 1, 1, 1, 0, 0]],
            )

    def test_ratio_partition_repr(self):
        splitter = split.RatioPartition(0.5, 0.5, **self.split_opts)
        self.assertEqual(
            repr(splitter),
            "RatioPartition(property_converter=CustomConverter, "
            "ascending=True, ratios=(0.5, 0.5))",
        )

        splitter = split.RatioPartition(
            *(0.6, 0.2, 0.2),
            ascending=False,
            **self.split_opts,
        )
        self.assertEqual(
            repr(splitter),
            "RatioPartition(property_converter=CustomConverter, "
            "ascending=False, ratios=(0.6, 0.2, 0.2))",
        )

    def test_ratio_partition_raises(self):
        template = "ratio must be between 0 (exclusive) and 1 (inclusive), got"
        for ratio in [0.0, 1.000001, 2.4]:
            with self.subTest(ratio=ratio):
                with self.assertRaises(ValueError) as context:
                    split.RatioHoldout(ratio, **self.split_opts)
                self.assertEqual(
                    str(context.exception),
                    f"{template} {ratio}",
                )

        with self.assertRaises(ValueError) as context:
            split.RatioPartition(0.2, 0.5, **self.split_opts)
        self.assertEqual(
            str(context.exception),
            "Ratios must sum up to 1, specified ratios (0.2, 0.5) sum up "
            "to 0.7 instead",
        )

        with self.assertRaises(ValueError) as context:
            split.RatioPartition(0.2, 0.8, 0, **self.split_opts)
        self.assertEqual(
            str(context.exception),
            "Ratios must be strictly positive: got (0.2, 0.8, 0)",
        )

        with self.assertRaises(ValueError) as context:
            split.RatioPartition(0.2, 0.9, -0.1, **self.split_opts)
        self.assertEqual(
            str(context.exception),
            "Ratios must be strictly positive: got (0.2, 0.9, -0.1)",
        )

    def test_ratio_partition(self):
        with self.subTest(ratios=(0.5, 0.5)):
            y, masks = self.lsc.split(
                split.RatioPartition(0.5, 0.5, **self.split_opts),
            )
            self.assertEqual(y.T.tolist(), self.y_t_list)
            self.assertEqual(
                masks["train"].T.tolist(),
                [[1, 1, 1, 1, 0, 0, 0, 0]],
            )
            self.assertEqual(
                masks["test"].T.tolist(),
                [[0, 0, 0, 0, 1, 1, 1, 1]],
            )

        with self.subTest(ratios=(0.6, 0.2, 0.2)):
            y, masks = self.lsc.split(
                split.RatioPartition(0.6, 0.2, 0.2, **self.split_opts),
            )
            self.assertEqual(y.T.tolist(), self.y_t_list)
            self.assertEqual(
                masks["train"].T.tolist(),
                [[1, 1, 1, 1, 0, 0, 0, 0]],
            )
            self.assertEqual(
                masks["val"].T.tolist(),
                [[0, 0, 0, 0, 1, 1, 0, 0]],
            )
            self.assertEqual(
                masks["test"].T.tolist(),
                [[0, 0, 0, 0, 0, 0, 1, 1]],
            )

        with self.subTest(ratios=(0.6, 0.2, 0.2), ascending=False):
            y, masks = self.lsc.split(
                split.RatioPartition(
                    *(0.6, 0.2, 0.2),
                    ascending=False,
                    **self.split_opts,
                ),
            )
            self.assertEqual(y.T.tolist(), self.y_t_list)
            self.assertEqual(
                masks["train"].T.tolist(),
                [[0, 0, 0, 0, 1, 1, 1, 1]],
            )
            self.assertEqual(
                masks["val"].T.tolist(),
                [[0, 0, 1, 1, 0, 0, 0, 0]],
            )
            self.assertEqual(
                masks["test"].T.tolist(),
                [[1, 1, 0, 0, 0, 0, 0, 0]],
            )

    def test_random_ratio_partition(self):
        with self.subTest(ratios=(0.5, 0.5), shuffle=False):
            y, masks = self.lsc.split(
                split.RandomRatioPartition(0.5, 0.5, shuffle=False),
            )
            self.assertEqual(y.T.tolist(), self.y_t_list)
            self.assertEqual(list(masks), ["train", "test"])
            self.assertEqual(
                masks["train"].T.tolist(),
                [[1, 1, 1, 1, 0, 0, 0, 0]],
            )
            self.assertEqual(
                masks["test"].T.tolist(),
                [[0, 0, 0, 0, 1, 1, 1, 1]],
            )

        for random_state in [0, 32, 60]:
            with self.subTest(ratios=(0.5, 0.5), random_state=random_state):
                y, masks = self.lsc.split(
                    split.RandomRatioPartition(0.5, 0.5, random_state=random_state),
                )

                # Manually compute expected random mask
                random_x = np.random.default_rng(random_state).random(8)
                mask = np.zeros(8, dtype=bool)
                mask[random_x.argsort()[:4]] = 1

                self.assertEqual(y.T.tolist(), self.y_t_list)
                self.assertEqual(list(masks), ["train", "test"])
                self.assertEqual(masks["train"].T.tolist(), [mask.tolist()])
                self.assertEqual(masks["test"].T.tolist(), [(~mask).tolist()])

    def test_random_ratio_holdout(self):
        with self.subTest(ratio=0.5, shuffle=False):
            y, masks = self.lsc.split(
                split.RandomRatioHoldout(0.5, shuffle=False),
            )
            self.assertEqual(y.T.tolist(), self.y_t_list)
            self.assertEqual(list(masks), ["test"])
            self.assertEqual(
                masks["test"].T.tolist(),
                [[1, 1, 1, 1, 0, 0, 0, 0]],
            )

        for random_state in [0, 32, 60]:
            with self.subTest(ratios=(0.5, 0.5), random_state=random_state):
                y, masks = self.lsc.split(
                    split.RandomRatioHoldout(0.5, random_state=random_state),
                )

                # Manually compute expected random mask
                random_x = np.random.default_rng(random_state).random(8)
                mask = np.zeros(8, dtype=bool)
                mask[random_x.argsort()[:4]] = 1

                self.assertEqual(y.T.tolist(), self.y_t_list)
                self.assertEqual(masks["test"].T.tolist(), [mask.tolist()])

    def test_all_holdout(self):
        y, masks = self.lsc.split(split.AllHoldout())
        self.assertEqual(y.T.tolist(), self.y_t_list)
        self.assertEqual(list(masks), ["test"])
        self.assertEqual(masks["test"].T.tolist(), [[1, 1, 1, 1, 1, 1, 1, 1]])


if __name__ == "__main__":
    unittest.main()
