import os
import unittest

import numpy as np
from commonvar import SAMPLE_DATA_DIR
from NLEval import valsplit
from NLEval.label import filters
from NLEval.label import LabelsetCollection
from NLEval.label import split
from NLEval.util.exceptions import IDExistsError
from NLEval.util.exceptions import IDNotExistError
from sklearn.model_selection import KFold


class TestLabelsetCollection(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.toy1_gmt_fp = os.path.join(SAMPLE_DATA_DIR, "toy1.gmt")
        self.toy1_prop_fp = os.path.join(SAMPLE_DATA_DIR, "toy1_property.tsv")
        self.toy1_label_ids = ["Group1", "Group2", "Group3", "Group4"]
        self.toy1_InfoLst = [
            "Description1",
            "Description2",
            "Description3",
            "Description4",
        ]
        self.toy1_labelsets = [
            {"ID1", "ID2", "ID3"},
            {"ID2", "ID4", "ID5", "ID6"},
            {"ID2", "ID6"},
            {"ID7"},
        ]
        self.toy1_property = [1, 0, 9, 2]

    def setUp(self):
        self.lsc = LabelsetCollection()
        self.lsc.add_labelset(["a", "b", "c"], "Labelset1", "Description1")
        self.lsc.add_labelset(["b", "d"], "Labelset2", "Description2")

    def template_test_input_for_getters(self, fun):
        """Template for testing inputs for methods with only one positional
        argument as ID, i.e. `.get_info`, `get_labelset`, and `get_noccur`."""
        # input type other than str --> TypeError
        self.assertRaises(TypeError, fun, 1)
        self.assertRaises(TypeError, fun, ["1"])
        # input unknown ID --> IDNotExistError
        self.assertRaises(IDNotExistError, fun, "1")

    def test_sizes(self):
        self.assertEqual(self.lsc.sizes, [3, 2])

    def test_get_info(self):
        self.template_test_input_for_getters(self.lsc.get_info)
        self.assertEqual(self.lsc.get_info("Labelset1"), "Description1")
        self.assertEqual(self.lsc.get_info("Labelset2"), "Description2")

    def test_get_labelset(self):
        self.template_test_input_for_getters(self.lsc.get_labelset)
        self.assertEqual(self.lsc.get_labelset("Labelset1"), {"a", "b", "c"})
        self.assertEqual(self.lsc.get_labelset("Labelset2"), {"b", "d"})

    def test_get_noccur(self):
        self.template_test_input_for_getters(self.lsc.get_noccur)
        self.assertEqual(self.lsc.get_noccur("a"), 1)
        self.assertEqual(self.lsc.get_noccur("b"), 2)
        self.assertEqual(self.lsc.get_noccur("c"), 1)
        self.assertEqual(self.lsc.get_noccur("d"), 1)

    def test_pop_entity(self):
        self.template_test_input_for_getters(self.lsc.pop_entity)
        self.lsc.pop_entity("b")
        self.assertEqual(self.lsc.entity.map, {"a": 0, "c": 1, "d": 2})
        self.assertEqual(self.lsc.get_labelset("Labelset1"), {"a", "c"})
        self.assertEqual(self.lsc.get_labelset("Labelset2"), {"d"})

    def test_get_negative(self):
        self.template_test_input_for_getters(self.lsc.get_negative)
        # test unspecified negative
        self.assertEqual(self.lsc.get_negative("Labelset1"), {"d"})
        self.assertEqual(self.lsc.get_negative("Labelset2"), {"a", "c"})
        # test if unrelated entities included
        self.lsc.entity.add_id("h")
        self.assertEqual(self.lsc.get_negative("Labelset2"), {"a", "c"})
        # test specified negative
        self.lsc.set_negative(["a"], "Labelset2")
        self.assertEqual(self.lsc.get_negative("Labelset2"), {"a"})

    def test_set_negative(self):
        # test with known entity ID
        self.lsc.set_negative(["a"], "Labelset2")
        self.assertEqual(self.lsc.get_negative("Labelset2"), {"a"})
        self.lsc.set_negative(["c"], "Labelset2")
        self.assertEqual(self.lsc.get_negative("Labelset2"), {"c"})
        # test with knwon positive ID --> IDExistsError
        self.assertRaises(
            IDExistsError,
            self.lsc.set_negative,
            ["b"],
            "Labelset2",
        )
        self.assertRaises(
            IDExistsError,
            self.lsc.set_negative,
            ["a", "d"],
            "Labelset2",
        )
        # test with unknown entity ID --> IDNotExistError
        self.assertRaises(
            IDNotExistError,
            self.lsc.set_negative,
            ["e"],
            "Labelset2",
        )
        self.assertRaises(
            IDNotExistError,
            self.lsc.set_negative,
            ["a", "e"],
            "Labelset2",
        )

    def test_eq(self):
        # make two identical labelset collections by shuffling the order of labelset
        shuffle_idx = [3, 0, 2, 1]
        lsc1 = LabelsetCollection()
        lsc2 = LabelsetCollection()
        for idx1 in range(4):
            idx2 = shuffle_idx[idx1]
            for lsc, idx in zip((lsc1, lsc2), (idx1, idx2)):
                lsc.add_labelset(
                    list(self.toy1_labelsets[idx]),
                    self.toy1_label_ids[idx],
                    self.toy1_InfoLst[idx],
                )
        self.assertEqual(lsc1, lsc2)
        # test if different description
        lsc3 = lsc2.copy()
        lsc3.set_property("Group1", "Info", "Some other description")
        self.assertNotEqual(lsc1, lsc3)
        # test if different labelset with same label_id
        lsc3 = lsc2.copy()
        lsc3.update_labelset(["a"], "Group1")
        self.assertNotEqual(lsc1, lsc3)
        # make sure lsc2 still the same as lsc1
        self.assertEqual(lsc1, lsc2)

    def test_from_gmt(self):
        lsc = LabelsetCollection.from_gmt(self.toy1_gmt_fp)
        self.assertEqual(lsc.label_ids, self.toy1_label_ids)
        self.assertEqual(lsc.prop["Info"], self.toy1_InfoLst)
        self.assertEqual(lsc.prop["Labelset"], self.toy1_labelsets)

    def test_add_labelset(self):
        with self.subTest(msg="Input checks"):
            # test lst input type, only list of string allowed
            self.assertRaises(TypeError, self.lsc.add_labelset, 1, "Labelset3")
            self.assertRaises(
                TypeError,
                self.lsc.add_labelset,
                ["1", 2],
                "Labelset3",
            )
            self.assertRaises(
                TypeError,
                self.lsc.add_labelset,
                "123",
                "Labelset3",
            )
            # test label ID input type --> TypeError
            self.assertRaises(TypeError, self.lsc.add_labelset, ["a"], 123)
            self.assertRaises(TypeError, self.lsc.add_labelset, ["a"], [1, 2])
            self.assertRaises(
                TypeError,
                self.lsc.add_labelset,
                ["a"],
                ["Labelset"],
            )
            # test label info input type --> TypeError
            self.assertRaises(
                TypeError,
                self.lsc.add_labelset,
                ["a"],
                "Labelset3",
                [1, 2, 3],
            )
            self.assertRaises(
                TypeError,
                self.lsc.add_labelset,
                ["a"],
                "Labelset3",
                ["Description"],
            )
            # make sure no new label added with exception
            self.assertEqual(self.lsc.label_ids, ["Labelset1", "Labelset2"])
            # test add existing label ID --> IDExistsError
            self.assertRaises(
                IDExistsError,
                self.lsc.add_labelset,
                ["e", "f"],
                "Labelset1",
            )
            # test label info specification --> Info default to 'NA' if not specified
            self.lsc.add_labelset(["e"], "Labelset3")
            self.assertEqual(
                self.lsc._prop["Info"],
                ["Description1", "Description2", "NA"],
            )
        with self.subTest(msg="Labelset loading checks"):
            # test input empty labelset
            self.lsc.add_labelset([], "Labelset4")
            # check if labelset loaded correctly
            self.assertEqual(
                self.lsc._prop["Labelset"],
                [{"a", "b", "c"}, {"b", "d"}, {"e"}, set()],
            )
            # check if entity map setup correctly
            self.assertEqual(
                self.lsc.entity.map,
                {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4},
            )

    def test_pop_labelset(self):
        with self.subTest(msg="Input checks"):
            # test wrong label_id type --> TypeError
            self.assertRaises(TypeError, self.lsc.pop_labelset, 1)
            self.assertRaises(TypeError, self.lsc.pop_labelset, ["Labelset1"])
            # test not exist label_id --> IDNotExistError
            self.assertRaises(
                IDNotExistError,
                self.lsc.pop_labelset,
                "Labelset3",
            )
            # make sure nothing poped
            self.assertEqual(self.lsc.lst, ["Labelset1", "Labelset2"])
        # make sure enties that are no longer in any labelset are popped
        self.lsc.pop_labelset("Labelset1")
        self.assertEqual(self.lsc.label_ids, ["Labelset2"])
        self.assertEqual(self.lsc.entity.map, {"b": 0, "d": 1})
        self.lsc.pop_labelset("Labelset2")
        self.assertEqual(self.lsc.label_ids, [])
        self.assertEqual(self.lsc.entity.map, {})

    def test_update_labelset(self):
        with self.subTest(msg="Input checks"):
            # test lst input, only list of string allowed
            self.assertRaises(
                TypeError,
                self.lsc.update_labelset,
                1,
                "Labelset1",
            )
            self.assertRaises(
                TypeError,
                self.lsc.update_labelset,
                ["1", 2],
                "Labelset1",
            )
            self.assertRaises(
                TypeError,
                self.lsc.update_labelset,
                "123",
                "Labelset1",
            )
            # test label_id input type
            self.assertRaises(TypeError, self.lsc.update_labelset, ["a"], 123)
            self.assertRaises(
                TypeError,
                self.lsc.update_labelset,
                ["a"],
                [1, 2],
            )
            self.assertRaises(
                TypeError,
                self.lsc.update_labelset,
                ["a"],
                ["Labelset1"],
            )
            # test reset not exist label_id --> IDNotExistError
            self.assertRaises(
                IDNotExistError,
                self.lsc.update_labelset,
                ["a"],
                "Labelset3",
            )
        # test update nothing --> labelset stays the same
        self.lsc.update_labelset([], "Labelset1")
        self.assertEqual(self.lsc.get_labelset("Labelset1"), {"a", "b", "c"})
        # test update existing --> labelset stays the same
        self.lsc.update_labelset(["a", "b", "c"], "Labelset1")
        self.assertEqual(self.lsc.get_labelset("Labelset1"), {"a", "b", "c"})
        # test update partially new
        self.lsc.update_labelset(["a", "d"], "Labelset1")
        self.assertEqual(
            self.lsc.get_labelset("Labelset1"),
            {"a", "b", "c", "d"},
        )
        # test update all new
        self.lsc.update_labelset(["e"], "Labelset1")
        self.assertEqual(
            self.lsc.get_labelset("Labelset1"),
            {"a", "b", "c", "d", "e"},
        )
        # check if new entity added to list
        self.assertEqual(
            self.lsc.entity.map,
            {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4},
        )

    def test_reset_labelset(self):
        self.template_test_input_for_getters(self.lsc.reset_labelset)
        # check if labelset reset to empty set correctly
        self.lsc.reset_labelset("Labelset1")
        self.assertEqual(self.lsc.get_labelset("Labelset1"), set())
        self.assertEqual(self.lsc.get_labelset("Labelset2"), {"b", "d"})
        # makesure list of labelsets untouched
        self.assertEqual(self.lsc.label_ids, ["Labelset1", "Labelset2"])
        # make sure entities that are nolongler in any labelset are popped
        self.assertEqual(self.lsc.entity.map, {"b": 0, "d": 1})

    def test_load_entity_properties(self):
        self.lsc.load_entity_properties(self.toy1_prop_fp, "toy1_prop", 0, int)
        self.assertEqual(self.lsc.entity.prop["toy1_prop"], self.toy1_property)
        # test loading property with existing property name --> IDExistsError
        self.assertRaises(
            IDExistsError,
            self.lsc.load_entity_properties,
            self.toy1_prop_fp,
            "toy1_prop",
            0,
            int,
        )
        # make sure entity with properties different from default don't get popped
        self.lsc.pop_labelset("Labelset1")
        self.lsc.pop_labelset("Labelset2")
        self.assertEqual(self.lsc.entity.map, {"a": 0, "c": 1, "d": 2})


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
            mask_names=["train", "val", "test"],
        )

    def test_raise_label_name(self):
        self.assertRaises(
            IDNotExistError,
            self.lsc.split,
            KFold(n_splits=2).split,
            labelset_name="Labelset3",
        )

    def test_raise_property_name(self):
        self.assertRaises(
            IDNotExistError,
            self.lsc.split,
            KFold(n_splits=2).split,
            property_name="something",
        )

    def test_reorder(self):
        y, _ = self.lsc.split(KFold(n_splits=2).split)
        self.assertEqual(y.T.tolist(), [[1, 1, 1, 0], [0, 1, 0, 1]])

        y, _ = self.lsc.split(
            KFold(n_splits=2).split,
            target_ids=["a", "c", "b", "d"],
        )
        self.assertEqual(y.T.tolist(), [[1, 1, 1, 0], [0, 0, 1, 1]])

        y, _ = self.lsc.split(
            KFold(n_splits=2).split,
            target_ids=["a", "e", "c", "b", "d", "f"],
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

        y, masks = self.lsc.split(
            KFold(n_splits=2).split,
            labelset_name="Labelset1",
        )
        self.assertEqual(y.T.tolist(), [1, 1, 1, 0])
        self.assertEqual(list(masks), ["train", "test"])
        self.assertEqual(masks["train"].T.tolist(), train_mask)
        self.assertEqual(masks["test"].T.tolist(), test_mask)

        y, masks = self.lsc.split(
            KFold(n_splits=2).split,
            labelset_name="Labelset1",
            target_ids=["a", "e", "c", "b", "d", "f"],
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
        for i, j in enumerate(["a", "b", "c", "d", "e", "f", "g", "h"]):
            self.lsc.entity.set_property(j, "test_property", i)

        self.y_t_list = [
            [1, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ]

    def test_threshold_holdout_repr(self):
        for threshold in [-4, 0.1, 4, 10.31]:
            with self.subTest(threshold):
                splitter = split.ThresholdHoldout(threshold)
                self.assertEqual(
                    repr(splitter),
                    f"ThresholdHoldout(ascending=True, threshold={threshold})",
                )

    def test_threshold_holdout_raises(self):
        self.assertRaises(TypeError, split.ThresholdHoldout, "6")

    def test_threshold_holdout(self):
        with self.subTest(threshold=4):
            y, masks = self.lsc.split(
                split.ThresholdHoldout(4),
                property_name="test_property",
            )
            self.assertEqual(y.T.tolist(), self.y_t_list)
            self.assertEqual(list(masks), ["test"])
            self.assertEqual(
                masks["test"].T.tolist(),
                [[1, 1, 1, 1, 0, 0, 0, 0]],
            )

        with self.subTest(threshold=2, ascending=False):
            y, masks = self.lsc.split(
                split.ThresholdHoldout(2, ascending=False),
                property_name="test_property",
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
            split.ThresholdHoldout(4),
            property_name="test_property",
            consider_negative=True,
        )

        with self.subTest(threshold=4, labelset_name="Labelset1"):
            y, masks = self.lsc.split(
                split.ThresholdHoldout(4),
                labelset_name="Labelset1",
                property_name="test_property",
                consider_negative=True,
            )
            self.assertEqual(y.T.tolist(), self.y_t_list[0])
            self.assertEqual(
                masks["test"].T.tolist(),
                [[1, 1, 1, 0, 0, 0, 0, 0]],
            )

        with self.subTest(threshold=4, labelset_name="Labelset2"):
            y, masks = self.lsc.split(
                split.ThresholdHoldout(4),
                labelset_name="Labelset2",
                property_name="test_property",
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
                split.ThresholdHoldout(4),
                labelset_name="Labelset3",
                property_name="test_property",
                consider_negative=True,
            )
            self.assertEqual(y.T.tolist(), self.y_t_list[2])
            self.assertEqual(
                masks["test"].T.tolist(),
                [[1, 1, 1, 1, 0, 0, 0, 0]],
            )

    def test_threshold_holdout_raises_target_ids(self):
        splitter = split.ThresholdHoldout(4)

        # target_ids contains all ids in the labelset, should work
        self.lsc.split(
            splitter,
            target_ids=["a", "b", "c", "d", "e", "f", "g", "h"],
            property_name="test_property",
        )
        self.lsc.split(
            splitter,
            target_ids=["a", "c", "b", "o", "k", "d", "e", "f", "g", "h"],
            property_name="test_property",
        )

        # Missting "g"
        with self.assertRaises(ValueError) as context:
            self.lsc.split(
                splitter,
                target_ids=["a", "b", "c", "d", "e", "f", "h"],
                property_name="test_property",
            )
        self.assertEqual(
            str(context.exception),
            "target_ids must contain all of entity_ids, but 'g' is missing",
        )

    def test_ratio_holdout_repr(self):
        for ratio in [0.1, 0.5, 0.9]:
            with self.subTest(ratio=ratio):
                splitter = split.RatioHoldout(ratio)
                self.assertEqual(
                    repr(splitter),
                    f"RatioHoldout(ascending=True, ratio={ratio})",
                )

    def test_ratio_partition_raises(self):
        for ratio in [0, 1, 2.4]:
            with self.subTest(ratio=ratio):
                with self.assertRaises(ValueError) as context:
                    split.RatioHoldout(ratio)
                self.assertEqual(
                    str(context.exception),
                    "ratio must be strictly between 0 and 1, got {ratio}",
                )

    def test_ratio_holdout(self):
        with self.subTest(ratio=0.2):
            y, masks = self.lsc.split(
                split.RatioHoldout(0.2),
                property_name="test_property",
            )
            self.assertEqual(y.T.tolist(), self.y_t_list)
            self.assertEqual(list(masks), ["test"])
            self.assertEqual(
                masks["test"].T.tolist(),
                [[1, 0, 0, 0, 0, 0, 0, 0]],
            )

        with self.subTest(ratio=0.2, ascending=False):
            y, masks = self.lsc.split(
                split.RatioHoldout(0.2, ascending=False),
                property_name="test_property",
            )
            self.assertEqual(y.T.tolist(), self.y_t_list)
            self.assertEqual(list(masks), ["test"])
            self.assertEqual(
                masks["test"].T.tolist(),
                [[0, 0, 0, 0, 0, 0, 0, 1]],
            )

        with self.subTest(ratio=0.5):
            y, masks = self.lsc.split(
                split.RatioHoldout(0.5),
                property_name="test_property",
            )
            self.assertEqual(
                masks["test"].T.tolist(),
                [[1, 1, 1, 1, 0, 0, 0, 0]],
            )

    def test_threshold_partition_repr(self):
        with self.subTest(thresholds=(4,)):
            splitter = split.ThresholdPartition(4)
            self.assertEqual(
                repr(splitter),
                "ThresholdPartition(ascending=True, thresholds=(4,))",
            )

        with self.subTest(thresholds=(2, 7)):
            splitter = split.ThresholdPartition(2, 7)
            self.assertEqual(
                repr(splitter),
                "ThresholdPartition(ascending=True, thresholds=(2, 7))",
            )

        with self.subTest(thresholds=(6, 1, 2)):
            splitter = split.ThresholdPartition(6, 1, 2)
            self.assertEqual(
                repr(splitter),
                "ThresholdPartition(ascending=True, thresholds=(1, 2, 6))",
            )

        with self.subTest(thresholds=(6, 1, 2), ascending=False):
            splitter = split.ThresholdPartition(6, 1, 2, ascending=False)
            self.assertEqual(
                repr(splitter),
                "ThresholdPartition(ascending=False, thresholds=(6, 2, 1))",
            )

    def test_threshold_partition_raises(self):
        with self.assertRaises(ValueError) as context:
            split.ThresholdPartition(5, 4, 5)
        self.assertEqual(
            str(context.exception),
            "Cannot have duplicated thresholds: 5 occured 2 times from "
            "the input (5, 4, 5)",
        )

        with self.assertRaises(ValueError) as context:
            split.ThresholdPartition()
        self.assertEqual(str(context.exception), "No thresholds specified")

        self.assertRaises(
            TypeError,
            split.ThresholdPartition,
            "6",
        )

    def test_threshold_partition(self):
        with self.subTest(thresholds=(4,)):
            y, masks = self.lsc.split(
                split.ThresholdPartition(4),
                property_name="test_property",
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

        with self.subTest(thresholds=(2, 7)):
            y, masks = self.lsc.split(
                split.ThresholdPartition(2, 7),
                property_name="test_property",
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
                split.ThresholdPartition(6, 1, 2),
                property_name="test_property",
                mask_names=["mask1", "mask2", "mask3", "mask4"],
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
                split.ThresholdPartition(5, 10, 20),
                property_name="test_property",
                mask_names=["mask1", "mask2", "mask3", "mask4"],
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
                split.ThresholdPartition(-1),
                property_name="test_property",
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
                split.ThresholdPartition(2, 7),
                property_name="test_property",
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
                split.ThresholdPartition(5, 10, 20, ascending=False),
                property_name="test_property",
                mask_names=["mask1", "mask2", "mask3", "mask4"],
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
        splitter = split.RatioPartition(0.5, 0.5)
        self.assertEqual(
            repr(splitter),
            "RatioPartition(ascending=True, ratios=(0.5, 0.5))",
        )

        splitter = split.RatioPartition(0.6, 0.2, 0.2, ascending=False)
        self.assertEqual(
            repr(splitter),
            "RatioPartition(ascending=False, ratios=(0.6, 0.2, 0.2))",
        )

    def test_ratio_partition_raises(self):
        with self.assertRaises(ValueError) as context:
            split.RatioPartition(0.2, 0.5)
        self.assertEqual(
            str(context.exception),
            "Ratios must sum up to 1, specified ratios (0.2, 0.5) sum up "
            "to 0.7 instead",
        )

        with self.assertRaises(ValueError) as context:
            split.RatioPartition(0.2, 0.8, 0)
        self.assertEqual(
            str(context.exception),
            "Ratios must be strictly positive: got (0.2, 0.8, 0)",
        )

        with self.assertRaises(ValueError) as context:
            split.RatioPartition(0.2, 0.9, -0.1)
        self.assertEqual(
            str(context.exception),
            "Ratios must be strictly positive: got (0.2, 0.9, -0.1)",
        )

    def test_ratio_partition(self):
        with self.subTest(ratios=(0.5, 0.5)):
            y, masks = self.lsc.split(
                split.RatioPartition(0.5, 0.5),
                property_name="test_property",
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
                split.RatioPartition(0.6, 0.2, 0.2),
                property_name="test_property",
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
                split.RatioPartition(0.6, 0.2, 0.2, ascending=False),
                property_name="test_property",
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


class TestFilter(unittest.TestCase):
    def setUp(self):
        self.lsc = LabelsetCollection()
        self.lsc.add_labelset(["a", "b", "c"], "Group1")
        self.lsc.add_labelset(["b", "d"], "Group2")
        self.lsc.add_labelset(["e", "f", "g"], "Group3")
        self.lsc.add_labelset(["a", "f", "c"], "Group4")
        self.lsc.add_labelset(["a", "h"], "Group5")
        # Noccur=[3, 2, 2, 1, 1, 2, 1, 1]
        # Size=[3, 2, 3, 3, 2]

    def test_iapply(self):
        # Make sure iapply work as an inplace version of apply
        target_lst = ["a", "b", "c"]
        lsc = self.lsc.apply(
            filters.EntityExistenceFilter(target_lst=target_lst),
            inplace=False,
        )
        self.lsc.iapply(filters.EntityExistenceFilter(target_lst=target_lst))
        self.assertEqual(
            self.lsc.prop["Labelset"],
            lsc.prop["Labelset"],
        )
        self.assertEqual(lsc.entity.map, self.lsc.entity.map)

    def test_EntityExistenceFilter(self):
        # make sure default options of remove_specified=False work
        target_lst = ["a", "b", "c"]
        with self.subTest(target_lst=target_lst):
            lsc = self.lsc.apply(
                filters.EntityExistenceFilter(target_lst=target_lst),
                inplace=False,
            )
            self.assertEqual(
                lsc.prop["Labelset"],
                [{"a", "b", "c"}, {"b"}, set(), {"a", "c"}, {"a"}],
            )
            self.assertEqual(lsc.entity.map, {"a": 0, "b": 1, "c": 2})

        target_lst = ["a", "b", "c"]
        remove_specified = False
        with self.subTest(
            target_lst=target_lst,
            remove_specified=remove_specified,
        ):
            lsc = self.lsc.apply(
                filters.EntityExistenceFilter(
                    target_lst=target_lst,
                    remove_specified=remove_specified,
                ),
                inplace=False,
            )
            self.assertEqual(
                lsc.prop["Labelset"],
                [{"a", "b", "c"}, {"b"}, set(), {"a", "c"}, {"a"}],
            )
            self.assertEqual(lsc.entity.map, {"a": 0, "b": 1, "c": 2})

        target_lst = ["a", "b", "c"]
        remove_specified = True
        with self.subTest(
            target_lst=target_lst,
            remove_specified=remove_specified,
        ):
            lsc = self.lsc.apply(
                filters.EntityExistenceFilter(
                    target_lst=target_lst,
                    remove_specified=remove_specified,
                ),
                inplace=False,
            )
            self.assertEqual(
                lsc.prop["Labelset"],
                [set(), {"d"}, {"e", "f", "g"}, {"f"}, {"h"}],
            )
            self.assertEqual(
                lsc.entity.map,
                {"d": 0, "e": 1, "f": 2, "g": 3, "h": 4},
            )

        target_lst = ["b", "e", "h"]
        remove_specified = False
        with self.subTest(
            target_lst=target_lst,
            remove_specified=remove_specified,
        ):
            lsc = self.lsc.apply(
                filters.EntityExistenceFilter(
                    target_lst=target_lst,
                    remove_specified=remove_specified,
                ),
                inplace=False,
            )
            self.assertEqual(
                lsc.prop["Labelset"],
                [{"b"}, {"b"}, {"e"}, set(), {"h"}],
            )
            self.assertEqual(lsc.entity.map, {"b": 0, "e": 1, "h": 2})

        target_lst = ["b", "e", "h"]
        remove_specified = True
        with self.subTest(
            target_lst=target_lst,
            remove_specified=remove_specified,
        ):
            lsc = self.lsc.apply(
                filters.EntityExistenceFilter(
                    target_lst=target_lst,
                    remove_specified=remove_specified,
                ),
                inplace=False,
            )
            self.assertEqual(
                lsc.prop["Labelset"],
                [{"a", "c"}, {"d"}, {"f", "g"}, {"a", "f", "c"}, {"a"}],
            )
            self.assertEqual(
                lsc.entity.map,
                {"a": 0, "c": 1, "d": 2, "f": 3, "g": 4},
            )

    def test_LabelsetExistenceFilter(self):
        target_lst = ["Group1", "Group2"]
        with self.subTest(target_lst=target_lst):
            lsc = self.lsc.apply(
                filters.LabelsetExistenceFilter(target_lst=target_lst),
                inplace=False,
            )
            self.assertEqual(
                lsc.prop["Labelset"],
                [{"a", "b", "c"}, {"b", "d"}],
            )
            self.assertEqual(lsc.entity.map, {"a": 0, "b": 1, "c": 2, "d": 3})

        target_lst = ["Group1", "Group2"]
        remove_specified = False
        with self.subTest(target_lst=target_lst):
            lsc = self.lsc.apply(
                filters.LabelsetExistenceFilter(
                    target_lst=target_lst,
                    remove_specified=remove_specified,
                ),
                inplace=False,
            )
            self.assertEqual(
                lsc.prop["Labelset"],
                [{"a", "b", "c"}, {"b", "d"}],
            )
            self.assertEqual(lsc.entity.map, {"a": 0, "b": 1, "c": 2, "d": 3})

        target_lst = ["Group1", "Group2"]
        remove_specified = True
        with self.subTest(target_lst=target_lst):
            lsc = self.lsc.apply(
                filters.LabelsetExistenceFilter(
                    target_lst=target_lst,
                    remove_specified=remove_specified,
                ),
                inplace=False,
            )
            self.assertEqual(
                lsc.prop["Labelset"],
                [{"e", "f", "g"}, {"a", "f", "c"}, {"a", "h"}],
            )
            self.assertEqual(
                lsc.entity.map,
                {"a": 0, "c": 1, "e": 2, "f": 3, "g": 4, "h": 5},
            )

        target_lst = ["Group2", "Group5"]
        remove_specified = False
        with self.subTest(target_lst=target_lst):
            lsc = self.lsc.apply(
                filters.LabelsetExistenceFilter(
                    target_lst=target_lst,
                    remove_specified=remove_specified,
                ),
                inplace=False,
            )
            self.assertEqual(lsc.prop["Labelset"], [{"b", "d"}, {"a", "h"}])
            self.assertEqual(lsc.entity.map, {"a": 0, "b": 1, "d": 2, "h": 3})

        target_lst = ["Group2", "Group5"]
        remove_specified = True
        with self.subTest(target_lst=target_lst):
            lsc = self.lsc.apply(
                filters.LabelsetExistenceFilter(
                    target_lst=target_lst,
                    remove_specified=remove_specified,
                ),
                inplace=False,
            )
            self.assertEqual(
                lsc.prop["Labelset"],
                [{"a", "b", "c"}, {"e", "f", "g"}, {"a", "f", "c"}],
            )
            self.assertEqual(
                lsc.entity.map,
                {"a": 0, "b": 1, "c": 2, "e": 3, "f": 4, "g": 5},
            )

    def test_EntityRangeFilterNoccur(self):
        with self.subTest(min_val=2):
            lsc = self.lsc.apply(
                filters.EntityRangeFilterNoccur(min_val=2),
                inplace=False,
            )
            self.assertEqual(
                lsc.prop["Labelset"],
                [{"a", "b", "c"}, {"b"}, {"f"}, {"a", "f", "c"}, {"a"}],
            )
            self.assertEqual(lsc.entity.map, {"a": 0, "b": 1, "c": 2, "f": 3})
        with self.subTest(min_val=3):
            lsc = self.lsc.apply(
                filters.EntityRangeFilterNoccur(min_val=3),
                inplace=False,
            )
            self.assertEqual(
                lsc.prop["Labelset"],
                [{"a"}, set(), set(), {"a"}, {"a"}],
            )
            self.assertEqual(lsc.entity.map, {"a": 0})
        with self.subTest(min_val=4):
            lsc = self.lsc.apply(
                filters.EntityRangeFilterNoccur(min_val=4),
                inplace=False,
            )
            self.assertEqual(
                lsc.prop["Labelset"],
                [set(), set(), set(), set(), set()],
            )
            self.assertEqual(lsc.entity.map, {})
        with self.subTest(max_val=2):
            lsc = self.lsc.apply(
                filters.EntityRangeFilterNoccur(max_val=2),
                inplace=False,
            )
            self.assertEqual(
                lsc.prop["Labelset"],
                [{"b", "c"}, {"b", "d"}, {"e", "f", "g"}, {"f", "c"}, {"h"}],
            )
            self.assertEqual(
                lsc.entity.map,
                {"b": 0, "c": 1, "d": 2, "e": 3, "f": 4, "g": 5, "h": 6},
            )
        with self.subTest(max_val=1):
            lsc = self.lsc.apply(
                filters.EntityRangeFilterNoccur(max_val=1),
                inplace=False,
            )
            self.assertEqual(
                lsc.prop["Labelset"],
                [set(), {"d"}, {"e", "g"}, set(), {"h"}],
            )
            self.assertEqual(lsc.entity.map, {"d": 0, "e": 1, "g": 2, "h": 3})
        with self.subTest(max_val=0):
            lsc = self.lsc.apply(
                filters.EntityRangeFilterNoccur(max_val=0),
                inplace=False,
            )
            self.assertEqual(
                lsc.prop["Labelset"],
                [set(), set(), set(), set(), set()],
            )
            self.assertEqual(lsc.entity.map, {})
        with self.subTest(min_val=2, max_val=2):
            lsc = self.lsc.apply(
                filters.EntityRangeFilterNoccur(min_val=2, max_val=2),
                inplace=False,
            )
            self.assertEqual(
                lsc.prop["Labelset"],
                [{"b", "c"}, {"b"}, {"f"}, {"f", "c"}, set()],
            )
            self.assertEqual(lsc.entity.map, {"b": 0, "c": 1, "f": 2})

    def test_LabelsetRangeFilterSize(self):
        with self.subTest(min_val=3):
            lsc = self.lsc.apply(
                filters.LabelsetRangeFilterSize(min_val=3),
                inplace=False,
            )
            self.assertEqual(lsc.label_ids, ["Group1", "Group3", "Group4"])
            self.assertEqual(
                lsc.prop["Labelset"],
                [{"a", "b", "c"}, {"e", "f", "g"}, {"a", "f", "c"}],
            )
            self.assertEqual(
                lsc.entity.map,
                {"a": 0, "b": 1, "c": 2, "e": 3, "f": 4, "g": 5},
            )
        with self.subTest(min_val=4):
            lsc = self.lsc.apply(
                filters.LabelsetRangeFilterSize(min_val=4),
                inplace=False,
            )
            self.assertEqual(lsc.label_ids, [])
            self.assertEqual(lsc.prop["Labelset"], [])
            self.assertEqual(lsc.entity.map, {})
        with self.subTest(max_val=2):
            lsc = self.lsc.apply(
                filters.LabelsetRangeFilterSize(max_val=2),
                inplace=False,
            )
            self.assertEqual(lsc.label_ids, ["Group2", "Group5"])
            self.assertEqual(lsc.prop["Labelset"], [{"b", "d"}, {"a", "h"}])
            self.assertEqual(lsc.entity.map, {"a": 0, "b": 1, "d": 2, "h": 3})
        with self.subTest(max_val=1):
            lsc = self.lsc.apply(
                filters.LabelsetRangeFilterSize(max_val=1),
                inplace=False,
            )
            self.assertEqual(lsc.label_ids, [])
            self.assertEqual(lsc.prop["Labelset"], [])
            self.assertEqual(lsc.entity.map, {})

    def test_LabelsetRangeFilterJaccard(self):
        with self.subTest(min_val=0.9):
            lsc = self.lsc.apply(
                filters.LabelsetRangeFilterJaccard(max_val=0.9),
                inplace=False,
            )
            self.assertEqual(
                lsc.label_ids,
                ["Group1", "Group2", "Group3", "Group4", "Group5"],
            )
        with self.subTest(min_val=0.4):
            lsc = self.lsc.apply(
                filters.LabelsetRangeFilterJaccard(max_val=0.4),
                inplace=False,
            )
            self.assertEqual(
                lsc.label_ids,
                ["Group2", "Group3", "Group4", "Group5"],
            )
        with self.subTest(min_val=0.2):
            lsc = self.lsc.apply(
                filters.LabelsetRangeFilterJaccard(max_val=0.2),
                inplace=False,
            )
            self.assertEqual(
                lsc.label_ids,
                ["Group2", "Group3", "Group5"],
            )
        with self.subTest(min_val=0):
            lsc = self.lsc.apply(
                filters.LabelsetRangeFilterJaccard(max_val=0),
                inplace=False,
            )
            self.assertEqual(
                lsc.label_ids,
                ["Group2", "Group5"],
            )

    def test_LabelsetRangeFilterTrainTestPos(self):
        train_index = np.array(["a", "b", "c", "d"])
        test_index = np.array(["e", "f", "g", "h"])
        with self.subTest(train=train_index, test=test_index):
            splitter = valsplit.Holdout.CustomHold(train_index, test_index)
            splitter._train_index = splitter.custom_train_index
            splitter._test_index = splitter.custom_test_index
            self.lsc.valsplit = splitter
            lsc = self.lsc.apply(
                filters.LabelsetRangeFilterTrainTestPos(min_val=1),
                inplace=False,
            )
            self.assertEqual(lsc.label_ids, ["Group4", "Group5"])
            self.assertEqual(
                lsc.prop["Labelset"],
                [{"a", "f", "c"}, {"a", "h"}],
            )
            self.assertEqual(lsc.entity.map, {"a": 0, "c": 1, "f": 2, "h": 3})

        train_index = np.array(["a", "c", "e"])
        test_index = np.array(["b", "d", "f"])
        with self.subTest(train=train_index, test=test_index):
            splitter = valsplit.Holdout.CustomHold(train_index, test_index)
            splitter._train_index = splitter.custom_train_index
            splitter._test_index = splitter.custom_test_index
            self.lsc.valsplit = splitter
            lsc = self.lsc.apply(
                filters.LabelsetRangeFilterTrainTestPos(min_val=1),
                inplace=False,
            )
            self.assertEqual(lsc.label_ids, ["Group1", "Group3", "Group4"])
            self.assertEqual(
                lsc.prop["Labelset"],
                [{"a", "b", "c"}, {"e", "f", "g"}, {"a", "f", "c"}],
            )
            self.assertEqual(
                lsc.entity.map,
                {"a": 0, "b": 1, "c": 2, "e": 3, "f": 4, "g": 5},
            )

        train_index = np.array(["a", "d"])
        valid_index = np.array(["b", "e"])
        test_index = np.array(["c", "f"])
        with self.subTest(
            train=train_index,
            test=test_index,
            valid=valid_index,
        ):
            splitter = valsplit.Holdout.CustomHold(
                train_index,
                test_index,
                valid_index,
            )
            splitter._train_index = splitter.custom_train_index
            splitter._test_index = splitter.custom_test_index
            splitter._valid_index = splitter.custom_valid_index
            self.lsc.valsplit = splitter
            lsc = self.lsc.apply(
                filters.LabelsetRangeFilterTrainTestPos(min_val=1),
                inplace=False,
            )
            self.assertEqual(lsc.label_ids, ["Group1"])
            self.assertEqual(lsc.prop["Labelset"], [{"a", "b", "c"}])
            self.assertEqual(lsc.entity.map, {"a": 0, "b": 1, "c": 2})

    def test_NegativeGeneratorHypergeom(self):
        # p-val threshold set to 0.5 since most large,
        # group1-group4 smallest with pval = 0.286
        self.lsc.apply(
            filters.NegativeGeneratorHypergeom(p_thresh=0.5),
            inplace=True,
        )
        # test wheter negative selected correctly for group1,
        # 'f' should be excluded due to sim with group2
        self.assertEqual(self.lsc.get_negative("Group1"), {"d", "e", "g", "h"})

        # increase p-val thtreshold to 0.7 will also include group2 and group3,
        # where pval = 0.643
        self.lsc.apply(
            filters.NegativeGeneratorHypergeom(p_thresh=0.7),
            inplace=True,
        )
        self.assertEqual(self.lsc.get_negative("Group1"), {"e", "g"})

        # set p-val threshold to be greater than 1 -> no negative left
        self.lsc.apply(
            filters.NegativeGeneratorHypergeom(p_thresh=1.1),
            inplace=True,
        )
        self.assertEqual(self.lsc.get_negative("Group1"), set())


if __name__ == "__main__":
    unittest.main()
