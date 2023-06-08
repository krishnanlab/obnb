import unittest

from obnb.label import LabelsetCollection, filters, split


class TestFilter(unittest.TestCase):
    def setUp(self):
        """
                a   b   c   d   e   f   g   h
        -------------------------------------
        Group1  x   x   x
        Group2      x       x
        Group3                  x   x   x
        Group4  x       x           x
        Group5  x                           x
        """
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

    def test_LabelsetPairwiseFilterJaccard(self):
        with self.subTest(min_val=0.9):
            lsc = self.lsc.apply(
                filters.LabelsetPairwiseFilterJaccard(
                    max_val=0.9,
                    size_constraint="larger",
                ),
                inplace=False,
            )
            self.assertEqual(
                lsc.label_ids,
                ["Group1", "Group2", "Group3", "Group4", "Group5"],
            )
        with self.subTest(min_val=0.4):
            lsc = self.lsc.apply(
                filters.LabelsetPairwiseFilterJaccard(
                    max_val=0.4,
                    size_constraint="larger",
                ),
                inplace=False,
            )
            self.assertEqual(
                lsc.label_ids,
                ["Group2", "Group3", "Group4", "Group5"],
            )
        with self.subTest(min_val=0.2):
            lsc = self.lsc.apply(
                filters.LabelsetPairwiseFilterJaccard(
                    max_val=0.2,
                    size_constraint="larger",
                ),
                inplace=False,
            )
            self.assertEqual(
                lsc.label_ids,
                ["Group2", "Group3", "Group5"],
            )
        with self.subTest(min_val=0):
            lsc = self.lsc.apply(
                filters.LabelsetPairwiseFilterJaccard(
                    max_val=0,
                    size_constraint="larger",
                ),
                inplace=False,
            )
            self.assertEqual(
                lsc.label_ids,
                ["Group2", "Group5"],
            )

    def test_LabelsetPairwiseFilterOverlap(self):
        with self.subTest(min_val=0.9):
            lsc = self.lsc.apply(
                filters.LabelsetPairwiseFilterOverlap(
                    max_val=0.9,
                    size_constraint="larger",
                ),
                inplace=False,
            )
            self.assertEqual(
                lsc.label_ids,
                ["Group1", "Group2", "Group3", "Group4", "Group5"],
            )

        with self.subTest(min_val=0.6):
            lsc = self.lsc.apply(
                filters.LabelsetPairwiseFilterOverlap(
                    max_val=0.6,
                    size_constraint="larger",
                ),
                inplace=False,
            )
            self.assertEqual(
                lsc.label_ids,
                ["Group2", "Group3", "Group4", "Group5"],
            )

        with self.subTest(min_val=0.3):
            lsc = self.lsc.apply(
                filters.LabelsetPairwiseFilterOverlap(
                    max_val=0.3,
                    size_constraint="larger",
                ),
                inplace=False,
            )
            self.assertEqual(
                lsc.label_ids,
                ["Group2", "Group5"],
            )

    def test_LabelsetRangeFilterSplit(self):
        # Setup mock properties for generating splits
        self.lsc.entity.new_property("test_property", 0, int)
        split_opts = {
            "property_converter": {
                j: i for i, j in enumerate(sorted(self.lsc.entity_ids))
            },
        }
        log_level = "INFO"

        # Train = [a, b], Test = [c, d, e, f, g, h], Group3 does not have any
        # positives in the training split, hence should be removed
        splitter = split.ThresholdPartition(2, **split_opts)
        with self.subTest(splitter=splitter):
            lsc = self.lsc.apply(
                filters.LabelsetRangeFilterSplit(
                    1,
                    splitter,
                    count_negatives=False,
                    log_level=log_level,
                ),
            )
            self.assertEqual(
                lsc.label_ids,
                ["Group1", "Group2", "Group4", "Group5"],
            )

        # Same as above, but take into account of negatives. Group1 is removed
        # due to the lack of negatives (all negatives in training set)
        splitter = split.ThresholdPartition(2, **split_opts)
        with self.subTest(splitter=splitter):
            lsc = self.lsc.apply(
                filters.LabelsetRangeFilterSplit(
                    1,
                    splitter,
                    count_negatives=True,
                    log_level=log_level,
                ),
            )
            self.assertEqual(
                lsc.label_ids,
                ["Group2", "Group4", "Group5"],
            )

        # Train = [a], Test = [b, c, d, e, f, g, h], Both Group2 and Group3 do
        # not have any positives in the training split, hence should be removed
        splitter = split.ThresholdPartition(1, **split_opts)
        with self.subTest(splitter=splitter):
            lsc = self.lsc.apply(
                filters.LabelsetRangeFilterSplit(
                    1,
                    splitter,
                    count_negatives=False,
                    log_level=log_level,
                ),
            )
            self.assertEqual(
                lsc.label_ids,
                ["Group1", "Group4", "Group5"],
            )

        splitter = split.RatioPartition(0.25, 0.75, **split_opts)
        with self.subTest(splitter=splitter):
            """Iteratively apply ratio filter until nothing changes.

            Notes:
                The dotted line indicates the position of the partitioning;
                'x' indicates the position of removed labels/entities due to
                filtering at current step.

            1) 1/3 split applied to the original y matrix:
                    1   2   3   4   5
                a   1   0   0   1   1
                b   1   1   0   0   0
                ---------------------
                c   1   0   0   1   0
                d   0   1   0   0   0
                e   0   0   1   0   0 x (only group3 uses e)
                f   0   0   1   1   0
                g   0   0   1   0   0 x (only group3 uses g)
                h   0   0   0   0   1
                            x

                Group3 removed since the first split do not have any positives,
                but we required the minimum number of positives to be min_val=1.

            2) 1/3 split applied to the new matrix:
                    1   2   4   5
                a   1   0   1   1
                -----------------
                b   1   1   0   0
                c   1   0   1   0
                d   0   1   0   0 x (only group2 uses d in the new matrix)
                f   0   0   1   0
                h   0   0   0   1
                        x

                Group2 removed since the first split do no have any positives.

            3) 1/3 split applied to the new matrix from 2)
                    1   4   5
                a   1   1   1
                -------------
                b   1   0   0
                c   1   1   0
                f   0   1   0
                h   0   0   1

                Nothing will be filtered out in this matrix since the split
                filtering criterion is met, i.e. at least one positives in
                each split.

            So our final filtered labelset would be:
                * Group1: a, b, c
                * Group4: a, c, f
                * Group5: a, h

            """
            # TODO: automate this recursion..
            lsc = self.lsc.copy()
            old_num_labels = None
            while len(lsc) != old_num_labels:
                old_num_labels = len(lsc)
                lsc = lsc.apply(
                    filters.LabelsetRangeFilterSplit(
                        1,
                        splitter,
                        count_negatives=False,
                        log_level=log_level,
                    ),
                )

            self.assertEqual(lsc.label_ids, ["Group1", "Group4", "Group5"])
            self.assertEqual(lsc.get_labelset("Group1"), {"a", "b", "c"})
            self.assertEqual(lsc.get_labelset("Group4"), {"a", "c", "f"})
            self.assertEqual(lsc.get_labelset("Group5"), {"a", "h"})

    def test_NegativeGeneratorHypergeom(self):
        # p-val threshold set to 0.5 since most large,
        # group1-group4 smallest with pval = 0.286
        self.lsc.apply(
            filters.NegativeGeneratorHypergeom(p_thresh=0.5),
            inplace=True,
        )
        # test whether negative selected correctly for group1,
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
