import unittest

import numpy as np
from NLEval.graph import FeatureVec
from NLEval.util import idhandler
from NLEval.util.exceptions import IDExistsError
from NLEval.util.exceptions import IDNotExistError


class TestIDmap(unittest.TestCase):
    def setUp(self):
        self.idmap = idhandler.IDmap()
        self.idmap.add_id("a")

    def tearDown(self):
        self.idmap = None

    def test_size(self):
        self.assertEqual(self.idmap.size, 1)
        self.idmap.add_id("b")
        self.assertEqual(self.idmap.size, 2)

    def test_map(self):
        self.assertEqual(self.idmap.map, {"a": 0})
        self.idmap.add_id("b")
        self.assertEqual(self.idmap.map, {"a": 0, "b": 1})

    def test_lst(self):
        self.assertEqual(self.idmap.lst, ["a"])
        self.idmap.add_id("b")
        self.assertEqual(self.idmap.lst, ["a", "b"])

    def test_add_id(self):
        self.assertRaises(TypeError, self.idmap.add_id, (1, 2, 3))
        self.assertRaises(TypeError, self.idmap.add_id, [1, 2, 3])
        self.assertRaises(TypeError, self.idmap.add_id, 10)
        self.assertRaises(IDExistsError, self.idmap.add_id, "a")
        self.idmap.add_id("10")
        self.idmap.add_id("10.1")
        self.idmap.add_id("abc")
        self.assertEqual(self.idmap.lst, ["a", "10", "10.1", "abc"])

    def test_getitem(self):
        self.assertEqual(self.idmap["a"], 0)

    def test_getID(self):
        self.assertEqual(self.idmap.getID(0), "a")

    def test_IDary2idxary(self):
        self.idmap.add_id("b")
        self.assertEqual(self.idmap[["b", "a"]][0], 1)
        self.assertEqual(self.idmap[["b", "a"]][1], 0)
        self.assertEqual(self.idmap[np.array(["a", "b"])][0], 0)
        self.assertEqual(self.idmap[np.array(["a", "b"])][1], 1)

    def test_contains(self):
        self.assertTrue("a" in self.idmap)
        self.assertFalse("b" in self.idmap)
        self.idmap.add_id("b")
        self.assertTrue("b" in self.idmap)

    def test_eq(self):
        self.idmap.add_id("b")
        idmap = idhandler.IDmap()
        idmap.add_id("b")
        idmap.add_id("a")
        self.assertTrue(self.idmap == idmap)
        self.idmap.add_id("c")
        self.idmap.add_id("d")
        idmap.add_id("d")
        self.assertFalse(self.idmap == idmap)
        idmap.add_id("c")
        self.assertTrue(self.idmap == idmap)

    def test_iter(self):
        self.idmap.add_id("b")
        self.idmap.add_id("x")
        lst = ["a", "b", "x"]
        for i, j in enumerate(self.idmap):
            with self.subTest(i=i):
                self.assertEqual(j, lst[i])

    def test_copy(self):
        idmap_shallow_copy = self.idmap
        idmap_deep_copy = self.idmap.copy()
        # shallow
        self.idmap.add_id("b")
        self.assertEqual(idmap_shallow_copy, self.idmap)
        # deep
        self.assertNotEqual(idmap_deep_copy, self.idmap)

    def test_pop_id(self):
        self.idmap.add_id("b")
        self.idmap.add_id("c")
        self.assertEqual(self.idmap.lst, ["a", "b", "c"])
        self.assertEqual(self.idmap.map, {"a": 0, "b": 1, "c": 2})
        self.assertRaises(IDNotExistError, self.idmap.pop_id, "d")
        self.assertEqual(self.idmap.pop_id("b"), 1)
        # make sure both lst and data poped
        self.assertEqual(self.idmap.lst, ["a", "c"])
        # make sure data updated with new mapping
        self.assertEqual(self.idmap.map, {"a": 0, "c": 1})

    def test_add(self):
        idmap = idhandler.IDmap()
        idmap.add_id("b")
        idmap_combined = self.idmap + idmap
        self.assertEqual(idmap_combined.map, {"a": 0, "b": 1})
        self.assertEqual(idmap_combined.lst, ["a", "b"])

    def test_sub(self):
        idmap = self.idmap.copy()
        idmap.add_id("b")
        self.idmap.add_id("c")
        diff = idmap - self.idmap
        self.assertEqual(diff.map, {"b": 0})
        self.assertEqual(diff.lst, ["b"])
        diff = self.idmap - idmap
        self.assertEqual(diff.map, {"c": 0})
        self.assertEqual(diff.lst, ["c"])

    def test_reset(self):
        idmap = self.idmap.copy()

        self.assertEqual(idmap.lst, ["a"])
        self.assertEqual(idmap.map, {"a": 0})

        idmap.reset(["b", "c"])
        self.assertEqual(idmap.lst, ["b", "c"])
        self.assertEqual(idmap.map, {"b": 0, "c": 1})

        idmap.reset()
        self.assertEqual(idmap.lst, [])
        self.assertEqual(idmap.map, {})


class TestIDmapAlign(unittest.TestCase):
    def setUp(self):
        self.ids1 = ["a", "b", "c", "d"]
        self.ids2 = ["c", "b", "a", "e", "f"]
        self.ids_intersection = ["a", "b", "c"]
        self.ids_union = ["a", "b", "c", "d", "e", "f"]

        self.ids1_map = {"a": 0, "b": 1, "c": 2, "d": 3}
        self.ids2_map = {"c": 0, "b": 1, "a": 2, "e": 3, "f": 4}
        self.ids_intersection_map = {"a": 0, "b": 1, "c": 2}
        self.ids_union_map = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5}

        self.mat1 = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
        self.mat2 = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])

        self.fvec1 = FeatureVec.from_mat(self.mat1, self.ids1)
        self.fvec2 = FeatureVec.from_mat(self.mat2, self.ids2)

    def test_align_raises(self):
        idmap = self.fvec1.idmap.copy()
        self.assertRaises(TypeError, idmap.align, self.ids1)
        self.assertRaises(ValueError, idmap.align, None)

        # Aligned with idmap with no common IDs
        new_idmap = idhandler.IDmap.from_list(["e", "f"])
        self.assertRaises(ValueError, idmap.align, new_idmap, join="right")
        self.assertRaises(ValueError, idmap.align, new_idmap, join="left")

    def test_align_right(self):
        idmap1 = self.fvec1.idmap.copy()
        idmap2 = self.fvec2.idmap.copy()

        self.assertEqual(idmap1.lst, self.ids1)
        self.assertEqual(idmap2.lst, self.ids2)

        right_idx, left_idx = idmap1.align(idmap2, join="right", update=False)
        self.assertEqual(idmap1.lst, self.ids2)
        self.assertEqual(idmap1.map, self.ids2_map)
        self.assertEqual(idmap2.lst, self.ids2)
        self.assertEqual(idmap2.map, self.ids2_map)

        new_mat1 = np.zeros((len(self.ids2), self.mat1.shape[1]))
        new_mat1[right_idx] = self.mat1[left_idx]
        self.assertEqual(
            new_mat1.astype(int).tolist(),
            [[2, 3, 4], [1, 2, 3], [0, 1, 2], [0, 0, 0], [0, 0, 0]],
        )

    def test_align_left(self):
        idmap1 = self.fvec1.idmap.copy()
        idmap2 = self.fvec2.idmap.copy()

        idmap1.align(idmap2, join="left", update=False)
        self.assertEqual(idmap1.lst, self.ids1)
        self.assertEqual(idmap1.map, self.ids1_map)
        self.assertEqual(idmap2.lst, self.ids2)
        self.assertEqual(idmap2.map, self.ids2_map)

        right_idx, left_idx = idmap1.align(idmap2, join="left", update=True)
        self.assertEqual(idmap1.lst, self.ids1)
        self.assertEqual(idmap1.map, self.ids1_map)
        self.assertEqual(idmap2.lst, self.ids1)
        self.assertEqual(idmap2.map, self.ids1_map)

        new_mat2 = np.zeros((len(self.ids1), self.mat2.shape[1]))
        new_mat2[right_idx] = self.mat2[left_idx]
        self.assertEqual(
            new_mat2.astype(int).tolist(),
            [[2, 3], [1, 2], [0, 1], [0, 0]],
        )

    def test_align_intersection(self):
        idmap1 = self.fvec1.idmap.copy()
        idmap2 = self.fvec2.idmap.copy()

        idmap1.align(idmap2, join="intersection", update=False)
        self.assertEqual(idmap1.lst, self.ids_intersection)
        self.assertEqual(idmap1.map, self.ids_intersection_map)
        self.assertEqual(idmap2.lst, self.ids2)
        self.assertEqual(idmap2.map, self.ids2_map)

        idmap1 = self.fvec1.idmap.copy()
        left_idx, right_idx = idmap1.align(
            idmap2,
            join="intersection",
            update=True,
        )
        self.assertEqual(idmap1.lst, self.ids_intersection)
        self.assertEqual(idmap1.map, self.ids_intersection_map)
        self.assertEqual(idmap2.lst, self.ids_intersection)
        self.assertEqual(idmap2.map, self.ids_intersection_map)

        new_mat1 = np.zeros((len(self.ids_intersection), self.mat1.shape[1]))
        new_mat1[:] = self.mat1[left_idx]
        self.assertEqual(
            new_mat1.astype(int).tolist(),
            [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
        )

        new_mat2 = np.zeros((len(self.ids_intersection), self.mat2.shape[1]))
        new_mat2[:] = self.mat2[right_idx]
        self.assertEqual(
            new_mat2.astype(int).tolist(),
            [[2, 3], [1, 2], [0, 1]],
        )

    def test_align_union(self):
        idmap1 = self.fvec1.idmap.copy()
        idmap2 = self.fvec2.idmap.copy()

        idmap1.align(idmap2, join="union", update=False)
        self.assertEqual(idmap1.lst, self.ids_union)
        self.assertEqual(idmap1.map, self.ids_union_map)
        self.assertEqual(idmap2.lst, self.ids2)
        self.assertEqual(idmap2.map, self.ids2_map)

        idmap1 = self.fvec1.idmap.copy()
        left_idx, right_idx = idmap1.align(
            idmap2,
            join="union",
            update=True,
        )
        self.assertEqual(idmap1.lst, self.ids_union)
        self.assertEqual(idmap1.map, self.ids_union_map)
        self.assertEqual(idmap2.lst, self.ids_union)
        self.assertEqual(idmap2.map, self.ids_union_map)

        new_mat1 = np.zeros((len(self.ids_union), self.mat1.shape[1]))
        new_mat1[left_idx] = self.mat1
        self.assertEqual(
            new_mat1.astype(int).tolist(),
            [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [0, 0, 0], [0, 0, 0]],
        )

        new_mat2 = np.zeros((len(self.ids_union), self.mat2.shape[1]))
        new_mat2[right_idx] = self.mat2
        self.assertEqual(
            new_mat2.astype(int).tolist(),
            [[2, 3], [1, 2], [0, 1], [0, 0], [3, 4], [4, 5]],
        )


if __name__ == "__main__":
    unittest.main()
