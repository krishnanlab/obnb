import unittest

import numpy as np

from nleval.exception import IDExistsError, IDNotExistError
from nleval.util import idhandler


class TestIDlst(unittest.TestCase):
    def setUp(self):
        self.IDlst1 = idhandler.IDlst()
        self.IDlst2 = idhandler.IDlst()
        self.IDlst3 = idhandler.IDlst()
        self.IDlst4 = idhandler.IDlst()
        self.lst = ["a", "b", "c"]
        for i in self.lst:
            self.IDlst1.add_id(i)
            self.IDlst2.add_id(i)
        self.IDlst3.add_id("a")
        self.IDlst3.add_id("b")
        self.IDlst4.add_id("c")

    def test_iter(self):
        for i, j in zip(self.IDlst1, self.lst):
            self.assertEqual(i, j)

    def template_test_type_consistency(self, fun):
        self.assertRaises(TypeError, fun, self.lst)
        self.assertRaises(TypeError, fun, 10)

    def test_eq(self):
        self.assertTrue(self.IDlst1 == self.IDlst2)
        self.assertFalse(self.IDlst1 != self.IDlst2)
        self.IDlst1.add_id("d")
        self.assertTrue(self.IDlst1 != self.IDlst2)
        self.IDlst2.add_id("d")
        self.assertTrue(self.IDlst1 == self.IDlst2)
        # test if orders matter
        self.IDlst1.add_id("e")
        self.IDlst1.add_id("f")
        self.IDlst2.add_id("f")
        self.IDlst2.add_id("e")
        self.assertTrue(self.IDlst1 == self.IDlst2)
        self.template_test_type_consistency(self.IDlst1.__eq__)

    def test_add(self):
        self.assertTrue(self.IDlst1 == self.IDlst3 + self.IDlst4)
        self.IDlst4.add_id("d")
        self.assertFalse(self.IDlst1 == self.IDlst3 + self.IDlst4)
        self.IDlst1.add_id("d")
        self.assertTrue(self.IDlst1 == self.IDlst3 + self.IDlst4)
        self.template_test_type_consistency(self.IDlst1.__add__)

    def test_sub(self):
        self.assertTrue(self.IDlst3 == self.IDlst1 - self.IDlst4)
        self.assertTrue(self.IDlst4 == self.IDlst1 - self.IDlst3)
        self.assertTrue(idhandler.IDlst() == self.IDlst1 - self.IDlst1)
        self.template_test_type_consistency(self.IDlst1.__sub__)

    def test_and(self):
        self.assertEqual(idhandler.IDlst(), self.IDlst3 & self.IDlst4)
        self.assertEqual(self.IDlst4, self.IDlst1 & self.IDlst4)
        self.assertEqual(self.IDlst3, self.IDlst1 & self.IDlst3)

    def test_xor(self):
        self.assertEqual(self.IDlst1, self.IDlst3 ^ self.IDlst4)
        self.IDlst4.add_id("b")
        self.IDlst1.pop_id("b")
        self.assertEqual(self.IDlst1, self.IDlst3 ^ self.IDlst4)

    def test_contains(self):
        for i in self.lst:
            self.assertTrue(i in self.IDlst1)
        self.assertTrue("c" not in self.IDlst3)
        # test type check
        self.assertRaises(TypeError, self.IDlst1.__contains__, 10)

    def test_getitem(self):
        for idx, ID in enumerate(self.lst):
            self.assertEqual(self.IDlst1[ID], idx)
        self.assertRaises(IDNotExistError, self.IDlst1.__getitem__, "d")
        self.assertTrue(all(self.IDlst1[self.lst] == np.array([0, 1, 2])))
        self.assertRaises(IDNotExistError, self.IDlst3.__getitem__, self.lst)
        self.assertRaises(TypeError, self.IDlst3.__getitem__, ["a", 0])

    def test_size(self):
        self.assertEqual(self.IDlst1.size, 3)
        self.assertEqual(self.IDlst3.size, 2)
        self.assertEqual(idhandler.IDlst().size, 0)

        idlst = idhandler.IDlst()
        self.assertTrue(idlst.isempty())
        idlst.add_id("a")
        self.assertFalse(idlst.isempty())
        idlst.pop_id("a")
        self.assertTrue(idlst.isempty())

    def test_copy(self):
        idlst_shallow_copy = self.IDlst1
        idlst_deep_copy = self.IDlst1.copy()
        # shallow
        self.IDlst1.add_id("d")
        self.assertEqual(idlst_shallow_copy, self.IDlst1)
        # deep
        self.assertNotEqual(idlst_deep_copy, self.IDlst1)

    def test_pop_id(self):
        self.assertEqual(self.IDlst1.pop_id("c"), 2)
        self.assertEqual(self.IDlst1, self.IDlst3)
        self.assertRaises(IDNotExistError, self.IDlst1.pop_id, "c")
        self.assertRaises(TypeError, self.IDlst1.pop_id, 1)

    def test_add_id(self):
        self.assertEqual(self.IDlst1.lst, ["a", "b", "c"])
        # test basic add_id with string
        self.IDlst1.add_id("d")
        self.assertEqual(self.IDlst1.lst, ["a", "b", "c", "d"])
        # test add_id with with int --> TypeError
        self.assertRaises(TypeError, self.IDlst1.add_id, 10)
        # test add existing ID --> error
        self.assertRaises(IDExistsError, self.IDlst1.add_id, "a")

    def test_get_id(self):
        for idx, ID in enumerate(self.lst):
            self.assertEqual(self.IDlst1.get_id(idx), ID)
        # test type check
        self.assertRaises(TypeError, self.IDlst1.get_id, "asdf")

    def test_get_ids(self):
        self.assertEqual(self.IDlst1.get_ids(range(3)), ["a", "b", "c"])
        self.assertEqual(self.IDlst1.get_ids([1, 2]), ["b", "c"])
        with self.assertRaises(TypeError):
            self.assertEqual(self.IDlst1.get_ids(["1", 2]), ["b", "c"])

    def test_from_list(self):
        idlst = idhandler.IDlst.from_list(self.lst)
        self.assertEqual(idlst, self.IDlst1)
        # test type check
        tpl = ("a", "b", "c")
        self.assertRaises(TypeError, idhandler.IDlst.from_list, tpl)
        # test redundant input
        lst = ["a", "b", "c", "a"]
        self.assertRaises(IDExistsError, idhandler.IDlst.from_list, lst)

    def test_update(self):
        # test type check
        self.assertRaises(TypeError, self.IDlst1.update, 1)
        self.assertRaises(TypeError, self.IDlst1.update, "a")
        # test type check in list
        self.assertRaises(TypeError, self.IDlst1.update, [1, 2, 3])
        self.assertRaises(TypeError, self.IDlst1.update, ["1", "2", 3])
        # test all overlap --> same as original
        self.assertEqual(self.IDlst1.update(self.lst), 0)
        self.assertEqual(self.IDlst1.lst, self.lst)
        # test partial overlap --> same as xor
        self.assertEqual(self.IDlst1.update(["b", "c", "d"]), 1)
        self.assertEqual(self.IDlst1.lst, ["a", "b", "c", "d"])
        # test not overlap --> same as or
        self.assertEqual(self.IDlst1.update(["e", "f"]), 2)
        self.assertEqual(self.IDlst1.lst, ["a", "b", "c", "d", "e", "f"])


if __name__ == "__main__":
    unittest.main()
