import unittest

import numpy as np
from NLEval.util import checkers, IDHandler
from NLEval.util.Exceptions import IDExistsError, IDNotExistError


class TestIDlst(unittest.TestCase):
    def setUp(self):
        self.IDlst1 = IDHandler.IDlst()
        self.IDlst2 = IDHandler.IDlst()
        self.IDlst3 = IDHandler.IDlst()
        self.IDlst4 = IDHandler.IDlst()
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
        self.assertTrue(IDHandler.IDlst() == self.IDlst1 - self.IDlst1)
        self.template_test_type_consistency(self.IDlst1.__sub__)

    def test_and(self):
        self.assertEqual(IDHandler.IDlst(), self.IDlst3 & self.IDlst4)
        self.assertEqual(self.IDlst4, self.IDlst1 & self.IDlst4)
        self.assertEqual(self.IDlst3, self.IDlst1 & self.IDlst3)

    def test_xor(self):
        self.assertEqual(self.IDlst1, self.IDlst3 ^ self.IDlst4)
        self.IDlst4.add_id("b")
        self.IDlst1.popID("b")
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
        self.assertEqual(IDHandler.IDlst().size, 0)

        idlst = IDHandler.IDlst()
        self.assertTrue(idlst.isempty())
        idlst.add_id("a")
        self.assertFalse(idlst.isempty())
        idlst.popID("a")
        self.assertTrue(idlst.isempty())

    def test_copy(self):
        idlst_shallow_copy = self.IDlst1
        idlst_deep_copy = self.IDlst1.copy()
        # shallow
        self.IDlst1.add_id("d")
        self.assertEqual(idlst_shallow_copy, self.IDlst1)
        # deep
        self.assertNotEqual(idlst_deep_copy, self.IDlst1)

    def test_popID(self):
        self.assertEqual(self.IDlst1.popID("c"), 2)
        self.assertEqual(self.IDlst1, self.IDlst3)
        self.assertRaises(IDNotExistError, self.IDlst1.popID, "c")
        self.assertRaises(TypeError, self.IDlst1.popID, 1)

    def test_add_id(self):
        self.assertEqual(self.IDlst1.lst, ["a", "b", "c"])
        # test basic add_id with string
        self.IDlst1.add_id("d")
        self.assertEqual(self.IDlst1.lst, ["a", "b", "c", "d"])
        # test add_id with with int --> TypeError
        self.assertRaises(TypeError, self.IDlst1.add_id, 10)
        # test add existing ID --> error
        self.assertRaises(IDExistsError, self.IDlst1.add_id, "a")

    def test_getID(self):
        for idx, ID in enumerate(self.lst):
            self.assertEqual(self.IDlst1.getID(idx), ID)
        # test type check
        self.assertRaises(TypeError, self.IDlst1.getID, "asdf")

    def test_from_list(self):
        idlst = IDHandler.IDlst.from_list(self.lst)
        self.assertEqual(idlst, self.IDlst1)
        # test type check
        tpl = ("a", "b", "c")
        self.assertRaises(TypeError, IDHandler.IDlst.from_list, tpl)
        # test redundant input
        lst = ["a", "b", "c", "a"]
        self.assertRaises(IDExistsError, IDHandler.IDlst.from_list, lst)

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


class TestIDmap(unittest.TestCase):
    def setUp(self):
        self.idmap = IDHandler.IDmap()
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
        idmap = IDHandler.IDmap()
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

    def test_popID(self):
        self.idmap.add_id("b")
        self.idmap.add_id("c")
        self.assertEqual(self.idmap.lst, ["a", "b", "c"])
        self.assertEqual(self.idmap.map, {"a": 0, "b": 1, "c": 2})
        self.assertRaises(IDNotExistError, self.idmap.popID, "d")
        self.assertEqual(self.idmap.popID("b"), 1)
        # make sure both lst and data poped
        self.assertEqual(self.idmap.lst, ["a", "c"])
        # make sure data updated with new mapping
        self.assertEqual(self.idmap.map, {"a": 0, "c": 1})

    def test_add(self):
        idmap = IDHandler.IDmap()
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


class TestIDprop(unittest.TestCase):
    def setUp(self):
        self.IDprop1 = IDHandler.IDprop()
        self.IDprop2 = IDHandler.IDprop()

    def test_eq(self):
        # test if two object have same set of IDs
        self.IDprop1.add_id("a")
        self.IDprop1.add_id("b")
        self.IDprop2.add_id("a")
        self.assertNotEqual(self.IDprop1, self.IDprop2)
        # test if two object have same set of props
        self.IDprop2.add_id("b")
        self.assertEqual(self.IDprop1, self.IDprop2)
        self.IDprop1._prop = {"p1": [None, None], "p2": [None, None]}
        self.assertNotEqual(self.IDprop1, self.IDprop2)
        self.IDprop2._prop = {"p2": [None, None]}
        self.assertNotEqual(self.IDprop1, self.IDprop2)
        self.IDprop2._prop = {"p2": [None, None], "p1": [None, None]}
        self.assertEqual(self.IDprop1, self.IDprop2)
        # test if two object have same prop values
        self.IDprop1._prop = {"p1": [1, None], "p2": [None, 2]}
        self.assertNotEqual(self.IDprop1, self.IDprop2)
        self.IDprop2._prop = {"p2": [1, None], "p1": [None, 2]}
        self.assertNotEqual(self.IDprop1, self.IDprop2)
        self.IDprop2._prop = {"p2": [None, 2], "p1": [1, None]}
        self.assertEqual(self.IDprop1, self.IDprop2)
        # test type check
        self.assertRaises(TypeError, self.IDprop1, IDHandler.IDlst())
        self.assertRaises(TypeError, self.IDprop1, IDHandler.IDmap())
        self.assertRaises(TypeError, self.IDprop1, ["a", "b", "c"])

    def test_propLst(self):
        self.IDprop1.newProp("x")
        self.IDprop1.newProp("y")
        self.IDprop1.newProp("z")
        self.assertEqual(self.IDprop1.propLst, ["x", "y", "z"])

    def test_newProp(self):
        # test property name type check
        self.assertRaises(TypeError, self.IDprop1.newProp, 10)
        self.IDprop1.newProp("10")
        # test property existance check
        self.assertRaises(IDExistsError, self.IDprop1.newProp, "10")
        # test type consistency between default value and type
        self.assertRaises(
            TypeError,
            self.IDprop1.newProp,
            "x",
            default_val=int(10),
            default_type=float,
        )
        self.IDprop1.newProp("x", default_val=int(10), default_type=int)
        # test newProp on empty object
        self.IDprop1.add_id("a")
        self.IDprop1.add_id("b")
        # test newProp on object with some IDs
        self.IDprop2.add_id("a")
        self.IDprop2.add_id("b")
        self.IDprop2.newProp("x", default_val=int(10), default_type=int)
        self.assertEqual(self.IDprop1._prop["x"], self.IDprop2._prop["x"])
        # test deepcopy of default val
        self.IDprop1.newProp("y", default_val=[], default_type=list)
        self.IDprop1._prop["y"][0].append(1)
        self.assertEqual(self.IDprop1._prop["y"], [[1], []])
        # test if default values and types set correctly
        with self.subTest(
            mst="Check if default properties values set correctly",
        ):
            self.assertEqual(
                self.IDprop1.prop_default_val,
                {"10": None, "x": 10, "y": []},
            )
        with self.subTest(
            mst="Check if default properties types set correctly",
        ):
            self.assertEqual(
                self.IDprop1.prop_default_type,
                {"10": None, "x": type(int()), "y": type(list())},
            )

    def test_setProp(self):
        self.IDprop1.add_id("a")
        self.IDprop1.newProp("x", 1, int)
        # test wrong ID type --> TypeError
        self.assertRaises(TypeError, self.IDprop1.setProp, 1, "x", 10)
        # test not exist ID --> IDNotExistError
        self.assertRaises(IDNotExistError, self.IDprop1.setProp, "b", "x", 10)
        # test wrong prop name type --> TypeError
        self.assertRaises(TypeError, self.IDprop1.setProp, "a", 1, 10)
        # test not exist prop name --> IDNotExistError
        self.assertRaises(IDNotExistError, self.IDprop1.setProp, "a", "y", 10)
        # test wrong prop val type --> TypeError
        self.assertRaises(TypeError, self.IDprop1.setProp, "a", "x", "10")
        self.assertRaises(TypeError, self.IDprop1.setProp, "a", "x", 10.0)
        # test if correct val set
        self.IDprop1.setProp("a", "x", 10)
        self.assertEqual(self.IDprop1.prop, {"x": [10]})
        self.IDprop1.add_id("b")
        self.IDprop1.setProp("b", "x", 34)
        self.assertEqual(self.IDprop1.prop, {"x": [10, 34]})

    def test_getProp(self):
        self.IDprop1.add_id("a")
        self.IDprop1.newProp("x", 10, int)
        # test wrong ID type --> TypeError
        self.assertRaises(TypeError, self.IDprop1.getProp, 1, "x")
        # test not exist ID value --> IDNotExistError
        self.assertRaises(IDNotExistError, self.IDprop1.getProp, "b", "x")
        # test wrong prop name type --> TypeError
        self.assertRaises(TypeError, self.IDprop1.getProp, "a", 1)
        # test not exist prop name --> IDNotExistError
        self.assertRaises(IDNotExistError, self.IDprop1.getProp, "a", "y")
        # test if correct val retrieved
        self.assertEqual(self.IDprop1.getProp("a", "x"), 10)
        self.IDprop1.setProp("a", "x", 20)
        self.assertEqual(self.IDprop1.getProp("a", "x"), 20)

    def test_delProp(self):
        self.IDprop1.newProp("x", 1, int)
        self.IDprop1.newProp("y", "1", str)
        self.IDprop1.newProp("z", [1], list)
        self.assertEqual(
            self.IDprop1.prop_default_val,
            {"x": 1, "y": "1", "z": [1]},
        )
        self.assertEqual(
            self.IDprop1.prop_default_type,
            {"x": int, "y": str, "z": list},
        )
        self.assertEqual(self.IDprop1.prop, {"x": [], "y": [], "z": []})
        # test if property deleted properly on empty ID list
        self.IDprop1.delProp("y")
        self.assertEqual(self.IDprop1.prop_default_val, {"x": 1, "z": [1]})
        self.assertEqual(self.IDprop1.prop_default_type, {"x": int, "z": list})
        self.assertEqual(self.IDprop1.prop, {"x": [], "z": []})
        # test wrong ID type --> TypeError
        self.assertRaises(TypeError, self.IDprop1.delProp, 1)
        self.assertRaises(TypeError, self.IDprop1.delProp, [1, 2])
        # test not exist prop name --> IDNotExistError
        self.assertRaises(IDNotExistError, self.IDprop1.delProp, "X")
        self.assertRaises(IDNotExistError, self.IDprop1.delProp, "Z")
        # test if property deleted properly on filled IDlist
        self.IDprop1.add_id("a")
        self.IDprop1.add_id("b")
        self.IDprop1.add_id("c")
        self.assertEqual(
            self.IDprop1.prop,
            {"x": [1, 1, 1], "z": [[1], [1], [1]]},
        )
        self.IDprop1.delProp("z")
        self.assertEqual(self.IDprop1.prop, {"x": [1, 1, 1]})
        self.IDprop1.delProp("x")
        self.assertEqual(self.IDprop1.prop, {})

    def test_getAllProp(self):
        self.IDprop1.add_id("a")
        self.IDprop1.newProp("x", 10, int)
        self.IDprop1.newProp("y", 20.0, float)
        # test wrong ID type --> TypeError
        self.assertRaises(TypeError, self.IDprop1.getAllProp, 1)
        # test wrong ID val --> IDNotExistError
        self.assertRaises(IDNotExistError, self.IDprop1.getAllProp, "b")
        # test if all prop val retrieved correctly
        self.assertEqual(self.IDprop1.getAllProp("a"), {"x": 10, "y": 20.0})

    def test_popID(self):
        self.IDprop1.newProp("x", 1, int)
        self.IDprop1.newProp("y", "1", str)
        self.IDprop1.add_id("a")
        self.IDprop1.add_id("b", {"x": 2, "y": "2"})
        self.IDprop1.add_id("c", {"x": 3, "y": "3"})
        # test wrong ID type --> TypeError
        self.assertRaises(TypeError, self.IDprop1.popID, 1)
        # test wrong ID val --> IDNotExistError
        self.assertRaises(IDNotExistError, self.IDprop1.popID, "d")
        # test if poped correctly
        self.assertEqual(self.IDprop1.popID("b"), 1)
        self.assertEqual(self.IDprop1.lst, ["a", "c"])
        self.assertEqual(self.IDprop1.propLst, ["x", "y"])
        self.assertEqual(self.IDprop1._prop, {"x": [1, 3], "y": ["1", "3"]})
        self.IDprop1.popID("a")
        self.IDprop1.popID("c")
        self.assertEqual(self.IDprop1._prop, {"x": [], "y": []})

    def test_add_id(self):
        self.IDprop1.newProp("x", 1, int)
        self.IDprop1.newProp("y", "1", str)
        # no specification of properties, use all default values
        self.IDprop1.add_id("a")
        self.assertEqual(self.IDprop1.getProp("a", "x"), 1)
        self.assertEqual(self.IDprop1.getProp("a", "y"), "1")
        # fully specified properties
        self.IDprop1.add_id("b", {"x": 2, "y": "2"})
        self.assertEqual(self.IDprop1.getProp("b", "x"), 2)
        self.assertEqual(self.IDprop1.getProp("b", "y"), "2")
        # test wrong ID type --> TypeError
        self.assertRaises(TypeError, self.IDprop1.add_id, (1, 2, 3))
        # test addd existed ID --> IDExistsError
        self.assertRaises(IDExistsError, self.IDprop1.add_id, "a")
        # test wrong prop type --> TypeError
        self.assertRaises(TypeError, self.IDprop1.add_id, "c", ("x", "y"))
        # test wrong prop keys --> IDNotExistError
        self.assertRaises(
            IDNotExistError,
            self.IDprop1.add_id,
            "c",
            {"x": 3, "z": "3"},
        )
        # test wrong prop val type --> TypeError
        self.assertRaises(
            TypeError,
            self.IDprop1.add_id,
            "c",
            {"x": 3, "y": 3.0},
        )
        # test partial specification, use default values for unspecified properties
        self.IDprop1.add_id(
            "c",
            {"x": 3},
        )  # only 'x' specified, 'y' will be default
        self.assertEqual(self.IDprop1.getProp("c", "x"), 3)
        self.assertEqual(self.IDprop1.getProp("c", "y"), "1")
        # test empty property dictionary, should be same as no `prop=None`
        self.IDprop1.add_id("d", {})
        self.assertEqual(self.IDprop1.getProp("d", "x"), 1)
        self.assertEqual(self.IDprop1.getProp("d", "y"), "1")
        # test if deepcopy used for fill in default properties for existing entities
        self.IDprop2.add_id("a")
        self.IDprop2.add_id("b")
        self.IDprop2.newProp("x", list())
        self.IDprop2.getProp("a", "x").append(1)
        self.assertEqual(self.IDprop2.getProp("a", "x"), [1])
        self.assertEqual(self.IDprop2.getProp("b", "x"), [])
        # test if deepcopy used for filling in missing properties of new ID
        self.IDprop2.add_id("c")
        self.IDprop2.add_id("d")
        self.IDprop2.getProp("c", "x").append(2)
        self.assertEqual(self.IDprop2.getProp("a", "x"), [1])
        self.assertEqual(self.IDprop2.getProp("b", "x"), [])
        self.assertEqual(self.IDprop2.getProp("c", "x"), [2])
        self.assertEqual(self.IDprop2.getProp("d", "x"), [])


class TestCheckers(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        n = 10
        self.n = n
        self.n_str = str(n)
        self.n_int = int(n)
        self.n_npint = int(n)
        self.n_npint64 = np.int64(n)
        self.n_float = float(n)
        self.n_npfloat = float(n)
        self.n_npfloat128 = np.float128(n)
        self.n_int_tuple = (int(n), int(n), int(n))
        self.n_int_lst = [int(n), int(n), int(n)]
        self.n_int_npary = np.array([n, n, n], dtype=int)
        self.n_float_tuple = (float(n), float(n), float(n))
        self.n_float_lst = [float(n), float(n), float(n)]
        self.n_float_npary = np.array([n, n, n], dtype=float)

    def test_INT_TYPE(self):
        self.assertIsInstance(self.n_int, checkers.INT_TYPE)
        self.assertIsInstance(self.n_npint, checkers.INT_TYPE)
        self.assertIsInstance(self.n_npint64, checkers.INT_TYPE)
        self.assertNotIsInstance(self.n_float, checkers.INT_TYPE)
        self.assertNotIsInstance(self.n_npfloat, checkers.INT_TYPE)
        self.assertNotIsInstance(self.n_npfloat128, checkers.INT_TYPE)

    def test_FLOAT_TYPE(self):
        self.assertNotIsInstance(self.n_int, checkers.FLOAT_TYPE)
        self.assertNotIsInstance(self.n_npint, checkers.FLOAT_TYPE)
        self.assertNotIsInstance(self.n_npint64, checkers.FLOAT_TYPE)
        self.assertIsInstance(self.n_float, checkers.FLOAT_TYPE)
        self.assertIsInstance(self.n_npfloat, checkers.FLOAT_TYPE)
        self.assertIsInstance(self.n_npfloat128, checkers.FLOAT_TYPE)

    def test_ITERABLE_TYPE(self):
        n_int_tuple = (1, 2, 3)
        n_int_lst = [1, 2, 3]
        n_int_ary = np.array([1, 2, 3])
        self.assertIsInstance(n_int_tuple, checkers.ITERABLE_TYPE)
        self.assertIsInstance(n_int_lst, checkers.ITERABLE_TYPE)
        self.assertIsInstance(n_int_ary, checkers.ITERABLE_TYPE)

    def test_checkType(self):
        checkers.checkType("n_int", int, self.n_int)
        self.assertRaises(
            TypeError,
            checkers.checkType,
            "n_int",
            int,
            self.n_float,
        )
        checkers.checkType("n_float", float, self.n_float)
        self.assertRaises(
            TypeError,
            checkers.checkType,
            "n_float",
            float,
            self.n_int,
        )
        checkers.checkType("n_str", str, self.n_str)

    def test_checkTypeInIterable(self):
        checkers.checkTypesInIterable(
            "n_int_tuple",
            checkers.INT_TYPE,
            self.n_int_tuple,
        )
        checkers.checkTypesInIterable(
            "n_int_lst",
            checkers.INT_TYPE,
            self.n_int_lst,
        )
        checkers.checkTypesInIterable(
            "n_int_npary",
            checkers.INT_TYPE,
            self.n_int_npary,
        )
        self.assertRaises(
            TypeError,
            checkers.checkTypesInIterable,
            "n_int_tuple",
            checkers.FLOAT_TYPE,
            self.n_int_tuple,
        )
        self.assertRaises(
            TypeError,
            checkers.checkTypesInIterable,
            "n_int_lst",
            checkers.FLOAT_TYPE,
            self.n_int_lst,
        )
        self.assertRaises(
            TypeError,
            checkers.checkTypesInIterable,
            "n_int_npary",
            checkers.FLOAT_TYPE,
            self.n_int_npary,
        )
        checkers.checkTypesInIterable(
            "n_float_tuple",
            checkers.FLOAT_TYPE,
            self.n_float_tuple,
        )
        checkers.checkTypesInIterable(
            "n_float_lst",
            checkers.FLOAT_TYPE,
            self.n_float_lst,
        )
        checkers.checkTypesInIterable(
            "n_float_npary",
            checkers.FLOAT_TYPE,
            self.n_float_npary,
        )
        self.assertRaises(
            TypeError,
            checkers.checkTypesInIterable,
            "n_float_tuple",
            checkers.INT_TYPE,
            self.n_float_tuple,
        )
        self.assertRaises(
            TypeError,
            checkers.checkTypesInIterable,
            "n_float_lst",
            checkers.INT_TYPE,
            self.n_float_lst,
        )
        self.assertRaises(
            TypeError,
            checkers.checkTypesInIterable,
            "n_float_npary",
            checkers.INT_TYPE,
            self.n_float_npary,
        )

    def test_checkTypeErrNone(self):
        self.assertRaises(ValueError, checkers.checkTypeErrNone, "n", int, None)
        self.assertRaises(
            ValueError,
            checkers.checkTypeErrNone,
            "n",
            float,
            None,
        )
        self.assertRaises(ValueError, checkers.checkTypeErrNone, "n", str, None)

    def test_checkTypeAllowNone(self):
        checkers.checkTypeAllowNone("n", int, None)
        checkers.checkTypeAllowNone("n", float, None)
        checkers.checkTypeAllowNone("n", str, None)

    def test_checkNumpyArrayNDim(self):
        ary1 = np.ones(2)
        ary2 = np.ones((2, 2))
        ary3 = np.ones((2, 2, 2))
        checkers.checkNumpyArrayNDim("ary1", 1, ary1)
        checkers.checkNumpyArrayNDim("ary2", 2, ary2)
        checkers.checkNumpyArrayNDim("ary3", 3, ary3)
        self.assertRaises(
            ValueError,
            checkers.checkNumpyArrayNDim,
            "ary1",
            4,
            ary1,
        )
        self.assertRaises(
            ValueError,
            checkers.checkNumpyArrayNDim,
            "ary2",
            4,
            ary2,
        )
        self.assertRaises(
            ValueError,
            checkers.checkNumpyArrayNDim,
            "ary3",
            4,
            ary3,
        )

    def test_checkNumpyArrayShape(self):
        ary1 = np.ones(2)
        ary2 = np.ones((2, 2))
        ary3 = np.ones((2, 2, 2))
        checkers.checkNumpyArrayShape("ary1", 2, ary1)
        checkers.checkNumpyArrayShape("ary1", (2,), ary1)
        checkers.checkNumpyArrayShape("ary2", (2, 2), ary2)
        checkers.checkNumpyArrayShape("ary3", (2, 2, 2), ary3)
        self.assertRaises(
            ValueError,
            checkers.checkNumpyArrayShape,
            "ary1",
            1,
            ary1,
        )
        self.assertRaises(
            ValueError,
            checkers.checkNumpyArrayShape,
            "ary1",
            (
                1,
                2,
            ),
            ary1,
        )
        self.assertRaises(
            ValueError,
            checkers.checkNumpyArrayShape,
            "ary2",
            (2, 1),
            ary2,
        )
        self.assertRaises(
            ValueError,
            checkers.checkNumpyArrayShape,
            "ary3",
            1,
            ary3,
        )

    def test_checkNumpyArrayIsNumeric(self):
        # test numpy numeric array --> success
        ary1 = np.random.random(3).astype(int)
        ary2 = np.random.random((3, 3)).astype(float)
        ary3 = np.random.random(5)
        checkers.checkNumpyArrayIsNumeric("ar1", ary1)
        checkers.checkNumpyArrayIsNumeric("ar2", ary2)
        checkers.checkNumpyArrayIsNumeric("ar3", ary3)

        # test numpy string array --> error
        ary1 = np.ones(3, dtype=str)
        ary2 = np.ones((3, 3), dtype=str)
        self.assertRaises(
            TypeError,
            checkers.checkNumpyArrayIsNumeric,
            "ar1",
            ary1,
        )
        self.assertRaises(
            TypeError,
            checkers.checkNumpyArrayIsNumeric,
            "ar2",
            ary2,
        )

        # test list --> error
        lst = [1, 2, 3]
        self.assertRaises(
            TypeError,
            checkers.checkNumpyArrayIsNumeric,
            "lst",
            lst,
        )

        # test string --> error
        string = "asdfgasdglafknklasd"
        self.assertRaises(
            TypeError,
            checkers.checkNumpyArrayIsNumeric,
            "string",
            string,
        )


if __name__ == "__main__":
    unittest.main()
