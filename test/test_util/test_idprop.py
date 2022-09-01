import unittest

from NLEval.exception import IDExistsError, IDNotExistError
from NLEval.util import idhandler


class TestIDprop(unittest.TestCase):
    def setUp(self):
        self.IDprop1 = idhandler.IDprop()
        self.IDprop2 = idhandler.IDprop()

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
        self.assertRaises(TypeError, self.IDprop1, idhandler.IDlst())
        self.assertRaises(TypeError, self.IDprop1, idhandler.IDmap())
        self.assertRaises(TypeError, self.IDprop1, ["a", "b", "c"])

    def test_properties(self):
        self.IDprop1.new_property("x")
        self.IDprop1.new_property("y")
        self.IDprop1.new_property("z")
        self.assertEqual(self.IDprop1.properties, ["x", "y", "z"])

    def test_new_property(self):
        # test property name type check
        self.assertRaises(TypeError, self.IDprop1.new_property, 10)
        self.IDprop1.new_property("10")
        # test property existance check
        self.assertRaises(IDExistsError, self.IDprop1.new_property, "10")
        # test type consistency between default value and type
        self.assertRaises(
            TypeError,
            self.IDprop1.new_property,
            "x",
            default_val=int(10),
            default_type=float,
        )
        self.IDprop1.new_property("x", default_val=int(10), default_type=int)
        # test new_property on empty object
        self.IDprop1.add_id("a")
        self.IDprop1.add_id("b")
        # test new_property on object with some IDs
        self.IDprop2.add_id("a")
        self.IDprop2.add_id("b")
        self.IDprop2.new_property("x", default_val=int(10), default_type=int)
        self.assertEqual(self.IDprop1._prop["x"], self.IDprop2._prop["x"])
        # test deepcopy of default val
        self.IDprop1.new_property("y", default_val=[], default_type=list)
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

    def test_set_property(self):
        self.IDprop1.add_id("a")
        self.IDprop1.new_property("x", 1, int)
        # test wrong ID type --> TypeError
        self.assertRaises(TypeError, self.IDprop1.set_property, 1, "x", 10)
        # test not exist ID --> IDNotExistError
        self.assertRaises(
            IDNotExistError,
            self.IDprop1.set_property,
            "b",
            "x",
            10,
        )
        # test wrong prop name type --> TypeError
        self.assertRaises(TypeError, self.IDprop1.set_property, "a", 1, 10)
        # test not exist prop name --> IDNotExistError
        self.assertRaises(
            IDNotExistError,
            self.IDprop1.set_property,
            "a",
            "y",
            10,
        )
        # test wrong prop val type --> TypeError
        self.assertRaises(TypeError, self.IDprop1.set_property, "a", "x", "10")
        self.assertRaises(TypeError, self.IDprop1.set_property, "a", "x", 10.0)
        # test if correct val set
        self.IDprop1.set_property("a", "x", 10)
        self.assertEqual(self.IDprop1.prop, {"x": [10]})
        self.IDprop1.add_id("b")
        self.IDprop1.set_property("b", "x", 34)
        self.assertEqual(self.IDprop1.prop, {"x": [10, 34]})

    def test_get_property(self):
        self.IDprop1.add_id("a")
        self.IDprop1.new_property("x", 10, int)
        # test wrong ID type --> TypeError
        self.assertRaises(TypeError, self.IDprop1.get_property, 1, "x")
        # test not exist ID value --> IDNotExistError
        self.assertRaises(IDNotExistError, self.IDprop1.get_property, "b", "x")
        # test wrong prop name type --> TypeError
        self.assertRaises(TypeError, self.IDprop1.get_property, "a", 1)
        # test not exist prop name --> IDNotExistError
        self.assertRaises(IDNotExistError, self.IDprop1.get_property, "a", "y")
        # test if correct val retrieved
        self.assertEqual(self.IDprop1.get_property("a", "x"), 10)
        self.IDprop1.set_property("a", "x", 20)
        self.assertEqual(self.IDprop1.get_property("a", "x"), 20)

    def test_remove_property(self):
        self.IDprop1.new_property("x", 1, int)
        self.IDprop1.new_property("y", "1", str)
        self.IDprop1.new_property("z", [1], list)
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
        self.IDprop1.remove_property("y")
        self.assertEqual(self.IDprop1.prop_default_val, {"x": 1, "z": [1]})
        self.assertEqual(self.IDprop1.prop_default_type, {"x": int, "z": list})
        self.assertEqual(self.IDprop1.prop, {"x": [], "z": []})
        # test wrong ID type --> TypeError
        self.assertRaises(TypeError, self.IDprop1.remove_property, 1)
        self.assertRaises(TypeError, self.IDprop1.remove_property, [1, 2])
        # test not exist prop name --> IDNotExistError
        self.assertRaises(IDNotExistError, self.IDprop1.remove_property, "X")
        self.assertRaises(IDNotExistError, self.IDprop1.remove_property, "Z")
        # test if property deleted properly on filled IDlist
        self.IDprop1.add_id("a")
        self.IDprop1.add_id("b")
        self.IDprop1.add_id("c")
        self.assertEqual(
            self.IDprop1.prop,
            {"x": [1, 1, 1], "z": [[1], [1], [1]]},
        )
        self.IDprop1.remove_property("z")
        self.assertEqual(self.IDprop1.prop, {"x": [1, 1, 1]})
        self.IDprop1.remove_property("x")
        self.assertEqual(self.IDprop1.prop, {})

    def test_get_all_properties(self):
        self.IDprop1.add_id("a")
        self.IDprop1.new_property("x", 10, int)
        self.IDprop1.new_property("y", 20.0, float)
        # test wrong ID type --> TypeError
        self.assertRaises(TypeError, self.IDprop1.get_all_properties, 1)
        # test wrong ID val --> IDNotExistError
        self.assertRaises(IDNotExistError, self.IDprop1.get_all_properties, "b")
        # test if all prop val retrieved correctly
        self.assertEqual(
            self.IDprop1.get_all_properties("a"),
            {"x": 10, "y": 20.0},
        )

    def test_pop_id(self):
        self.IDprop1.new_property("x", 1, int)
        self.IDprop1.new_property("y", "1", str)
        self.IDprop1.add_id("a")
        self.IDprop1.add_id("b", {"x": 2, "y": "2"})
        self.IDprop1.add_id("c", {"x": 3, "y": "3"})
        # test wrong ID type --> TypeError
        self.assertRaises(TypeError, self.IDprop1.pop_id, 1)
        # test wrong ID val --> IDNotExistError
        self.assertRaises(IDNotExistError, self.IDprop1.pop_id, "d")
        # test if poped correctly
        self.assertEqual(self.IDprop1.pop_id("b"), 1)
        self.assertEqual(self.IDprop1.lst, ["a", "c"])
        self.assertEqual(self.IDprop1.properties, ["x", "y"])
        self.assertEqual(self.IDprop1._prop, {"x": [1, 3], "y": ["1", "3"]})
        self.IDprop1.pop_id("a")
        self.IDprop1.pop_id("c")
        self.assertEqual(self.IDprop1._prop, {"x": [], "y": []})

    def test_add_id(self):
        self.IDprop1.new_property("x", 1, int)
        self.IDprop1.new_property("y", "1", str)
        # no specification of properties, use all default values
        self.IDprop1.add_id("a")
        self.assertEqual(self.IDprop1.get_property("a", "x"), 1)
        self.assertEqual(self.IDprop1.get_property("a", "y"), "1")
        # fully specified properties
        self.IDprop1.add_id("b", {"x": 2, "y": "2"})
        self.assertEqual(self.IDprop1.get_property("b", "x"), 2)
        self.assertEqual(self.IDprop1.get_property("b", "y"), "2")
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
        self.assertEqual(self.IDprop1.get_property("c", "x"), 3)
        self.assertEqual(self.IDprop1.get_property("c", "y"), "1")
        # test empty property dictionary, should be same as no `prop=None`
        self.IDprop1.add_id("d", {})
        self.assertEqual(self.IDprop1.get_property("d", "x"), 1)
        self.assertEqual(self.IDprop1.get_property("d", "y"), "1")
        # test if deepcopy used for fill in default properties for existing entities
        self.IDprop2.add_id("a")
        self.IDprop2.add_id("b")
        self.IDprop2.new_property("x", list())
        self.IDprop2.get_property("a", "x").append(1)
        self.assertEqual(self.IDprop2.get_property("a", "x"), [1])
        self.assertEqual(self.IDprop2.get_property("b", "x"), [])
        # test if deepcopy used for filling in missing properties of new ID
        self.IDprop2.add_id("c")
        self.IDprop2.add_id("d")
        self.IDprop2.get_property("c", "x").append(2)
        self.assertEqual(self.IDprop2.get_property("a", "x"), [1])
        self.assertEqual(self.IDprop2.get_property("b", "x"), [])
        self.assertEqual(self.IDprop2.get_property("c", "x"), [2])
        self.assertEqual(self.IDprop2.get_property("d", "x"), [])


if __name__ == "__main__":
    unittest.main()
