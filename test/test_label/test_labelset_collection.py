import os
import unittest

from commonvar import SAMPLE_DATA_DIR

from obnb.exception import IDExistsError, IDNotExistError
from obnb.graph import OntologyGraph
from obnb.label import LabelsetCollection


class TestLabelsetCollection(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.toy1_gmt_path = os.path.join(SAMPLE_DATA_DIR, "toy1.gmt")
        self.toy1_prop_path = os.path.join(SAMPLE_DATA_DIR, "toy1_property.tsv")
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
        """Template for testing inputs for methods with only one positional argument as
        ID, i.e. `.get_info`, `get_labelset`, and `get_noccur`."""
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
        # test with known positive ID --> IDExistsError
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
        lsc = LabelsetCollection.from_gmt(self.toy1_gmt_path)
        self.assertEqual(lsc.label_ids, self.toy1_label_ids)
        self.assertEqual(lsc.prop["Info"], self.toy1_InfoLst)
        self.assertEqual(lsc.prop["Labelset"], self.toy1_labelsets)

    def test_from_dict(self):
        input_dict = {"a": "L1", "b": "L2", "c": "L1", "f": "L2", "h": "L1"}
        lsc = LabelsetCollection.from_dict(input_dict)
        self.assertEqual(lsc.get_labelset("L1"), {"a", "c", "h"})
        self.assertEqual(lsc.get_labelset("L2"), {"b", "f"})

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
            # make sure nothing popped
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
        self.lsc.load_entity_properties(
            self.toy1_prop_path,
            "toy1_prop",
            0,
            int,
        )
        self.assertEqual(self.lsc.entity.prop["toy1_prop"], self.toy1_property)
        # test loading property with existing property name --> IDExistsError
        self.assertRaises(
            IDExistsError,
            self.lsc.load_entity_properties,
            self.toy1_prop_path,
            "toy1_prop",
            0,
            int,
        )
        # make sure entity with properties different from default don't get popped
        self.lsc.pop_labelset("Labelset1")
        self.lsc.pop_labelset("Labelset2")
        self.assertEqual(self.lsc.entity.map, {"a": 0, "c": 1, "d": 2})

    def test_get_y(self):
        input_dict = {"a": "L1", "b": "L2", "c": "L1", "f": "L2", "h": "L1"}
        lsc = LabelsetCollection.from_dict(input_dict)
        lsc.set_negative(["b"], "L1")

        y = lsc.get_y(("a", "b", "c", "f", "h"))
        self.assertEqual(y.T.tolist(), [[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]])

        y = lsc.get_y(("a", "b", "c", "f", "h"), "L1")
        self.assertEqual(y.T.tolist(), [[1, 0, 1, 0, 1]])

        y, m = lsc.get_y(("a", "b", "c", "f", "h"), "L1", return_y_mask=True)
        self.assertEqual(y.T.tolist(), [[1, 0, 1, 0, 1]])
        self.assertEqual(m.T.tolist(), [[1, 1, 1, 0, 1]])

        y, m = lsc.get_y(("a", "b", "c", "f", "h"), "L2", return_y_mask=True)
        self.assertEqual(y.T.tolist(), [[0, 1, 0, 1, 0]])
        self.assertEqual(m.T.tolist(), [[1, 1, 1, 1, 1]])

        y = lsc.get_y(("a", "c", "b", "x", "f", "h"))
        self.assertEqual(y.T.tolist(), [[1, 1, 0, 0, 0, 1], [0, 0, 1, 0, 1, 0]])

    def test_read_ontology_graph(self):
        graph = OntologyGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.set_node_name("a", "A")
        graph.set_node_name("b", "B")
        graph.set_node_attr("a", ["x", "y"])
        graph.set_node_attr("b", ["a", "y", "z"])

        lsc = LabelsetCollection()
        lsc.read_ontology_graph(graph, min_size=1)
        self.assertEqual(lsc.get_labelset("a"), {"x", "y"})
        self.assertEqual(lsc.get_labelset("b"), {"a", "y", "z"})
        self.assertEqual(lsc.get_info("a"), "A")
        self.assertEqual(lsc.get_info("b"), "B")

        lsc = LabelsetCollection()
        lsc.read_ontology_graph(graph, min_size=3)
        self.assertEqual(lsc.get_labelset("b"), {"a", "y", "z"})
        self.assertEqual(lsc.get_info("b"), "B")
        self.assertFalse("a" in lsc.label_ids)

        with self.subTest("Namespace"):
            graph.add_edge("c", "b")
            graph.set_node_attr("c", ["a", "b"])

            lsc = LabelsetCollection()
            lsc.read_ontology_graph(graph, min_size=0, namespace="a")
            self.assertEqual(sorted(lsc.label_ids), [])

            lsc = LabelsetCollection()
            lsc.read_ontology_graph(graph, min_size=0, namespace="b")
            self.assertEqual(sorted(lsc.label_ids), ["c"])

            lsc = LabelsetCollection()
            lsc.read_ontology_graph(graph, min_size=0, namespace="c")
            self.assertEqual(sorted(lsc.label_ids), [])


if __name__ == "__main__":
    unittest.main()
