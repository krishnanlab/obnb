import io
import shutil
import tempfile
import unittest

import NLEval.data
import pytest
from parameterized import parameterized

# Name, reprocess, redownload
full_data_test_param = [
    ("Download", False, False),
    ("Reuse", False, False),
    ("Reprocess", True, False),
    ("Redownload", True, True),
]


class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory for testing data: {cls.tmp_dir}")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)
        print(f"Removed temporary directory: {cls.tmp_dir}")

    @pytest.mark.longruns
    def test_biogrid(self):
        graph = NLEval.data.BioGRID(self.tmp_dir)
        self.assertEqual(graph.size, 25711)
        self.assertEqual(graph.num_edges, 1200394)

    @parameterized.expand(full_data_test_param)
    def test_bioplex(self, name, reprocess, redownload):
        with self.subTest(name):
            graph = NLEval.data.BioPlex(
                self.tmp_dir,
                reprocess=reprocess,
                redownload=redownload,
            )
            self.assertEqual(graph.size, 8364)
            self.assertEqual(graph.num_edges, 71408)

    @unittest.skip("Sometimes DisGeNet is just not working...")
    def test_disgenet(self):
        lsc = NLEval.data.DisGeNet(self.tmp_dir)

    @pytest.mark.longruns
    def test_funcoup(self):
        graph = NLEval.data.FunCoup(self.tmp_dir)
        self.assertEqual(graph.size, 17783)
        self.assertEqual(graph.num_edges, 10027588)

    @parameterized.expand([("GOBP",), ("GOCC",), ("GOMF",)])
    @pytest.mark.longruns
    def test_go(self, name):
        with self.subTest(name):
            lsc = getattr(NLEval.data, name)(self.tmp_dir)

    @pytest.mark.longruns
    def test_hippie(self):
        graph = NLEval.data.HIPPIE(self.tmp_dir)
        self.assertEqual(graph.size, 17955)
        self.assertEqual(graph.num_edges, 770754)

    @pytest.mark.longruns
    def test_humannet(self):
        graph = NLEval.data.HumanNet(self.tmp_dir)
        self.assertEqual(graph.size, 17787)
        self.assertEqual(graph.num_edges, 849002)

    @pytest.mark.longruns
    def test_string(self):
        graph = NLEval.data.STRING(self.tmp_dir)
        self.assertEqual(graph.size, 18513)
        self.assertEqual(graph.num_edges, 11038228)


if __name__ == "__main__":
    unittest.main()
