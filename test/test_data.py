import gc
import os
import shutil
import tempfile
import unittest

import pytest
from parameterized import parameterized

import NLEval.data
from NLEval.util.timer import Timeout

LEVEL = "DEBUG"

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
        # Clean after every test function call
        cls.tmp_dir = tempfile.mkdtemp()
        # Clean at the end of the test class
        cls.tmp_dir_preserve = tempfile.mkdtemp()
        print(
            f"Created temporary directory for testing data: {cls.tmp_dir}, "
            f"{cls.tmp_dir_preserve}",
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)
        shutil.rmtree(cls.tmp_dir_preserve)
        print(f"Removed temporary directories: {cls.tmp_dir}, {cls.tmp_dir_preserve}")

    def setUp(self):
        self.graph = self.lsc = None

    def tearDown(self):
        for f in os.listdir(self.tmp_dir):
            path = os.path.join(self.tmp_dir, f)
            shutil.rmtree(path)
            print(f"Removed {path}")
        del self.graph, self.lsc
        gc.collect()

    @pytest.mark.longruns
    def test_biogrid(self):
        self.graph = NLEval.data.BioGRID(self.tmp_dir, log_level=LEVEL)
        self.assertEqual(self.graph.size, 19276)
        self.assertEqual(self.graph.num_edges, 1100282)

    @parameterized.expand(full_data_test_param)
    @pytest.mark.mediumruns
    def test_bioplex(self, name, reprocess, redownload):
        with self.subTest(name):
            self.graph = NLEval.data.BioPlex(
                self.tmp_dir_preserve,
                reprocess=reprocess,
                redownload=redownload,
                log_level=LEVEL,
            )
            self.assertEqual(self.graph.size, 8116)
            self.assertEqual(self.graph.num_edges, 71046)

    @pytest.mark.mediumruns
    def test_disgenet(self):
        with Timeout(600):
            self.lsc = NLEval.data.DisGeNet(self.tmp_dir, log_level=LEVEL)

    @pytest.mark.longruns
    def test_funcoup(self):
        self.graph = NLEval.data.FunCoup(self.tmp_dir, log_level=LEVEL)
        self.assertEqual(self.graph.size, 17662)
        self.assertEqual(self.graph.num_edges, 9732196)

    @parameterized.expand([("GOBP",), ("GOCC",), ("GOMF",)])
    @pytest.mark.longruns
    def test_go(self, name):
        with self.subTest(name):
            self.lsc = getattr(NLEval.data, name)(self.tmp_dir, log_level=LEVEL)

    @pytest.mark.longruns
    def test_hippie(self):
        self.graph = NLEval.data.HIPPIE(self.tmp_dir, log_level=LEVEL)
        self.assertEqual(self.graph.size, 17879)
        self.assertEqual(self.graph.num_edges, 769474)

    @pytest.mark.longruns
    def test_humannet(self):
        self.graph = NLEval.data.HumanNet(self.tmp_dir, log_level=LEVEL)
        self.assertEqual(self.graph.size, 17740)
        self.assertEqual(self.graph.num_edges, 848552)

    @pytest.mark.longruns
    def test_pcnet(self):
        self.graph = NLEval.data.PCNet(self.tmp_dir, log_level=LEVEL)
        self.assertEqual(self.graph.size, 18276)
        self.assertEqual(self.graph.num_edges, 5198988)

    @pytest.mark.longruns
    @pytest.mark.highmemory
    def test_string(self):
        self.graph = NLEval.data.STRING(self.tmp_dir, log_level=LEVEL)
        self.assertEqual(self.graph.size, 18484)
        self.assertEqual(self.graph.num_edges, 11021544)


if __name__ == "__main__":
    unittest.main()
