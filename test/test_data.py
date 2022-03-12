import gc
import os
import shutil
import tempfile
import unittest

import NLEval.data
import pytest
from NLEval.util.timer import Timeout
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
        print(
            f"Removed temporary directories: {cls.tmp_dir}, "
            f"{cls.tmp_dir_preserve}",
        )

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
        self.graph = NLEval.data.BioGRID(self.tmp_dir, verbose=True)
        self.assertEqual(self.graph.size, 25711)
        self.assertEqual(self.graph.num_edges, 1200394)

    @parameterized.expand(full_data_test_param)
    def test_bioplex(self, name, reprocess, redownload):
        with self.subTest(name):
            self.graph = NLEval.data.BioPlex(
                self.tmp_dir_preserve,
                reprocess=reprocess,
                redownload=redownload,
                verbose=True,
            )
            self.assertEqual(self.graph.size, 8364)
            self.assertEqual(self.graph.num_edges, 71408)

    @pytest.mark.xfail(
        raises=TimeoutError,
        reason="Sometimes DisGeNet is just not working...",
    )
    @pytest.mark.longruns
    def test_disgenet(self):
        with Timeout(600):
            self.lsc = NLEval.data.DisGeNet(self.tmp_dir, verbose=True)

    @pytest.mark.longruns
    @pytest.mark.highmemory
    def test_funcoup(self):
        self.graph = NLEval.data.FunCoup(self.tmp_dir, verbose=True)
        self.assertEqual(self.graph.size, 17783)
        self.assertEqual(self.graph.num_edges, 10027588)

    @parameterized.expand([("GOBP",), ("GOCC",), ("GOMF",)])
    @pytest.mark.longruns
    def test_go(self, name):
        with self.subTest(name):
            self.lsc = getattr(NLEval.data, name)(self.tmp_dir, verbose=True)

    @pytest.mark.longruns
    def test_hippie(self):
        self.graph = NLEval.data.HIPPIE(self.tmp_dir, verbose=True)
        self.assertEqual(self.graph.size, 17955)
        self.assertEqual(self.graph.num_edges, 770754)

    @pytest.mark.longruns
    def test_humannet(self):
        self.graph = NLEval.data.HumanNet(self.tmp_dir, verbose=True)
        self.assertEqual(self.graph.size, 17787)
        self.assertEqual(self.graph.num_edges, 849002)

    @pytest.mark.longruns
    @pytest.mark.highmemory
    def test_string(self):
        self.graph = NLEval.data.STRING(self.tmp_dir, verbose=True)
        self.assertEqual(self.graph.size, 18513)
        self.assertEqual(self.graph.num_edges, 11038228)


if __name__ == "__main__":
    unittest.main()
