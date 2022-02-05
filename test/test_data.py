import shutil
import tempfile
import unittest

import NLEval.data


class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory for testing data: {cls.tmp_dir}")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)
        print(f"Removed temporary directory: {cls.tmp_dir}")

    @unittest.skip("Large file download")
    def test_biogrid(self):
        graph = NLEval.data.BioGRID(self.tmp_dir)
        self.assertEqual(graph.size, 25711)
        self.assertEqual(graph.num_edges, 1203620)

    def test_bioplex(self):
        graph = NLEval.data.BioPlex(self.tmp_dir)
        self.assertEqual(graph.size, 8364)
        self.assertEqual(graph.num_edges, 71408)

    @unittest.skip("Large file download")
    def test_funcoup(self):
        graph = NLEval.data.FunCoup(self.tmp_dir)
        self.assertEqual(graph.size, 17783)
        self.assertEqual(graph.num_edges, 10027589)

    @unittest.skip("Large file download")
    def test_hippie(self):
        graph = NLEval.data.HIPPIE(self.tmp_dir)
        self.assertEqual(graph.size, 17955)
        self.assertEqual(graph.num_edges, 775084)

    @unittest.skip("Large file download")
    def test_humannet(self):
        graph = NLEval.data.HumanNet(self.tmp_dir)
        self.assertEqual(graph.size, 17787)
        self.assertEqual(graph.num_edges, 849002)

    @unittest.skip("Large file download")
    def test_string(self):
        graph = NLEval.data.STRING(self.tmp_dir)
        self.assertEqual(graph.size, 18513)
        self.assertEqual(graph.num_edges, 11038240)


if __name__ == "__main__":
    unittest.main()
