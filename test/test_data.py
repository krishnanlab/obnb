import gc
import os
import shutil
import tempfile
import unittest
from urllib.parse import urljoin

import pytest
from parameterized import parameterized

import NLEval
import NLEval.data
from NLEval._config.config import NLEDATA_URL_DICT
from NLEval.util.download import download_unzip
from NLEval.util.exceptions import DataNotFoundError
from NLEval.util.timer import Timeout

opts = {
    "log_level": "DEBUG",
    "version": NLEval.__data_version__,
}
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

        data_url = urljoin(NLEDATA_URL_DICT[NLEval.__data_version__], ".cache.zip")
        download_unzip(data_url, cls.tmp_dir_preserve)

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
        self.graph = NLEval.data.BioGRID(self.tmp_dir, **opts)
        self.assertEqual(self.graph.size, 19263)
        self.assertEqual(self.graph.num_edges, 1099756)

    @parameterized.expand(full_data_test_param)
    @pytest.mark.mediumruns
    def test_bioplex(self, name, reprocess, redownload):
        with self.subTest(name):
            self.graph = NLEval.data.BioPlex(
                self.tmp_dir_preserve,
                reprocess=reprocess,
                redownload=redownload,
                **opts,
            )
            self.assertEqual(self.graph.size, 8108)
            self.assertEqual(self.graph.num_edges, 71004)

    @pytest.mark.mediumruns
    def test_disgenet(self):
        with Timeout(600):
            self.lsc = NLEval.data.DisGeNet(self.tmp_dir, **opts)

    @pytest.mark.longruns
    def test_funcoup(self):
        self.graph = NLEval.data.FunCoup(self.tmp_dir, **opts)
        self.assertEqual(self.graph.size, 17905)
        self.assertEqual(self.graph.num_edges, 10042420)

    @parameterized.expand([("GOBP",), ("GOCC",), ("GOMF",)])
    @pytest.mark.longruns
    def test_go(self, name):
        with self.subTest(name):
            self.lsc = getattr(NLEval.data, name)(self.tmp_dir, **opts)

    @pytest.mark.longruns
    def test_hippie(self):
        self.graph = NLEval.data.HIPPIE(self.tmp_dir, **opts)
        self.assertEqual(self.graph.size, 17830)
        self.assertEqual(self.graph.num_edges, 767644)

    @pytest.mark.longruns
    def test_humannet(self):
        self.graph = NLEval.data.HumanNet(self.tmp_dir, **opts)
        self.assertEqual(self.graph.size, 17739)
        self.assertEqual(self.graph.num_edges, 848414)

    @pytest.mark.longruns
    def test_pcnet(self):
        self.graph = NLEval.data.PCNet(self.tmp_dir, **opts)
        self.assertEqual(self.graph.size, 18256)
        self.assertEqual(self.graph.num_edges, 5190378)

    @pytest.mark.longruns
    @pytest.mark.highmemory
    def test_string(self):
        self.graph = NLEval.data.STRING(self.tmp_dir, **opts)
        self.assertEqual(self.graph.size, 18480)
        self.assertEqual(self.graph.num_edges, 11019492)


@pytest.mark.mediumruns
def test_archive_data_v1(tmpdir):
    print(tmpdir)
    with pytest.raises(ValueError):
        g = NLEval.data.BioGRID(tmpdir, version="nledata-vDNE-test")

    with pytest.raises(DataNotFoundError):
        g = NLEval.data.HIPPIE(tmpdir, version="nledata-v1.0-test")

    # TODO: check changed version redownload
    g = NLEval.data.BioGRID(tmpdir, version="nledata-v1.0-test")
    assert g.size == 19276
    assert g.num_edges == 1100282


if __name__ == "__main__":
    unittest.main()
