import gc
import os
import shutil
import tempfile
import unittest
from urllib.parse import urljoin

import pytest
from parameterized import parameterized

import nleval
import nleval.data
from nleval.config import NLEDATA_URL_DICT
from nleval.exception import DataNotFoundError
from nleval.util.download import download_unzip
from nleval.util.timer import Timeout

opts = {
    "log_level": "DEBUG",
    "version": nleval.__data_version__,
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

        data_url = urljoin(NLEDATA_URL_DICT[nleval.__data_version__], ".cache.zip")
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
        self.graph = nleval.data.BioGRID(self.tmp_dir, **opts)
        self.assertEqual(self.graph.size, 18948)
        self.assertEqual(self.graph.num_edges, 1103272)

    @parameterized.expand(full_data_test_param)
    @pytest.mark.mediumruns
    def test_bioplex(self, name, reprocess, redownload):
        with self.subTest(name):
            self.graph = nleval.data.BioPlex(
                self.tmp_dir_preserve,
                reprocess=reprocess,
                redownload=redownload,
                **opts,
            )
            self.assertEqual(self.graph.size, 8043)
            self.assertEqual(self.graph.num_edges, 71084)

    @pytest.mark.mediumruns
    def test_disgenet(self):
        with Timeout(600):
            self.lsc = nleval.data.DisGeNet(self.tmp_dir, **opts)

    @pytest.mark.longruns
    def test_funcoup(self):
        self.graph = nleval.data.FunCoup(self.tmp_dir, **opts)
        self.assertEqual(self.graph.size, 17905)
        self.assertEqual(self.graph.num_edges, 10042420)

    @parameterized.expand([("GOBP",), ("GOCC",), ("GOMF",)])
    @pytest.mark.longruns
    def test_go(self, name):
        with self.subTest(name):
            self.lsc = getattr(nleval.data, name)(self.tmp_dir, **opts)

    @pytest.mark.longruns
    def test_hippie(self):
        self.graph = nleval.data.HIPPIE(self.tmp_dir, **opts)
        self.assertEqual(self.graph.size, 17360)
        self.assertEqual(self.graph.num_edges, 768654)

    @pytest.mark.longruns
    def test_humannet(self):
        self.graph = nleval.data.HumanNet(self.tmp_dir, **opts)
        self.assertEqual(self.graph.size, 17206)
        self.assertEqual(self.graph.num_edges, 847098)

    @pytest.mark.longruns
    def test_pcnet(self):
        self.graph = nleval.data.PCNet(self.tmp_dir, **opts)
        self.assertEqual(self.graph.size, 16971)
        self.assertEqual(self.graph.num_edges, 5049344)

    @pytest.mark.longruns
    @pytest.mark.highmemory
    def test_string(self):
        self.graph = nleval.data.STRING(self.tmp_dir, **opts)
        self.assertEqual(self.graph.size, 17938)
        self.assertEqual(self.graph.num_edges, 10950160)


@pytest.mark.mediumruns
def test_archive_data_v1(tmpdir):
    print(tmpdir)
    with pytest.raises(ValueError):
        g = nleval.data.BioGRID(tmpdir, version="nledata-vDNE-test")

    with pytest.raises(DataNotFoundError):
        g = nleval.data.HIPPIE(tmpdir, version="nledata-v1.0-test")

    # TODO: check changed version redownload
    g = nleval.data.BioGRID(tmpdir, version="nledata-v1.0-test")
    assert g.size == 19276
    assert g.num_edges == 1100282


if __name__ == "__main__":
    unittest.main()
