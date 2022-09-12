import gc
import os
import shutil
import tempfile
import unittest
from itertools import product
from urllib.parse import urljoin

import pytest
from parameterized import parameterized

import nleval
import nleval.data
import nleval.graph
from nleval.config import NLEDATA_URL_DICT
from nleval.exception import DataNotFoundError
from nleval.feature.base import BaseFeature
from nleval.util.dataset_constructors import default_constructor
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
        self.assertEqual(self.graph.size, 18951)
        self.assertEqual(self.graph.num_edges, 1103298)

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
            self.assertEqual(self.graph.size, 8044)
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
        self.assertEqual(self.graph.size, 17364)
        self.assertEqual(self.graph.num_edges, 768654)

    @pytest.mark.longruns
    def test_humannet(self):
        self.graph = nleval.data.HumanNet(self.tmp_dir, **opts)
        self.assertEqual(self.graph.size, 17211)
        self.assertEqual(self.graph.num_edges, 847104)

    @pytest.mark.longruns
    def test_pcnet(self):
        self.graph = nleval.data.PCNet(self.tmp_dir, **opts)
        self.assertEqual(self.graph.size, 16968)
        self.assertEqual(self.graph.num_edges, 5047854)

    @pytest.mark.longruns
    @pytest.mark.highmemory
    def test_string(self):
        self.graph = nleval.data.STRING(self.tmp_dir, **opts)
        self.assertEqual(self.graph.size, 17942)
        self.assertEqual(self.graph.num_edges, 10951202)


@pytest.mark.mediumruns
def test_archive_data_v1(tmpdir):
    nleval.logger.info(f"{tmpdir=}")
    with pytest.raises(ValueError):
        g = nleval.data.BioGRID(
            tmpdir,
            version="nledata-vDNE-test",
            download_cache=False,
        )

    with pytest.raises(DataNotFoundError):
        g = nleval.data.HIPPIE(
            tmpdir,
            version="nledata-v1.0-test",
            download_cache=False,
        )

    # TODO: check changed version redownload
    g = nleval.data.BioGRID(tmpdir, version="nledata-v1.0-test", download_cache=False)
    assert g.size == 19276
    assert g.num_edges == 1100282


@pytest.mark.mediumruns
def test_dataset_constructor(subtests, tmpdir):
    nleval.logger.info(f"{tmpdir=}")
    datadir = tmpdir / "datasets"

    for graph_as_feature, use_dense_graph in product([True, False], [True, False]):
        with subtests.test(
            graph_as_feature=graph_as_feature,
            use_dense_graph=use_dense_graph,
        ):
            dataset = default_constructor(
                root=datadir,
                graph_name="BioGRID",
                label_name="DisGeNet",
                graph_as_feature=graph_as_feature,
                use_dense_graph=use_dense_graph,
                **opts,
            )

            if graph_as_feature:
                assert isinstance(dataset.feature, BaseFeature)
            else:
                assert dataset.feature is None

            if use_dense_graph:
                assert isinstance(dataset.graph, nleval.graph.DenseGraph)
            else:
                assert isinstance(dataset.graph, nleval.graph.SparseGraph)
