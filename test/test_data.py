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


def check_network_stats(data_dir, network_name, num_nodes, num_edges):
    graph = getattr(nleval.data, network_name)(data_dir, **opts)
    assert graph.size == num_nodes
    assert graph.num_edges == num_edges


@pytest.mark.longruns
@pytest.mark.parametrize(
    "network_name,num_nodes,num_edges",
    [
        ("BioGRID", 19_765, 1_554_790),
        ("BioPlex", 8_108, 71_004),
        ("ComPPIHumanInt", 17_014, 699_330),
        ("ConsensusPathDB", 17_735, 10_611_416),
        ("FunCoup", 17_891, 10_037_386),
        ("HIPPIE", 19_338, 1_542_044),
        ("HuRI", 8_099, 103_186),
        ("HumanNet", 18_452, 1_954_946),
        ("OmniPath", 16_329, 289_128),
        ("PCNet", 18_540, 5_366_706),
        ("ProteomeHD", 2_471, 125_172),
        ("SIGNOR", 5_291, 28_676),
        ("STRING", 18_480, 11_019_492),
    ],
)
def test_network_data(tmpdir, network_name, num_nodes, num_edges):
    check_network_stats(tmpdir, network_name, num_nodes, num_edges)


@pytest.mark.longruns
@pytest.mark.highmemory
@pytest.mark.parametrize(
    "network_name,num_nodes,num_edges",
    [
        ("HuMAP", 15_433, 35_052_604),
        ("HumanBaseTopGlobal", 25_689, 77_807_094),
    ],
)
def test_network_data_highmem(tmpdir, network_name, num_nodes, num_edges):
    check_network_stats(tmpdir, network_name, num_nodes, num_edges)


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
            self.assertEqual(self.graph.size, 8108)
            self.assertEqual(self.graph.num_edges, 71004)

    @pytest.mark.mediumruns
    def test_disgenet(self):
        with Timeout(600):
            self.lsc = nleval.data.DisGeNET(self.tmp_dir, **opts)

    @parameterized.expand([("GOBP",), ("GOCC",), ("GOMF",)])
    @pytest.mark.longruns
    def test_go(self, name):
        with self.subTest(name):
            self.lsc = getattr(nleval.data, name)(self.tmp_dir, **opts)


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
                graph_name="BioPlex",
                label_name="DisGeNET",
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
