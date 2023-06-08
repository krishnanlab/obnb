import itertools
import os
import os.path as osp
import shutil
import tempfile
import unittest
from copy import deepcopy

import ndex2
import numpy as np
from commonvar import SAMPLE_DATA_DIR
from parameterized import parameterized

from obnb.exception import EdgeNotExistError, IDExistsError, IDNotExistError
from obnb.graph import DenseGraph, DirectedSparseGraph, OntologyGraph, SparseGraph
from obnb.graph.base import BaseGraph
from obnb.util import idhandler


def shuffle_sparse(graph):
    n = graph.size
    shuffle_idx = np.random.choice(n, size=n, replace=False)
    new_graph = SparseGraph(
        weighted=graph.weighted,
        directed=graph.directed,
    )
    for i in shuffle_idx:
        node_id = graph.idmap.lst[i]
        new_graph.add_node(node_id)
    for idx1, node_id1 in enumerate(graph.idmap):
        for idx2, weight in graph.edge_data[graph.idmap[node_id1]].items():
            node_id2 = graph.idmap.lst[idx2]
            new_graph.add_edge(node_id1, node_id2, weight)
    return new_graph


def shuffle_dense(graph):
    n = graph.size
    shuffle_idx = np.random.choice(n, size=n, replace=False)
    new_graph = DenseGraph()

    for i in shuffle_idx:
        node_id = graph.idmap.lst[i]
        new_graph.idmap.add_id(node_id)
    new_graph.mat = np.zeros(graph.mat.shape)
    for idx1_new, idx1_old in enumerate(shuffle_idx):
        for idx2_new, idx2_old in enumerate(shuffle_idx):
            new_graph.mat[idx1_new, idx2_new] = graph.mat[idx1_old, idx2_old]
    return new_graph


class test_case1:
    def __init__(self):
        self.tw_path = os.path.join(SAMPLE_DATA_DIR, "toy1_weighted.edg")
        self.tu_path = os.path.join(SAMPLE_DATA_DIR, "toy1_unweighted.edg")
        self.temd_path = os.path.join(SAMPLE_DATA_DIR, "toy1.emd")
        self.IDlst = ["1", "3", "4", "2", "5"]
        self.data_unweighted = [
            {1: 1, 2: 1},
            {0: 1, 4: 1},
            {3: 1, 0: 1},
            {2: 1},
            {1: 1},
        ]
        self.data_weighted = [
            {1: 0.4},
            {0: 0.4, 4: 0.1},
            {3: 0.3},
            {2: 0.3},
            {1: 0.1},
        ]
        self.data_mat = np.array(
            [
                [1, 0, 0, 0.4, 0, 0],
                [4, 0, 0, 0, 0.3, 0],
                [3, 0.4, 0, 0, 0, 0.1],
                [2, 0, 0.3, 0, 0, 0],
                [5, 0, 0, 0.1, 0, 0],
            ],
        )
        self.data_mat.setflags(write=False)


class TestBaseGraph(unittest.TestCase):
    def setUp(self):
        self.graph = BaseGraph()

    def test_idmap_setter(self):
        with self.assertRaises(TypeError):
            self.graph.idmap = "asdg"

    def test_size(self):
        self.assertEqual(self.graph.size, 0)
        self.assertEqual(self.graph.num_nodes, 0)
        for i in range(5):
            with self.subTest(i=i):
                self.graph.idmap.add_id(str(i))
                self.assertEqual(self.graph.size, i + 1)
                self.assertEqual(self.graph.num_nodes, i + 1)

    def test_isempty(self):
        self.assertTrue(self.graph.isempty())
        self.graph.idmap.add_id("a")
        self.assertFalse(self.graph.isempty())

    def test_get_node_id(self):
        self.graph.idmap.add_id("a")
        self.graph.idmap.add_id("b")
        self.assertEqual(self.graph.get_node_id("a"), "a")
        self.assertEqual(self.graph.get_node_id(0), "a")
        self.assertEqual(self.graph.get_node_id("b"), "b")
        self.assertEqual(self.graph.get_node_id(1), "b")

    def test_get_node_idx(self):
        self.graph.idmap.add_id("a")
        self.graph.idmap.add_id("b")
        self.assertEqual(self.graph.get_node_idx("a"), 0)
        self.assertEqual(self.graph.get_node_idx(0), 0)
        self.assertEqual(self.graph.get_node_idx("b"), 1)
        self.assertEqual(self.graph.get_node_idx(1), 1)


class TestSparseGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = test_case1()
        cls.lst = [["1", "4"], ["2", "5"], ["5", "3", "2"]]

        cls.tmp_dir = tempfile.mkdtemp()
        cls.npz_path1 = osp.join(cls.tmp_dir, "network1.npz")
        cls.edge_index1 = np.array(
            [
                [0, 1, 1, 2, 2, 2, 3, 4],
                [1, 0, 2, 1, 3, 4, 2, 2],
            ],
        )
        cls.edge_weight1 = np.array([1, 1, 2, 2, 3, 0.5, 3, 0.5])
        cls.node_ids1 = np.array(["a", "b", "c", "d", "e"])
        cls.edge_data_weighted1 = [
            {1: 1},
            {0: 1, 2: 2},
            {1: 2, 3: 3, 4: 0.5},
            {2: 3},
            {2: 0.5},
        ]
        cls.edge_data_unweighted1 = [
            {1: 1},
            {0: 1, 2: 1},
            {1: 1, 3: 1, 4: 1},
            {2: 1},
            {2: 1},
        ]
        np.savez(
            cls.npz_path1,
            edge_index=cls.edge_index1,
            edge_weight=cls.edge_weight1,
            node_ids=cls.node_ids1,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_add_node(self):
        with self.subTest("Add single node"):
            graph = SparseGraph(weighted=False, directed=False)

            graph.add_node("a")
            self.assertEqual(sorted(graph.node_ids), ["a"])
            self.assertEqual(graph._edge_data, [{}])

            graph.add_node("b")
            self.assertEqual(sorted(graph.node_ids), ["a", "b"])
            self.assertEqual(graph._edge_data, [{}, {}])

            graph.add_node("a", exist_ok=True)
            self.assertEqual(sorted(graph.node_ids), ["a", "b"])
            self.assertEqual(graph._edge_data, [{}, {}])

            self.assertRaises(IDExistsError, graph.add_node, "a")
            self.assertRaises(IDExistsError, graph.add_node, "b")

        with self.subTest("Add multiple nodes"):
            graph = SparseGraph(weighted=False, directed=False)

            graph.add_nodes(["a", "b"])
            self.assertEqual(sorted(graph.node_ids), ["a", "b"])
            self.assertEqual(graph._edge_data, [{}, {}])

            self.assertRaises(IDExistsError, graph.add_node, "a")
            self.assertRaises(IDExistsError, graph.add_nodes, ["c", "b"])

    def test_add_edge(self):
        graph = SparseGraph()

        with self.subTest("Edge and node creations"):
            graph.add_edge("a", "b")
            self.assertEqual(sorted(graph.node_ids), ["a", "b"])
            self.assertEqual(graph._edge_data, [{1: 1.0}, {0: 1.0}])

        with self.subTest("Overwriting edge value (no edge reduction)"):
            graph.add_edge("a", "b", 0.8)
            self.assertEqual(graph._edge_data, [{1: 0.8}, {0: 0.8}])

        with self.subTest("Edge reduction with min"):
            graph.add_edge("a", "b", 1.0, reduction="min")
            self.assertEqual(graph._edge_data, [{1: 0.8}, {0: 0.8}])

            graph.add_edge("a", "b", 0.5, reduction="min")
            self.assertEqual(graph._edge_data, [{1: 0.5}, {0: 0.5}])

        with self.subTest("Edge reduction with max"):
            graph.add_edge("a", "b", 0.1, reduction="max")
            self.assertEqual(graph._edge_data, [{1: 0.5}, {0: 0.5}])

            graph.add_edge("a", "b", 1.0, reduction="max")
            self.assertEqual(graph._edge_data, [{1: 1.0}, {0: 1.0}])

        with self.subTest("More edges"):
            graph.add_edge("a", "c", 0.5)
            self.assertEqual(sorted(graph.node_ids), ["a", "b", "c"])
            self.assertEqual(
                graph._edge_data,
                [{1: 1.0, 2: 0.5}, {0: 1.0}, {0: 0.5}],
            )

    def test_remove_edge(self):
        with self.subTest("Test sparse graph (directed=False)"):
            graph = SparseGraph(directed=False)
            graph.add_edge("a", "b")
            graph.add_edge("b", "c")
            self.assertEqual(graph._edge_data, [{1: 1.0}, {0: 1.0, 2: 1.0}, {1: 1.0}])

            graph.remove_edge("a", "b")
            self.assertEqual(graph._edge_data, [{}, {2: 1.0}, {1: 1.0}])

            with self.assertRaises(EdgeNotExistError):
                graph.remove_edge("a", "b")

            graph.remove_edge("c", "b")
            self.assertEqual(graph._edge_data, [{}, {}, {}])

        with self.subTest("Test sparse graph (directed=True)"):
            graph = SparseGraph(directed=True)

            graph.add_edge("a", "b")
            graph.add_edge("b", "a")
            graph.add_edge("b", "c")
            self.assertEqual(graph._edge_data, [{1: 1.0}, {0: 1.0, 2: 1.0}, {}])

            graph.remove_edge("a", "b")
            self.assertEqual(graph._edge_data, [{}, {0: 1.0, 2: 1.0}, {}])

            with self.assertRaises(EdgeNotExistError):
                graph.remove_edge("a", "b")

            graph.remove_edge("b", "c")
            self.assertEqual(graph._edge_data, [{}, {0: 1.0}, {}])

    @parameterized.expand([(True, 3), (False, 4)])
    def test_num_nodes(self, directed, answer):
        graph = SparseGraph(weighted=False, directed=directed)
        graph.add_edge("a", "b")
        graph.add_edge("b", "a")
        graph.add_edge("c", "a")
        self.assertEqual(answer, graph.num_edges)

    @parameterized.expand(itertools.product((True, False), (True, False)))
    def test_add_edge_self_loops(self, directed, self_loops):
        with self.subTest(directed=directed, self_loops=self_loops):
            graph = SparseGraph(directed=directed, self_loops=self_loops)

            graph.add_edge("a", "a")
            edge_data = [{0: 1}] if self_loops else [{}]
            self.assertEqual(sorted(graph.node_ids), ["a"])
            self.assertEqual(graph._edge_data, edge_data)

            graph.add_edge("a", "b")
            edge_data[0][1] = 1
            edge_data.append({} if directed else {0: 1})
            self.assertEqual(sorted(graph.node_ids), ["a", "b"])
            self.assertEqual(graph._edge_data, edge_data)

    def test_read_edglst_unweighted(self):
        graph = SparseGraph.from_edglst(
            self.case.tu_path,
            weighted=False,
            directed=False,
        )
        self.assertEqual(graph.idmap.lst, self.case.IDlst)
        self.assertEqual(graph.edge_data, self.case.data_unweighted)

    def test_read_edglst_weighted(self):
        graph = SparseGraph.from_edglst(
            self.case.tw_path,
            weighted=True,
            directed=False,
        )
        self.assertEqual(graph.idmap.lst, self.case.IDlst)
        self.assertEqual(graph.edge_data, self.case.data_weighted)

    def test_read_npymat_weighted(self):
        graph = SparseGraph.from_npy(
            self.case.data_mat,
            weighted=True,
            directed=False,
        )
        self.assertEqual(graph.idmap.lst, self.case.IDlst)
        self.assertEqual(graph.edge_data, self.case.data_weighted)

    @parameterized.expand([(True,), (False,)])
    def test_read_npz(self, weighted):
        if weighted:
            edge_data = self.edge_data_weighted1
        else:
            edge_data = self.edge_data_unweighted1

        with self.subTest(weighted=weighted):
            graph = SparseGraph(weighted=weighted, directed=False)
            graph.read_npz(self.npz_path1)
            self.assertEqual(graph._edge_data, edge_data)
            self.assertEqual(graph.idmap.lst, self.node_ids1.tolist())

            graph = SparseGraph.from_npz(
                self.npz_path1,
                weighted=weighted,
                directed=False,
            )
            self.assertEqual(graph._edge_data, edge_data)
            self.assertEqual(graph.idmap.lst, self.node_ids1.tolist())

    @parameterized.expand([(True,), (False,)])
    def test_write_npz(self, weighted):
        with self.subTest(weighted=weighted):
            graph = SparseGraph(weighted=weighted, directed=False)
            graph._edge_data = self.edge_data_weighted1
            graph.idmap = graph.idmap.from_list(self.node_ids1.tolist())

            out_path = osp.join(self.tmp_dir, "test_save.npz")
            graph.save_npz(out_path, weighted=weighted)

            npz_files = np.load(out_path)
            self.assertEqual(
                npz_files["edge_index"].tolist(),
                self.edge_index1.tolist(),
            )
            self.assertEqual(
                npz_files["node_ids"].tolist(),
                self.node_ids1.tolist(),
            )
            if weighted:
                self.assertEqual(
                    npz_files["edge_weight"].tolist(),
                    self.edge_weight1.tolist(),
                )
            else:
                self.assertFalse("edge_weight" in npz_files)

    def template_test_construct_adj_vec(self, weighted, directed, lst=None):
        graph = SparseGraph.from_npy(
            self.case.data_mat,
            weighted=weighted,
            directed=directed,
        )
        adjmat = graph.to_adjmat()
        if not lst:
            lst = graph.idmap.lst
        for ID_lst in graph.idmap.lst:
            idx_lst = graph.idmap[ID_lst]
            with self.subTest(ID_lst=ID_lst, idx_lst=idx_lst):
                self.assertEqual(list(graph[ID_lst]), list(adjmat[idx_lst]))

    def test_construct_adj_vec_weighted(self):
        self.template_test_construct_adj_vec(weighted=True, directed=False)

    def test_construct_adj_vec_unweighted(self):
        self.template_test_construct_adj_vec(weighted=False, directed=False)

    def test_construct_adj_vec_weighted_multiple(self):
        self.template_test_construct_adj_vec(
            weighted=True,
            directed=False,
            lst=self.lst,
        )

    def test_construct_adj_vec_unweighted_multiple(self):
        self.template_test_construct_adj_vec(
            weighted=False,
            directed=False,
            lst=self.lst,
        )

    def test_eq(self):
        graph = SparseGraph.from_npy(
            self.case.data_mat,
            weighted=True,
            directed=False,
        )
        for i in range(5):
            # repeat shuffle test
            with self.subTest(i=i):
                self.assertTrue(graph == shuffle_sparse(graph))
        graph2 = deepcopy(graph)
        graph2.add_node("x")
        self.assertFalse(graph == graph2)

    def test_to_dense_graph(self):
        mat = np.array(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
            ],
        )
        ids = ["a", "b", "c", "d"]
        graph = SparseGraph.from_mat(mat, ids)
        graph = graph.to_dense_graph()
        self.assertIsInstance(graph, DenseGraph)
        self.assertEqual(list(graph.node_ids), ids)
        self.assertEqual(graph.mat.tolist(), mat.tolist())

    @parameterized.expand(
        [
            (["b", "c", "d"], [{1: 2}, {0: 2, 2: 3}, {1: 3}]),
            (["e", "a", "c"], [{2: 0.5}, {}, {0: 0.5}]),
        ],
    )
    def test_induced_subgraph(self, subgraph_node_ids, subgraph_edge_data):
        graph = SparseGraph(weighted=True, directed=False)
        graph.read_npz(self.npz_path1)

        with self.subTest(subgraph_node_ids=subgraph_node_ids):
            subgraph = graph.induced_subgraph(subgraph_node_ids)
            self.assertEqual(list(subgraph.node_ids), subgraph_node_ids)
            self.assertEqual(subgraph.edge_data, subgraph_edge_data)

    def test_connected_components(self):
        graph = SparseGraph(weighted=True, directed=False)
        graph.add_nodes(["a", "b", "c", "d", "e"])
        graph.add_edge("a", "b", 1)
        graph.add_edge("c", "d", 2)
        graph.add_edge("c", "e", 1)
        graph.add_edge("d", "e", 1)

        # Two connected components
        self.assertEqual(
            graph.connected_components(),
            [["c", "d", "e"], ["a", "b"]],
        )
        self.assertFalse(graph.is_connected())

        # Largest connected subgraph
        subgraph = graph.largest_connected_subgraph()
        self.assertTrue(subgraph.is_connected())
        self.assertEqual(subgraph.node_ids, ("c", "d", "e"))
        self.assertEqual(
            subgraph.edge_data,
            [{1: 2, 2: 1}, {0: 2, 2: 1}, {0: 1, 1: 1}],
        )


class TestDirectedSparseGraph(unittest.TestCase):
    def test_add_edge(self):
        graph = DirectedSparseGraph()

        with self.subTest("Edge and node creations"):
            graph.add_edge("a", "b")
            self.assertEqual(sorted(graph.node_ids), ["a", "b"])
            self.assertEqual(graph._edge_data, [{1: 1.0}, {}])
            self.assertEqual(graph._rev_edge_data, [{}, {0: 1.0}])

        with self.subTest("Overwriting edge value (no edge reduction)"):
            graph.add_edge("a", "b", 0.8)
            self.assertEqual(graph._edge_data, [{1: 0.8}, {}])
            self.assertEqual(graph._rev_edge_data, [{}, {0: 0.8}])

        with self.subTest("Edge reduction with min"):
            graph.add_edge("a", "b", 1.0, reduction="min")
            self.assertEqual(graph._edge_data, [{1: 0.8}, {}])
            self.assertEqual(graph._rev_edge_data, [{}, {0: 0.8}])

            graph.add_edge("a", "b", 0.5, reduction="min")
            self.assertEqual(graph._edge_data, [{1: 0.5}, {}])
            self.assertEqual(graph._rev_edge_data, [{}, {0: 0.5}])

        with self.subTest("Edge reduction with max"):
            graph.add_edge("a", "b", 0.1, reduction="max")
            self.assertEqual(graph._edge_data, [{1: 0.5}, {}])
            self.assertEqual(graph._rev_edge_data, [{}, {0: 0.5}])

            graph.add_edge("a", "b", 1.0, reduction="max")
            self.assertEqual(graph._edge_data, [{1: 1.0}, {}])
            self.assertEqual(graph._rev_edge_data, [{}, {0: 1.0}])

        with self.subTest("More edges"):
            graph.add_edge("b", "a", 1.0)
            self.assertEqual(graph._edge_data, [{1: 1.0}, {0: 1.0}])
            self.assertEqual(graph._rev_edge_data, [{1: 1.0}, {0: 1.0}])

            graph.add_edge("a", "c", 0.5)
            self.assertEqual(sorted(graph.node_ids), ["a", "b", "c"])
            self.assertEqual(
                graph._edge_data,
                [{1: 1.0, 2: 0.5}, {0: 1.0}, {}],
            )

    def test_remove_edge(self):
        graph = DirectedSparseGraph()

        graph.add_edge("a", "b")
        graph.add_edge("b", "a")
        graph.add_edge("b", "c")
        self.assertEqual(graph._edge_data, [{1: 1.0}, {0: 1.0, 2: 1.0}, {}])
        self.assertEqual(graph._rev_edge_data, [{1: 1.0}, {0: 1.0}, {1: 1.0}])

        graph.remove_edge("a", "b")
        self.assertEqual(graph._edge_data, [{}, {0: 1.0, 2: 1.0}, {}])
        self.assertEqual(graph._rev_edge_data, [{1: 1.0}, {}, {1: 1.0}])

        with self.assertRaises(EdgeNotExistError):
            graph.remove_edge("a", "b")

        graph.remove_edge("b", "c")
        self.assertEqual(graph._edge_data, [{}, {0: 1.0}, {}])
        self.assertEqual(graph._rev_edge_data, [{1: 1.0}, {}, {}])


class TestCX(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()

        # Test using BioGRID PPI (Z. mays)
        # https://www.ndexbio.org/viewer/networks/291b1f2c-7f05-11ec-b3be-0ac135e8bacf
        cls.biogridzm_uuid = "291b1f2c-7f05-11ec-b3be-0ac135e8bacf"
        cls.biogridzm_data_path = osp.join(cls.tmp_dir, "biogridzm_data.cx")
        cls.biogridzm_expected_edges = [
            ("542425", "541915"),
            ("541867", "541867"),
            ("541867", "542687"),
            ("103630348", "828791"),
            ("542391", "819549"),
            ("542391", "820356"),
            ("100384477", "542384"),
            ("841321", "542373"),
            ("100125650", "103637824"),
            ("541812", "542682"),
            ("100191882", "100191882"),
            ("100191882", "100125653"),
            ("3480", "542291"),
        ]

        # Test using the alternative NF-kaapaB pathway
        # https://www.ndexbio.org/viewer/networks/28e3e28a-7f05-11ec-b3be-0ac135e8bacf
        cls.anfkb_uuid = "28e3e28a-7f05-11ec-b3be-0ac135e8bacf"
        cls.anfkb_data_path = osp.join(cls.tmp_dir, "anfkb_data.cx")
        cls.anfkb_expected_edges = [
            ("BTRC", "NFKB2", "controls-state-change-of"),
            ("BTRC", "RELB", "controls-state-change-of"),
            ("CHUK", "NFKB2", "controls-phosphorylation-of"),
            ("CHUK", "RELB", "controls-state-change-of"),
            ("MAP3K14", "CHUK", "controls-phosphorylation-of"),
            ("MAP3K14", "NFKB2", "controls-phosphorylation-of"),
            ("MAP3K14", "RELB", "controls-state-change-of"),
            ("NFKB1", "RELB", "in-complex-with"),
            ("NFKB2", "RELB", "in-complex-with"),
        ]
        cls.anfkb_node_alias = {
            "BTRC": "Q5W141",
            "NFKB2": "D3DR83",
            "RELB": "Q6GTX7",
            "CHUK": "O15111",
            "MAP3K14": "D3DX67",
            "NFKB1": "B3KVE8",
        }

        # Test using the HCM-RH PE-measure 0.01 network
        # https://www.ndexbio.org/viewer/networks/292ef54e-7f05-11ec-b3be-0ac135e8bacf
        cls.hcmrh_uuid = "292ef54e-7f05-11ec-b3be-0ac135e8bacf"
        cls.hcmrh_data_path = osp.join(cls.tmp_dir, "hcmrh_data.cx")
        # source, target, interaction, k=0
        cls.hcmrh_expected_edges = [
            ("dome", "cytokine", "Complex", 0.5),
            ("PSMD4", "Death", "Activation", 0.5),
            ("INS", "glucose", "Inhibition", 0.5),
            ("Heart", "rehospitalization", "Inhibition", 0.5),
            ("DCM", "Death", "Activation", 0.5),
            ("ivabradine hydrochloride", "Heart Rate", "Inhibition", 0.5),
            ("NLRP3", "IL1B", "Activation", 0.5),
        ]

        uuids_data_paths = [
            (cls.biogridzm_uuid, cls.biogridzm_data_path),
            (cls.anfkb_uuid, cls.anfkb_data_path),
            (cls.hcmrh_uuid, cls.hcmrh_data_path),
        ]

        for uuid, data_path in uuids_data_paths:
            client = ndex2.client.Ndex2()
            client_resp = client.get_network_as_cx_stream(uuid)
            with open(data_path, "wb") as f:
                f.write(client_resp.content)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_dense_from_cx_stream_file_biogridzm(self):
        graph = DenseGraph.from_cx_stream_file(
            self.biogridzm_data_path,
            self_loops=True,
        )

        for node1, node2 in self.biogridzm_expected_edges:
            idx1 = graph.idmap[node1]
            idx2 = graph.idmap[node2]
            self.assertTrue(graph.mat[idx1, idx2] == 1)
            self.assertTrue(graph.mat[idx2, idx1] == 1)

    def test_dense_from_cx_stream_file_anfkb(self):
        for interaction_types in [
            None,
            ["controls-state-change-of"],
            ["controls-phosphorylation-of"],
            ["in-complex-with"],
            ["controls-state-change-of", "in-complex-with"],
        ]:
            for use_node_alias in False, True:
                with self.subTest(
                    interaction_types=interaction_types,
                    use_node_alias=use_node_alias,
                ):
                    node_id_prefix = "uniprot" if use_node_alias else None
                    graph = DenseGraph.from_cx_stream_file(
                        self.anfkb_data_path,
                        directed=True,
                        interaction_types=interaction_types,
                        node_id_entry="n",
                        node_id_prefix=node_id_prefix,
                        use_node_alias=use_node_alias,
                    )

                    for i, j, k in self.anfkb_expected_edges:
                        node1, node2 = i, j
                        if use_node_alias:
                            node1 = self.anfkb_node_alias[i]
                            node2 = self.anfkb_node_alias[j]

                        if interaction_types is None or k in interaction_types:
                            idx1 = graph.idmap[node1]
                            idx2 = graph.idmap[node2]
                            self.assertTrue(graph.mat[idx1, idx2] == 1)

                            # Check directedness
                            if (j, i) not in self.anfkb_expected_edges:
                                self.assertFalse(graph.mat[idx2, idx1] == 1)

    def test_dense_from_cx_stream_file_hcmrh(self):
        for interaction_types in [
            None,
            ["complex"],
            ["activation"],
            ["inhibition"],
            ["activation", "inhibition"],
        ]:
            with self.subTest(interaction_types=interaction_types):
                graph = DenseGraph.from_cx_stream_file(
                    self.hcmrh_data_path,
                    directed=True,
                    interaction_types=interaction_types,
                    node_id_entry="n",
                    node_id_prefix=None,
                    edge_weight_attr_name=" k = 0 ",
                )

                for i, j, k, l in self.hcmrh_expected_edges:
                    if interaction_types is None or k in interaction_types:
                        idx1 = graph.idmap[i]
                        idx2 = graph.idmap[j]
                        self.assertEqual(graph.mat[idx1, idx2], l)

                        # Check directedness
                        if (j, i) not in self.hcmrh_expected_edges:
                            self.assertEqual(graph.mat[idx2, idx1], 0)

    def test_sparse_from_cx_stream_file_biogridzm(self):
        graph = SparseGraph.from_cx_stream_file(
            self.biogridzm_data_path,
            self_loops=True,
        )

        for node1, node2 in self.biogridzm_expected_edges:
            idx1 = graph.idmap[node1]
            idx2 = graph.idmap[node2]
            self.assertTrue(idx2 in graph._edge_data[idx1])
            self.assertTrue(idx1 in graph._edge_data[idx2])

    def test_sparse_from_cx_stream_file_anfkb(self):
        for interaction_types in [
            None,
            ["controls-state-change-of"],
            ["controls-phosphorylation-of"],
            ["in-complex-with"],
            ["controls-state-change-of", "in-complex-with"],
        ]:
            with self.subTest(interaction_types=interaction_types):
                graph = SparseGraph.from_cx_stream_file(
                    self.anfkb_data_path,
                    directed=True,
                    interaction_types=interaction_types,
                    node_id_entry="n",
                    node_id_prefix=None,
                )

                for i, j, k in self.anfkb_expected_edges:
                    if interaction_types is None or k in interaction_types:
                        idx1 = graph.idmap[i]
                        idx2 = graph.idmap[j]
                        self.assertTrue(idx2 in graph._edge_data[idx1])

                        # Check directedness
                        if (j, i) not in self.anfkb_expected_edges:
                            self.assertFalse(idx1 in graph._edge_data[idx2])

    def test_sparse_from_cx_stream_file_hcmrh(self):
        for interaction_types in [
            None,
            ["complex"],
            ["activation"],
            ["inhibition"],
            ["activation", "inhibition"],
        ]:
            with self.subTest(interaction_types=interaction_types):
                graph = SparseGraph.from_cx_stream_file(
                    self.hcmrh_data_path,
                    directed=True,
                    interaction_types=interaction_types,
                    node_id_entry="n",
                    node_id_prefix=None,
                    edge_weight_attr_name=" k = 0 ",
                )

                for i, j, k, l in self.hcmrh_expected_edges:
                    if interaction_types is None or k in interaction_types:
                        idx1 = graph.idmap[i]
                        idx2 = graph.idmap[j]
                        self.assertEqual(graph._edge_data[idx1][idx2], l)

                        # Check directedness
                        if (j, i) not in self.hcmrh_expected_edges:
                            self.assertFalse(idx1 in graph._edge_data[idx2])


class TestDenseGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = test_case1()

    def check_graph(self, graph):
        """Compare graph with data, true if identical."""
        mat = self.case.data_mat[:, 1:]
        IDlst = [str(int(i)) for i in self.case.data_mat[:, 0]]
        for idx1, node_id1 in enumerate(IDlst):
            for idx2, node_id2 in enumerate(IDlst):
                with self.subTest(
                    idx1=idx1,
                    idx2=idx2,
                    node_id1=type(node_id1),
                    node_id2=node_id2,
                ):
                    self.assertEqual(
                        mat[idx1, idx2],
                        graph.get_edge(node_id1, node_id2),
                    )

    def test_add_node(self):
        graph = DenseGraph()

        # add_node not set up for DenseGraph yet
        with self.assertRaises(NotImplementedError):
            graph.add_node("a")

    def test_mat(self):
        graph = DenseGraph()
        graph.idmap.add_id("a")
        graph.idmap.add_id("b")
        graph.mat = np.random.random((2, 2))
        # test type check: only numpy array allowed
        with self.assertRaises(TypeError):
            graph.mat = [[1, 5], [2, 5]]
        # test dtype check: only numeric numpy array allowed
        with self.assertRaises(TypeError):
            graph.mat = np.ones((2, 2), dtype=str)
        # test ndim check: only 2D or empty matrix allowed
        with self.assertRaises(ValueError):
            graph.mat = np.ones((2, 2, 2))
        # test shape check: matrix should have same number of rows as the size of idmap
        with self.assertRaises(ValueError):
            graph.mat = np.ones((3, 2))
        graph.mat = np.random.random((2, 2))

    def test_get_edge(self):
        graph = DenseGraph.from_mat(
            self.case.data_mat[:, 1:],
            self.case.data_mat[:, 0].astype(int).astype(str).tolist(),
        )
        mat = self.case.data_mat[:, 1:]
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                node_id1 = graph.idmap.lst[i]
                node_id2 = graph.idmap.lst[j]
                self.assertEqual(graph.get_edge(node_id1, node_id2), mat[i, j])

    def test_from_edglst(self):
        graph = DenseGraph.from_edglst(
            self.case.tw_path,
            weighted=True,
            directed=False,
        )
        self.check_graph(graph)

    @parameterized.expand(
        [
            ([[0, 1, 0], [1, 0, 0], [1, 0, 0]], 3),
            ([[0, 1, 1], [1, 0, 0], [1, 0, 0]], 4),
        ],
    )
    def test_num_nodes(self, mat, answer):
        graph = DenseGraph.from_mat(np.array(mat), ["a", "b", "c"])
        self.assertEqual(answer, graph.num_edges)

    def test_from_mat(self):
        with self.subTest("From matrix with first column ids"):
            graph = DenseGraph.from_mat(
                self.case.data_mat[:, 1:],
                self.case.data_mat[:, 0].astype(int).astype(str).tolist(),
            )
            self.check_graph(graph)

        with self.subTest("Using idmap"):
            idmap = idhandler.IDmap()
            idmap.add_id("a")
            idmap.add_id("b")
            mat1 = np.random.random((2, 2))
            mat2 = np.random.random((3, 2))
            # test consistent size input, using idmap --> success
            DenseGraph.from_mat(mat1, idmap)
            # test consistent size input, using idlst --> success
            DenseGraph.from_mat(mat1, idmap.lst)
            # test inconsistent size input --> error
            self.assertRaises(
                ValueError,
                DenseGraph.from_mat,
                mat2,
                idmap,
            )

        with self.subTest("No ids specified."):
            mat = np.random.random((5, 3))
            graph = DenseGraph.from_mat(mat)
            self.assertEqual(graph.idmap.lst, ["0", "1", "2", "3", "4"])

    def test_eq(self):
        graph = DenseGraph.from_edglst(
            self.case.tw_path,
            weighted=True,
            directed=False,
        )
        for i in range(5):
            # repeat shuffle test
            with self.subTest(i=i):
                self.assertTrue(graph == shuffle_dense(graph))
        graph2 = deepcopy(graph)
        graph2.mat[2, 2] = 1
        self.assertFalse(graph == graph2)

    def test_to_sparse_graph(self):
        graph = DenseGraph.from_mat(
            self.case.data_mat[:, 1:],
            self.case.data_mat[:, 0].astype(int).astype(str).tolist(),
        ).to_sparse_graph()
        self.assertIsInstance(graph, SparseGraph)
        self.assertEqual(
            graph._edge_data,
            [{2: 0.4}, {3: 0.3}, {0: 0.4, 4: 0.1}, {1: 0.3}, {2: 0.1}],
        )

    @parameterized.expand(
        [
            (["4", "3", "2"], [[0, 0, 0.3], [0, 0, 0], [0.3, 0, 0]]),
            (["4", "3", "2"], [[0, 0, 0.3], [0, 0, 0], [0.3, 0, 0]]),
        ],
    )
    def test_induced_subgraph(self, node_ids, adjmat):
        graph = DenseGraph.from_mat(
            self.case.data_mat[:, 1:],
            self.case.data_mat[:, 0].astype(int).astype(str).tolist(),
        )
        subgraph = graph.induced_subgraph(node_ids)
        self.assertEqual(subgraph.mat.tolist(), adjmat)

    def test_connected_components(self):
        graph = DenseGraph.from_mat(
            np.array(
                [
                    [0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 2, 1],
                    [0, 0, 2, 0, 1],
                    [0, 0, 1, 1, 0],
                ],
            ),
            ["a", "b", "c", "d", "e"],
        )

        # Two connected components
        self.assertEqual(
            graph.connected_components(),
            [["c", "d", "e"], ["a", "b"]],
        )
        self.assertFalse(graph.is_connected())

        # Largest connected subgraph
        subgraph = graph.largest_connected_subgraph()
        self.assertTrue(subgraph.is_connected())
        self.assertEqual(subgraph.node_ids, ("c", "d", "e"))
        self.assertEqual(
            subgraph.mat.tolist(),
            [[0, 2, 1], [2, 0, 1], [1, 1, 0]],
        )


class TestOntologyGraph(unittest.TestCase):
    def test_edge_stats(self):
        graph = OntologyGraph()
        self.assertEqual(graph._edge_stats, [])

        graph.add_node("a")
        self.assertEqual(graph._edge_stats, [0])

        graph.add_node("b")
        self.assertEqual(graph._edge_stats, [0, 0])

        graph.add_edge("b", "a")
        self.assertEqual(graph._edge_stats, [1, 0])

        graph.add_edge("c", "a")
        self.assertEqual(graph._edge_stats, [2, 0, 0])

    def test_node_name(self):
        graph = OntologyGraph()

        graph.add_node("a")
        self.assertEqual(graph.get_node_name("a"), None)

        graph.set_node_name("a", "A")
        self.assertEqual(graph.get_node_name("a"), "A")
        self.assertEqual(graph.get_node_name(0), "A")

    def test_node_attr(self):
        graph = OntologyGraph()

        graph.add_node("a")
        graph.add_node("b")
        self.assertEqual(graph.get_node_attr("a"), None)
        self.assertEqual(graph.get_node_attr("b"), None)

        graph.set_node_attr("a", ["x", "y", "z"])
        self.assertEqual(graph.get_node_attr("a"), ["x", "y", "z"])
        self.assertEqual(graph.get_node_attr(0), ["x", "y", "z"])

        graph.update_node_attr("b", "x")
        self.assertEqual(graph.get_node_attr("b"), ["x"])
        graph.update_node_attr("b", "x")
        self.assertEqual(graph.get_node_attr("b"), ["x"])
        graph.update_node_attr("b", "z")
        self.assertEqual(graph.get_node_attr("b"), ["x", "z"])
        graph.update_node_attr("b", ["a", "y"])
        self.assertEqual(graph.get_node_attr("b"), ["a", "x", "y", "z"])

        graph._update_node_attr_partial("a", ["a", "x"])
        self.assertEqual(graph.get_node_attr("a"), ["x", "y", "z", "a", "x"])
        graph._update_node_attr_partial("b", "x")
        self.assertEqual(graph.get_node_attr("b"), ["a", "x", "y", "z", "x"])

        graph._update_node_attr_finalize("a")
        self.assertEqual(graph.get_node_attr("a"), ["a", "x", "y", "z"])
        self.assertEqual(graph.get_node_attr("b"), ["a", "x", "y", "z", "x"])

        graph._update_node_attr_partial("a", ["a", "x"])
        graph._update_node_attr_finalize()
        self.assertEqual(graph.get_node_attr("a"), ["a", "x", "y", "z"])
        self.assertEqual(graph.get_node_attr("b"), ["a", "x", "y", "z"])

    def test_complete_node_attrs(self):
        r"""

                a

        /       |        \
        b       c [x, y]  d [x]
        |       |
        e [w]   f [z]

        """
        graph = OntologyGraph()

        graph.add_nodes(["a", "b", "c", "d", "e", "f"])

        graph.add_edge("b", "a")
        graph.add_edge("c", "a")
        graph.add_edge("d", "a")
        graph.add_edge("e", "b")
        graph.add_edge("f", "c")

        self.assertEqual(
            graph._rev_edge_data,
            [{1: 1, 2: 1, 3: 1}, {4: 1}, {5: 1}, {}, {}, {}],
        )

        self.assertEqual(
            graph._edge_data,
            [{}, {0: 1}, {0: 1}, {0: 1}, {1: 1}, {2: 1}],
        )

        graph.set_node_attr("d", ["x"])
        graph.set_node_attr("c", ["x", "y"])
        graph.set_node_attr("f", ["z"])
        graph.set_node_attr("e", ["w"])

        graph.propagate_node_attrs()

        self.assertEqual(graph.get_node_attr("a"), ["w", "x", "y", "z"])
        self.assertEqual(graph.get_node_attr("b"), ["w"])
        self.assertEqual(graph.get_node_attr("c"), ["x", "y", "z"])
        self.assertEqual(graph.get_node_attr("d"), ["x"])
        self.assertEqual(graph.get_node_attr("e"), ["w"])
        self.assertEqual(graph.get_node_attr("f"), ["z"])

        # Test post complete_node_attrs after introducing new changes
        graph.add_edge("g", "e")
        graph.set_node_attr("g", ["z"])
        graph.propagate_node_attrs()
        self.assertEqual(graph.get_node_attr("a"), ["w", "x", "y", "z"])
        self.assertEqual(graph.get_node_attr("b"), ["w", "z"])
        self.assertEqual(graph.get_node_attr("c"), ["x", "y", "z"])
        self.assertEqual(graph.get_node_attr("d"), ["x"])
        self.assertEqual(graph.get_node_attr("e"), ["w", "z"])
        self.assertEqual(graph.get_node_attr("f"), ["z"])
        self.assertEqual(graph.get_node_attr("g"), ["z"])

    def test_ancestors(self):
        r"""

           a
        /  |  \
        b  c   d
        |  |  /
        e  f

        """
        graph = OntologyGraph()

        graph.add_nodes(["a", "b", "c", "d", "e", "f"])

        graph.add_edge("b", "a")
        graph.add_edge("c", "a")
        graph.add_edge("d", "a")
        graph.add_edge("e", "b")
        graph.add_edge("f", "c")
        graph.add_edge("f", "d")

        self.assertEqual(graph.ancestors("a"), set())
        self.assertEqual(graph.ancestors("b"), {"a"})
        self.assertEqual(graph.ancestors("c"), {"a"})
        self.assertEqual(graph.ancestors("d"), {"a"})
        self.assertEqual(graph.ancestors("e"), {"a", "b"})
        self.assertEqual(graph.ancestors("f"), {"a", "c", "d"})

    def test_restrict_to_branch(self):
        r"""

        a
        | \
        b   d
        |   |     \
        c   e [z]   f [x, y]

        |
        V

        d
        |    \
        e [z]  f [x, y]


        """
        graph = OntologyGraph()

        graph.add_nodes(["a", "b", "c", "d", "e", "f"])

        graph.add_edge("b", "a")
        graph.add_edge("c", "b")
        graph.add_edge("d", "a")
        graph.add_edge("e", "d")
        graph.add_edge("f", "d")

        graph.set_node_attr("e", ["z"])
        graph.set_node_attr("f", ["x", "y"])

        subgraph = graph.restrict_to_branch("d")
        self.assertEqual(subgraph.get_node_attr("e"), ["z"])
        self.assertEqual(subgraph.get_node_attr("f"), ["x", "y"])
        self.assertEqual(sorted(subgraph.node_ids), ["d", "e", "f"])

        subgraph_exclusive = graph.restrict_to_branch("d", inclusive=False)
        self.assertEqual(sorted(subgraph_exclusive.node_ids), ["e", "f"])

        self.assertRaises(IDNotExistError, graph.restrict_to_branch, "g")

    def test_read_obo(self):
        r"""A /      |       \ b      c (x, y)  d |       \      / e [z]       f."""
        obo_path = osp.join(SAMPLE_DATA_DIR, "toy_ontology.obo")
        with self.subTest(xref_prefix=None):
            # Do not capture xref when xref_prefix is unset
            graph = OntologyGraph()
            out = graph.read_obo(obo_path)

            self.assertEqual(
                list(graph.node_ids),
                ["ID:0", "ID:1", "ID:2", "ID:3", "ID:4", "ID:5"],
            )
            self.assertEqual(
                list(map(graph.get_node_name, graph.node_ids)),
                ["a", "b", "c", "d", "e", "f"],
            )

            self.assertEqual(
                out,
                {
                    "ID:0": {"ID:0"},
                    "ID:1": {"ID:1"},
                    "ID:2": {"ID:2"},
                    "ID:3": {"ID:3"},
                    "ID:4": {"ID:4"},
                    "ID:5": {"ID:5"},
                },
            )
            self.assertEqual(
                graph._rev_edge_data,
                [{1: 1, 2: 1, 3: 1}, {4: 1}, {5: 1}, {5: 1}, {}, {}],
            )

        with self.subTest(xref_prefix="ALIAS"):
            graph = OntologyGraph()
            out = graph.read_obo(obo_path, xref_prefix="ALIAS")

            self.assertEqual(
                out,
                {
                    "ALIAS:x": {"ID:2"},
                    "ALIAS:y": {"ID:2"},
                    "ALIAS:z": {"ID:4"},
                    "ID:0": {"ID:0"},
                    "ID:1": {"ID:1"},
                    "ID:2": {"ID:2"},
                    "ID:3": {"ID:3"},
                    "ID:4": {"ID:4"},
                    "ID:5": {"ID:5"},
                },
            )
            self.assertEqual(
                graph._rev_edge_data,
                [{1: 1, 2: 1, 3: 1}, {4: 1}, {5: 1}, {5: 1}, {}, {}],
            )

        with self.subTest(xref_prefix="ALIAS2"):
            # No matched xref with the defined prefix
            graph = OntologyGraph()
            out = graph.read_obo(obo_path, xref_prefix="ALIAS2")

            self.assertEqual(
                out,
                {
                    "ID:0": {"ID:0"},
                    "ID:1": {"ID:1"},
                    "ID:2": {"ID:2"},
                    "ID:3": {"ID:3"},
                    "ID:4": {"ID:4"},
                    "ID:5": {"ID:5"},
                },
            )
            self.assertEqual(
                graph._rev_edge_data,
                [{1: 1, 2: 1, 3: 1}, {4: 1}, {5: 1}, {5: 1}, {}, {}],
            )


if __name__ == "__main__":
    unittest.main()
