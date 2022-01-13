import os
import unittest
from copy import deepcopy

import numpy as np
from commonvar import SAMPLE_DATA_DIR
from NLEval.graph import DenseGraph
from NLEval.graph import FeatureVec
from NLEval.graph import SparseGraph
from NLEval.graph.base import BaseGraph
from NLEval.util import idhandler
from scipy.spatial import distance


def shuffle_sparse(graph):
    n = graph.size
    shuffle_idx = np.random.choice(n, size=n, replace=False)
    new_graph = SparseGraph(
        weighted=graph.weighted,
        directed=graph.directed,
    )
    for i in shuffle_idx:
        node_id = graph.idmap.lst[i]
        new_graph.add_id(node_id)
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
        self.tw_fp = os.path.join(SAMPLE_DATA_DIR, "toy1_weighted.edg")
        self.tu_fp = os.path.join(SAMPLE_DATA_DIR, "toy1_unweighted.edg")
        self.temd_fp = os.path.join(SAMPLE_DATA_DIR, "toy1.emd")
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


class TestBaseGraph(unittest.TestCase):
    def setUp(self):
        self.graph = BaseGraph()

    def test_idmap_setter(self):
        with self.assertRaises(TypeError):
            self.graph.idmap = "asdg"

    def test_size(self):
        self.assertEqual(self.graph.size, 0)
        for i in range(5):
            with self.subTest(i=i):
                self.graph.idmap.add_id(str(i))
                self.assertEqual(self.graph.size, i + 1)

    def test_isempty(self):
        self.assertTrue(self.graph.isempty())
        self.graph.idmap.add_id("a")
        self.assertFalse(self.graph.isempty())


class TestSparseGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = test_case1()
        cls.lst = [["1", "4"], ["2", "5"], ["5", "3", "2"]]

    def test_read_edglst_unweighted(self):
        graph = SparseGraph.from_edglst(
            self.case.tu_fp,
            weighted=False,
            directed=False,
        )
        self.assertEqual(graph.idmap.lst, self.case.IDlst)
        self.assertEqual(graph.edge_data, self.case.data_unweighted)

    def test_read_edglst_weighted(self):
        graph = SparseGraph.from_edglst(
            self.case.tw_fp,
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
        graph2.add_id("x")
        self.assertFalse(graph == graph2)


class TestDenseGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = test_case1()

    def check_graph(self, graph):
        """compare graph with data, true if identical"""
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
            self.case.tw_fp,
            weighted=True,
            directed=False,
        )
        self.check_graph(graph)

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

    def test_eq(self):
        graph = DenseGraph.from_edglst(
            self.case.tw_fp,
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


class TestFeatureVec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = test_case1()
        cls.vec_a = np.array([1, 2, 3])
        cls.vec_b = np.array([2, 4, 5])
        cls.vec_c = np.array([3, 5, 6])
        cls.vec_str = np.array(["1", "2", "3"])

    def test_dim(self):
        graph = FeatureVec()
        # test type int --> success
        graph.dim = int(10)
        self.assertEqual(graph.dim, 10)
        # test type np.int --> success
        graph.dim = int(9)
        self.assertEqual(graph.dim, 9)
        # test type np.int64 --> success
        graph.dim = np.int64(10)
        self.assertEqual(graph.dim, 10)
        # test int less than 1 --> error
        with self.assertRaises(ValueError):
            graph.dim = 0
        with self.assertRaises(ValueError):
            graph.dim = int(-10)
        # test type float --> error
        with self.assertRaises(TypeError):
            graph.dim = float(5)
        self.assertEqual(graph.dim, 10)
        # test type str --> error
        with self.assertRaises(TypeError):
            graph.dim = "5"
        self.assertEqual(graph.dim, 10)
        # test type np.float --> error
        with self.assertRaises(TypeError):
            graph.dim = float(5)
        self.assertEqual(graph.dim, 10)

    def test_mat(self):
        graph = FeatureVec()
        graph.idmap.add_id("a")
        graph.idmap.add_id("b")
        graph.idmap.add_id("c")
        mat1 = np.random.random((3, 5))
        mat2 = np.random.random((5, 7))
        mat3 = np.random.random((5, 5))
        # test if dim set automaticall
        self.assertEqual(graph.dim, None)
        graph.mat = mat1
        self.assertEqual(graph.dim, 5)
        # test if mat must match dim
        graph.idmap.add_id("d")
        graph.idmap.add_id("e")
        with self.assertRaises(ValueError):
            graph.mat = mat2
        # test if matrix recovered if exception raised due to size inconsistency
        self.assertTrue(np.all(graph.mat == mat1))
        graph.mat = mat3

    def test_get_edge(self):
        graph = FeatureVec.from_emd(self.case.temd_fp)
        temd_data = np.loadtxt(
            os.path.join(SAMPLE_DATA_DIR, "toy1.emd"),
            delimiter=" ",
            skiprows=1,
        )[:, 1:]
        for i, node_id1 in enumerate(graph.idmap):
            for j, node_id2 in enumerate(graph.idmap):
                calculated = distance.cosine(temd_data[i], temd_data[j])
                self.assertEqual(graph.get_edge(node_id1, node_id2), calculated)

    def test_add_vec(self):
        graph = FeatureVec(dim=4)
        # test if input vec must match preset dim
        self.assertRaises(ValueError, graph.add_vec, "a", self.vec_a)
        # test if only add ID when vec constructed successfully
        self.assertTrue(graph.idmap.size == 0)
        graph.dim = 3
        graph.add_vec("a", self.vec_a)
        graph.add_vec("b", self.vec_b)
        graph.add_vec("c", self.vec_c)
        self.assertEqual(graph.idmap.lst, ["a", "b", "c"])
        # test if input vec must be numeric
        self.assertRaises(TypeError, graph.add_vec, "str", self.vec_str)
        # test if only add_id when vec append to self.mat successfully
        self.assertEqual(graph.idmap.lst, ["a", "b", "c"])

        graph = FeatureVec()
        self.assertTrue(graph.dim is None)
        graph.add_vec("a", self.vec_a)
        # test if automatically set dim correctly
        self.assertEqual(graph.dim, 3)
        # test if captures inconsistency between number of IDs and number matrix entires
        graph.idmap.add_id("d")
        self.assertRaises(ValueError, graph.add_vec, "e", self.vec_a)

    def test_from_emd(self):
        graph = FeatureVec.from_emd(self.case.temd_fp)
        temd_data = np.loadtxt(
            os.path.join(SAMPLE_DATA_DIR, "toy1.emd"),
            delimiter=" ",
            skiprows=1,
        )[:, 1:]
        self.assertTrue(np.all(graph.mat == temd_data))


if __name__ == "__main__":
    unittest.main()
