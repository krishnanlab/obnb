import numpy as np
import pytest

from NLEval.graph.dense import DenseGraph
from NLEval.graph.sparse import DirectedSparseGraph, SparseGraph


def test_sprase_get_neighbors(subtests):
    g = SparseGraph(directed=False)
    g.add_nodes(["a", "b", "c", "d"])
    g.add_edge("a", "b")
    g.add_edge("a", "d")

    with subtests.test("Sparse graph (undirected)"):
        assert g.get_neighbors("a") == ["b", "d"]
        assert g.get_neighbors("a", "in") == ["b", "d"]
        assert g.get_neighbors("a", "out") == ["b", "d"]
        assert g.get_neighbors("b") == ["a"]
        assert g.get_neighbors("c") == []
        assert g.get_neighbors("d") == ["a"]

    # When the sparse graph is set to be directed, only allow getting out nbrs
    with subtests.test("Sparse graph (directed)"):
        g.directed = True
        assert g.get_neighbors("a", "out") == ["b", "d"]

        with pytest.raises(NotImplementedError):
            g.get_neighbors("a")
            g.get_neighbors("a", "in")
            g.get_neighbors("a", "both")


def test_dirsprase_get_neighbors(subtests):
    g = DirectedSparseGraph()
    g.add_nodes(["a", "b", "c", "d"])
    g.add_edge("a", "b")
    g.add_edge("b", "a")
    g.add_edge("a", "d")

    with subtests.test("Directed sparse graph 'a' neighbors"):
        assert g.get_neighbors("a") == ["b", "d"]
        assert g.get_neighbors("a", "both") == ["b", "d"]
        assert g.get_neighbors("a", "in") == ["b"]
        assert g.get_neighbors("a", "out") == ["b", "d"]

    with subtests.test("Directed sparse graph 'd' neighbors"):
        assert g.get_neighbors("d") == ["a"]
        assert g.get_neighbors("d", "both") == ["a"]
        assert g.get_neighbors("d", "in") == ["a"]
        assert g.get_neighbors("d", "out") == []


def test_dense_get_neighbors(subtests):
    adj = np.array(
        [
            [0, 1, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
    )
    g = DenseGraph.from_mat(adj, ["a", "b", "c", "d"])

    with subtests.test("Dense graph 'a' neighbors"):
        assert g.get_neighbors("a") == ["b", "d"]
        assert g.get_neighbors("a", "both") == ["b", "d"]
        assert g.get_neighbors("a", "in") == ["b"]
        assert g.get_neighbors("a", "out") == ["b", "d"]

    with subtests.test("Dense graph 'd' neighbors"):
        assert g.get_neighbors("d") == ["a"]
        assert g.get_neighbors("d", "both") == ["a"]
        assert g.get_neighbors("d", "in") == ["a"]
        assert g.get_neighbors("d", "out") == []
