import pytest

from nleval.graph.sparse import DirectedSparseGraph, SparseGraph


@pytest.fixture
def toy_graph():
    g = DirectedSparseGraph()
    g.add_edge("a", "b", 1.0)
    g.add_edge("b", "c", 1.0)
    return g


def test_to_undirected_sparse_graph(subtests, toy_graph):
    with subtests.test("default no reduction"):
        g1 = toy_graph.copy()
        g2 = g1.to_undirected_sparse_graph()
        assert isinstance(g2, SparseGraph)
        assert not g2.directed
        assert g2.edge_data == [{1: 1.0}, {0: 1.0, 2: 1.0}, {1: 1.0}]
        assert g2.node_ids == g1.node_ids

    with subtests.test("mean reduction"):
        g1 = toy_graph.copy()
        g1.add_edge("b", "a", 0.5)
        g2 = g1.to_undirected_sparse_graph(reduction="mean")
        assert g2.edge_data == [{1: 0.75}, {0: 0.75, 2: 1.0}, {1: 1.0}]

    with subtests.test("max reduction"):
        g1 = toy_graph.copy()
        g1.add_edge("b", "a", 0.5)
        g2 = g1.to_undirected_sparse_graph(reduction="max")
        assert g2.edge_data == [{1: 1.0}, {0: 1.0, 2: 1.0}, {1: 1.0}]

    with subtests.test("none reduction catch"):
        g1 = toy_graph.copy()
        g1.add_edge("b", "a", 0.5)
        pytest.raises(ValueError, g1.to_undirected_sparse_graph)
