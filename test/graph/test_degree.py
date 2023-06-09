import numpy as np
import pytest

from obnb.graph import DenseGraph, DirectedSparseGraph, SparseGraph


@pytest.fixture
def toy_adj():
    return np.array(
        [
            [0, 0.5, 0.5, 0],
            [0, 0, 0.5, 0],
            [0.5, 0.5, 0, 0.5],
            [0, 0, 0, 0],
        ],
    )


@pytest.fixture
def toy_sparse_graph(toy_adj):
    return SparseGraph.from_mat(toy_adj)


@pytest.fixture
def toy_directed_sparse_graph(toy_adj):
    return DirectedSparseGraph.from_mat(toy_adj)


@pytest.fixture
def toy_dense_graph(toy_adj):
    return DenseGraph.from_mat(toy_adj)


@pytest.mark.parametrize("graph_cls", [SparseGraph, DirectedSparseGraph, DenseGraph])
@pytest.mark.parametrize(
    "weighted,direction,ans",
    [
        (False, "out", [2, 1, 3, 0]),
        (True, "out", [1, 0.5, 1.5, 0]),
        (False, "in", [1, 2, 2, 1]),
        (True, "in", [0.5, 1, 1, 0.5]),
    ],
)
def test_sparse_deg(toy_adj, graph_cls, weighted, direction, ans):
    graph = graph_cls.from_mat(toy_adj)
    assert graph.degree(weighted=weighted, direction=direction).tolist() == ans
