import grape
import pytest

from nleval.ext.grape import grape_embed, grape_graph_from_nleval_sparse
from nleval.graph import SparseGraph

toy_edge_list = [
    ["a", "b", 0.3],
    ["b", "c", 0.8],
    ["c", "d", 1],
    ["d", "e", 0.2],
    ["e", "f", 0.2],
    ["f", "g", 0.2],
    ["g", "h", 0.2],
]
toy_num_edges = len(toy_edge_list)
toy_edge_weights = [i[-1] for i in toy_edge_list]

toy_graph = SparseGraph(weighted=True, directed=False)
for i, j, k in toy_edge_list:
    toy_graph.add_edge(i, j, k)


@pytest.mark.ext
def test_grape_graph_from_nleval_sparse(subtests):
    with subtests.test(weighted=True, directed=False):
        gpg = grape_graph_from_nleval_sparse(toy_graph)
        assert gpg.get_node_names() == list(toy_graph.node_ids)

        for src, dst, weight in toy_graph.edge_gen(compact=False):
            gpg_weight = gpg.get_edge_weight_from_node_names(src, dst)
            assert pytest.approx(gpg_weight) == weight


@pytest.mark.ext
@pytest.mark.parametrize(
    "model_name",
    [
        "FirstOrderLINEEnsmallen",
        "SecondOrderLINEEnsmallen",
        "DeepWalkCBOWEnsmallen",
        "DeepWalkGloVeEnsmallen",
        "DeepWalkSkipGramEnsmallen",
        "HOPEEnsmallen",
        "LaplacianEigenmapsEnsmallen",
        "Node2VecCBOWEnsmallen",
        "Node2VecGloVeEnsmallen",
        "Node2VecSkipGramEnsmallen",
        "SocioDimEnsmallen",
        "UnstructuredEnsmallen",
        "WalkletsCBOWEnsmallen",
        "WalkletsGloVeEnsmallen",
        "WalkletsSkipGramEnsmallen",
        "WeightedSPINE",
        "DegreeSPINE",
        "DegreeWINE",
        "ScoreSPINE",
        "ScoreWINE",
    ],
)
def test_grape_embed(model_name, subtests):
    embedder = getattr(grape.embedders, model_name)(embedding_size=5)
    grape_embed(toy_graph, embedder)
    grape_embed(toy_graph, embedder, as_array=True)
    grape_embed(toy_graph, model_name, embedding_size=5)
