import pytest

try:
    from obnb.ext.grape import embedders, grape_embed, grape_graph_from_obnb_sparse
except ModuleNotFoundError:
    pass


@pytest.mark.ext
def test_grape_graph_from_obnb_sparse(toy_graph_1, subtests):
    with subtests.test(weighted=True, directed=False):
        gpg = grape_graph_from_obnb_sparse(toy_graph_1)
        assert gpg.get_node_names() == list(toy_graph_1.node_ids)

        for src, dst, weight in toy_graph_1.edge_gen(compact=False):
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
def test_grape_embed(toy_graph_1, model_name, subtests):
    embedder = getattr(embedders, model_name)(embedding_size=5)
    grape_embed(toy_graph_1, embedder)
    grape_embed(toy_graph_1, embedder, as_array=True)
    grape_embed(toy_graph_1, model_name, dim=5)
