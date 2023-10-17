import pytest

from obnb import Dataset
from obnb.label import LabelsetCollection


@pytest.fixture(scope="function")
def toy_dataset_1(toy_graph_1):
    lsc = LabelsetCollection()
    lsc.add_labelset(["a", "b", "c", "d", "e"], "label1")
    lsc.add_labelset(["d", "e", "f", "g", "h"], "label2")
    return toy_graph_1, lsc


@pytest.mark.parametrize(
    "transform",
    [
        "OneHotLogDeg",
        "Constant",
        "RandomNormal",
        "SVD",
        "LapEigMap",
        "RandomWalkDiag",
        "RandProjGaussian",
        "RandProjSparse",
        "Adj",
    ],
)
def test_transform_nodefeat_basic(transform, toy_dataset_1):
    graph, label = toy_dataset_1
    emb_dim = 1 if transform == "Constant" else 4
    dataset = Dataset(
        graph=graph,
        label=label,
        transform=transform,
        transform_kwargs=dict(dim=emb_dim, as_feature=True),
        auto_generate_feature=None,
    )
    print(f"{transform}:\n{dataset.feature.mat}\n")


@pytest.mark.parametrize(
    "transform",
    [
        "Orbital",
        "LINE1",
        "LINE2",
        "Node2vec",
        "Walklets",
        "AttnWalk",
    ],
)
@pytest.mark.ext
def test_transform_nodefeat_ext(transform, toy_dataset_1):
    graph, label = toy_dataset_1
    emb_dim = 1 if transform == "Constant" else 4
    dataset = Dataset(
        graph=graph,
        label=label,
        transform=transform,
        transform_kwargs=dict(dim=emb_dim, as_feature=True),
        auto_generate_feature=None,
    )
    print(f"{transform}:\n{dataset.feature.mat}\n")
