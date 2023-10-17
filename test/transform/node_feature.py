import pytest

from obnb import Dataset
from obnb.label import LabelsetCollection


@pytest.fixture
def toy_dataset_1(toy_graph_1):
    lsc = LabelsetCollection()
    lsc.add_labelset(["a", "b", "c", "d", "e"], "label1")
    lsc.add_labelset(["d", "e", "f", "g", "h"], "label2")
    return toy_graph_1, lsc


def _test_transform_nodefeat_template(transform, toy_dataset):
    graph, label = toy_dataset

    if transform == "Constant":
        emb_dim = 4
    elif transform == "Adj":
        emb_dim = None
    else:
        emb_dim = 4

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
    _test_transform_nodefeat_template(transform, toy_dataset_1)


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
    _test_transform_nodefeat_template(transform, toy_dataset_1)


@pytest.mark.ext
def test_transform_nodefeat_converted(toy_dataset_1):
    graph, label = toy_dataset_1

    dataset = Dataset(
        graph=graph,
        label=label,
        transform="OneHotLogDeg",
        transform_kwargs=dict(dim=4, as_feature=True),
        auto_generate_feature=None,
    )

    dataset_pyg = dataset.to_pyg_data()
    assert "nodefeat_OneHotLogDeg" in dataset_pyg
    print(f"PyG dataset:\n{dataset_pyg}\n")

    dataset_dgl = dataset.to_dgl_data()
    assert "nodefeat_OneHotLogDeg" in dataset_dgl.ndata
    print(f"DGL dataset:\n{dataset_dgl}\n")
