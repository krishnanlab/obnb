import pytest

try:
    from nleval.ext.sknetwork import sknetwork_embed
except ModuleNotFoundError:
    pass


@pytest.mark.ext
@pytest.mark.parametrize("as_array", [True, False])
@pytest.mark.parametrize(
    "model_name",
    [
        "Spectral",
        "SVD",
        "GSVD",
        "PCA",
        "RandomProjection",
        "LouvainNE",
    ],
)
def test_sknetwork_embed(model_name, as_array, toy_graph_1):
    sknetwork_embed(toy_graph_1, model_name, dim=3, as_array=as_array)
