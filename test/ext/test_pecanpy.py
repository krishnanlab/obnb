import pytest

try:
    from obnb.ext.pecanpy import pecanpy_embed
except ModuleNotFoundError:
    pass


@pytest.mark.ext
@pytest.mark.parametrize(
    "mode",
    [
        "PreComp",
        "SparseOTF",
        "DenseOTF",
        "PreCompFirstOrder",
        "FirstOrderUnweighted",
    ],
)
def test_pecanpy_embed(toy_graph_1, mode):
    pecanpy_embed(
        toy_graph_1,
        mode="SparseOTF",
        dim=10,
        num_walks=2,
        walk_length=5,
        window_size=2,
        verbose=True,
    )


@pytest.mark.ext
def test_pecanpy_embed_err(toy_graph_1):
    pytest.raises(ValueError, pecanpy_embed, toy_graph_1, mode="NonSense")
    pytest.raises(TypeError, pecanpy_embed, toy_graph_1, mode=123)
    pytest.raises(TypeError, pecanpy_embed, 123)
