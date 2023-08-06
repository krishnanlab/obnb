import numpy as np
import pytest

try:
    from obnb.ext.attnwalk import attnwalk_embed
except ModuleNotFoundError:
    pass


@pytest.mark.ext
def test_attnwalk_embed(toy_graph_1):
    attnwalk_embed(toy_graph_1, window_size=5, epochs=2, verbose=True)

    _, attn = attnwalk_embed(toy_graph_1, window_size=5, epochs=2, return_attn=True)
    assert isinstance(attn, np.ndarray)
    assert attn.shape == (5,)
