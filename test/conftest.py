import os.path as osp
import pathlib

import pytest

from obnb.graph import SparseGraph


class CommonVar:
    home_dir = pathlib.Path(__file__).resolve().parent
    sample_data_dir = osp.join(home_dir, "sample_data")


@pytest.fixture
def commonvar():
    return CommonVar()


@pytest.fixture
def toy_graph_1():
    data = [
        ["a", "b", 0.3],
        ["b", "c", 0.8],
        ["c", "d", 1],
        ["d", "e", 0.2],
        ["e", "f", 0.2],
        ["f", "g", 0.2],
        ["g", "h", 0.2],
    ]

    g = SparseGraph(weighted=True, directed=False)
    for i, j, k in data:
        g.add_edge(i, j, k)

    return g
