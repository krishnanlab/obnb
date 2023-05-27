import pandas as pd
import pytest

try:
    import networkx as nx

    from nleval.ext.orbital_features import OrbitCountingMachine, orbital_feat_extract
except ModuleNotFoundError:
    pass


KARATE_CLUB_ORBITAL_FEAT_4 = pd.DataFrame(
    [
        ["0", 16, 17, 102, 18, 0],
        ["1", 9, 19, 24, 12, 0],
        ["2", 10, 34, 34, 11, 0],
        ["3", 6, 20, 5, 10, 0],
        ["4", 3, 16, 1, 2, 0],
        ["5", 4, 15, 3, 3, 0],
        ["6", 4, 15, 3, 3, 0],
        ["7", 4, 25, 0, 6, 0],
        ["8", 5, 44, 5, 5, 0],
        ["9", 2, 25, 1, 0, 0],
        ["10", 3, 16, 1, 2, 0],
        ["11", 1, 15, 0, 0, 0],
        ["12", 2, 18, 0, 1, 0],
        ["13", 5, 41, 4, 6, 0],
        ["14", 2, 25, 0, 1, 0],
        ["15", 2, 25, 0, 1, 0],
        ["16", 2, 4, 0, 1, 0],
        ["17", 2, 21, 0, 1, 0],
        ["18", 2, 25, 0, 1, 0],
        ["19", 3, 37, 2, 1, 0],
        ["20", 2, 25, 0, 1, 0],
        ["21", 2, 21, 0, 1, 0],
        ["22", 2, 25, 0, 1, 0],
        ["23", 5, 27, 6, 4, 0],
        ["24", 3, 8, 2, 1, 0],
        ["25", 3, 9, 2, 1, 0],
        ["26", 2, 17, 0, 1, 0],
        ["27", 4, 29, 5, 1, 0],
        ["28", 3, 28, 2, 1, 0],
        ["29", 4, 24, 2, 4, 0],
        ["30", 4, 33, 3, 3, 0],
        ["31", 6, 42, 12, 3, 0],
        ["32", 12, 23, 53, 13, 0],
        ["33", 17, 18, 121, 15, 0],
    ],
    columns=["id", "role_0", "role_1", "role_2", "role_3", "role_4"],
).set_index("id")


@pytest.mark.ext
def test_orbital_features_karate():
    g = nx.karate_club_graph()
    orb = OrbitCountingMachine(g, graphlet_size=3)

    feat = orb.extract_features().sort_index()
    assert KARATE_CLUB_ORBITAL_FEAT_4.values.tolist() == feat.values.tolist()


@pytest.mark.ext
def test_orbital_features_toy(toy_graph_1):
    orbital_feat_extract(toy_graph_1, graphlet_size=3, as_array=True)
