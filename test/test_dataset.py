import numpy as np
import pytest

from obnb.dataset.base import Dataset
from obnb.exception import IDNotExistError
from obnb.feature import MultiFeatureVec
from obnb.graph.dense import DenseGraph
from obnb.label.collection import LabelsetCollection


class Data:
    """Setup toy multi-feature vector object.

    .. code-block::

           f1  f2  f3
        a   1   2   3
        b   2   3   4
        c   3   4   5
        d   4   5   6
        e   5   6   7

    """

    raw_data = raw_data = {
        "a": [1, 2, 3],
        "b": [2, 3, 4],
        "c": [3, 4, 5],
        "d": [4, 5, 6],
        "e": [5, 6, 7],
    }
    ids = sorted(raw_data)
    fset_ids = fset_ids = ["f1", "f2", "f3"]
    raw_data_list = list(map(raw_data.get, ids))

    mat = np.vstack(raw_data_list)
    indptr = np.array([0, 1, 2, 3])
    feature = MultiFeatureVec.from_mat(
        mat,
        ids,
        indptr=indptr,
        fset_ids=fset_ids,
    )

    adj_mat = np.array(
        [
            [0, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ],
    )
    graph = DenseGraph.from_mat(adj_mat, ids)

    lsc = LabelsetCollection()
    lsc.add_labelset(ids, "labelset1")


@pytest.fixture
def data():
    return Data()


def test_set_idmap(data):
    # Test construction
    Dataset(graph=data.graph, label=data.lsc)
    Dataset(feature=data.feature, label=data.lsc)
    Dataset(feature=data.graph.to_feature(), label=data.lsc)
    pytest.raises(ValueError, Dataset, graph=data.graph)  # requires label
    pytest.raises(TypeError, Dataset, data.graph, label=data.lsc)  # only takes kwargs

    # Test set IDmap
    dataset = Dataset(graph=data.graph, feature=data.feature, label=data.lsc)
    assert dataset.idmap.lst == data.graph.idmap.lst
    assert dataset.idmap.lst == data.feature.idmap.lst

    # Wrong types
    pytest.raises(TypeError, Dataset, feature=data.graph, label=data.lsc)
    pytest.raises(TypeError, Dataset, graph=data.feature, label=data.lsc)

    dataset = Dataset(graph=data.graph, label=data.lsc)
    assert dataset.idmap.lst == data.graph.idmap.lst

    dataset = Dataset(feature=data.graph.to_feature(), label=data.lsc)
    assert dataset.idmap.lst == data.graph.idmap.lst

    dataset = Dataset(feature=data.feature, label=data.lsc)
    assert dataset.idmap.lst == data.feature.idmap.lst

    # Remove "d"
    data.graph.idmap.pop_id("d")
    assert data.graph.idmap.lst == ["a", "b", "c", "e"]
    with pytest.raises(ValueError, match="Misaligned IDs between graph and feature"):
        Dataset(graph=data.graph, feature=data.feature, label=data.lsc)

    # Reorder ids to ["a", "b", "c", "e", "d"]
    data.graph.idmap.add_id("d")
    assert data.graph.idmap.lst == ["a", "b", "c", "e", "d"]
    with pytest.raises(ValueError, match="Misaligned IDs between graph and feature"):
        Dataset(graph=data.graph, feature=data.feature, label=data.lsc)


def test_get_feat_from_idxs(subtests, data):
    dataset = Dataset(feature=data.feature, label=data.lsc)

    # Test get multiple featvecs
    test_list = [[0, 2], [3], [0, 1, 2, 3, 4]]
    for idxs in test_list:
        answer = [data.raw_data_list[i] for i in idxs]
        with subtests.test(idxs=idxs):
            assert dataset.get_feat(idxs, mode="idxs").tolist() == answer

    # Test get single featvec
    for idx in range(data.feature.size):
        answer = data.raw_data_list[idx]
        with subtests.test(idx=idx):
            assert dataset.get_feat(idx, mode="idxs").tolist() == answer

    # Index out of range
    pytest.raises(IndexError, dataset.get_feat, [3, 5])

    dataset = Dataset(graph=data.graph, label=data.lsc, auto_generate_feature=None)
    with pytest.raises(ValueError, match="feature not set"):
        dataset.get_feat([0, 1])


def test_get_feat_from_ids(subtests, data):
    dataset = Dataset(feature=data.feature, label=data.lsc)
    test_list = [["a", "c"], ["d"], ["a", "b", "c", "d", "e"]]

    # Test get multiple featvecs
    for ids in test_list:
        answer = [data.raw_data[i] for i in ids]
        with subtests.test(ids=ids):
            assert dataset.get_feat(ids, mode="ids").tolist() == answer

    # Test get single featvec
    for id_ in data.ids:
        answer = data.raw_data[id_]
        with subtests.test(id_=id_):
            assert dataset.get_feat(id_, mode="ids").tolist() == answer

    # Unknown node id "f"
    pytest.raises(IDNotExistError, dataset.get_feat, ["a", "f"], mode="ids")


def test_get_feat_from_mask(subtests, data):
    dataset = Dataset(feature=data.feature, label=data.lsc)
    test_list = [[1, 0, 1, 0, 0], [0, 0, 0, 1, 0], [1, 1, 1, 1, 1]]
    for mask in test_list:
        with subtests.test(mask=mask):
            assert dataset.get_feat(np.array(mask), mode="mask").tolist() == [
                data.raw_data_list[i] for i in np.where(mask)[0]
            ]

    # Incorrect mask size
    with pytest.raises(ValueError):
        dataset.get_feat(np.array([1, 0, 1, 0, 0, 0]), mode="mask")


def test_get_feat_auto(data):
    dataset = Dataset(feature=data.feature, label=data.lsc)
    idx = 0
    id_ = data.ids[idx]
    mask = np.zeros(len(data.ids))
    mask[idx] = 1
    answer = data.raw_data_list[idx]

    assert dataset.get_feat(idx, mode="auto").tolist() == answer
    assert dataset.get_feat(id_, mode="auto").tolist() == answer
    assert dataset.get_feat(mask, mode="auto").tolist() == [answer]

    pytest.raises(ValueError, dataset.get_feat, idx, mode="something")
    pytest.raises(ValueError, dataset.get_feat, [0, "1"], mode="auto")
    pytest.raises(ValueError, dataset.get_feat, np.array([0, 1, 0]), mode="auto")


@pytest.mark.xfail
def test_get_feat_fraom_idxs_dual(subtests, data):
    dataset = Dataset(feature=data.feature, dual=True, label=data.lsc)
    test_list = [[0, 2], [1], [0, 1, 2]]
    for idx in test_list:
        answer = [data.feature.mat[:, i].tolist() for i in idx]
        with subtests.test(idx=idx):
            assert dataset.get_feat(idx).tolist() == answer

    # Index out of range
    pytest.raises(IndexError, dataset.get_feat, [1, 3])

    # Dual mode should only work when with MultiFeatureVec
    with pytest.raises(
        TypeError,
        match="'dual' mode only works when the feature is of type MultiFeatureVec, "
        "but received type <class 'obnb.graph.dense.DenseGraph'>",
    ):
        dataset = Dataset(feature=data.graph.to_feature(), dual=True, label=data.lsc)

    # Dual mode should only work when all feature sets are one-dimensioanl
    fvec = MultiFeatureVec.from_mats(
        [np.random.random((10, 1)), np.random.random((10, 1))],
    )
    dataset = Dataset(feature=fvec, dual=True, label=data.lsc)
    with pytest.raises(
        ValueError,
        match="'dual' mode only works when the MultiFeatureVec only contains "
        "one-dimensional feature sets.",
    ):
        fvec = MultiFeatureVec.from_mats(
            [np.random.random((10, 1)), np.random.random((10, 2))],
        )
        dataset = Dataset(feature=fvec, dual=True, label=data.lsc)


@pytest.mark.xfail
def test_get_feat_from_ids_dual(subtests, data):
    dataset = Dataset(feature=data.feature, dual=True, label=data.lsc)
    test_ids_list = [["f1", "f2"], ["f1"], ["f1", "f2", "f3"]]
    test_idx_list = [[0, 1], [0], [0, 1, 2]]
    for ids, idx in zip(test_ids_list, test_idx_list):
        answer = [data.feature.mat[:, i].tolist() for i in idx]
        with subtests.test(ids=ids):
            assert dataset.get_feat_from_ids(ids).tolist() == answer

    # Unknown node id "f4"
    pytest.raises(IDNotExistError, dataset.get_feat_from_ids, ["f2", "f4"])


@pytest.mark.xfail
def test_get_feat_from_mask_dual(subtests, data):
    dataset = Dataset(feature=data.feature, dual=True, label=data.lsc)
    test_list = [[1, 0, 0], [1, 0, 1], [1, 1, 1]]
    fmat = data.feature.mat
    for mask in test_list:
        answer = [fmat[:, i].tolist() for i in np.where(mask)[0]]
        with subtests.test(mask=mask):
            assert dataset.get_feat(np.array(mask), mode="mask").tolist() == answer

    # Incorrect mask size
    pytest.raises(ValueError, dataset.get_feat, np.array([1, 0, 1, 0]), mode="mask")
