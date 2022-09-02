import numpy as np
import pytest

from NLEval.feature import FeatureVec
from NLEval.util.idhandler import IDmap


class Data:
    ids1 = ["a", "b", "c", "d"]
    ids2 = ["c", "b", "a", "e", "f"]
    ids_intersection = ["a", "b", "c"]
    ids_union = ["a", "b", "c", "d", "e", "f"]

    ids1_map = {"a": 0, "b": 1, "c": 2, "d": 3}
    ids2_map = {"c": 0, "b": 1, "a": 2, "e": 3, "f": 4}
    ids_intersection_map = {"a": 0, "b": 1, "c": 2}
    ids_union_map = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5}

    mat1 = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
    mat2 = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])

    fvec1 = FeatureVec.from_mat(mat1, ids1)
    fvec2 = FeatureVec.from_mat(mat2, ids2)


@pytest.fixture
def data():
    return Data()


def test_align_raises(data):
    pytest.raises(TypeError, data.fvec1.align, data.ids1)
    pytest.raises(ValueError, data.fvec1.align, None)


def test_align_to_idmap(data):
    fvec1, fvec2 = data.fvec1.copy(), data.fvec2.copy()
    idmap = IDmap.from_list(["b", "k", "a"])

    fvec1.align_to_idmap(idmap)
    assert fvec1.idmap.lst == idmap.lst
    assert fvec1.mat.tolist() == [[1, 2, 3], [0, 0, 0], [0, 1, 2]]

    fvec2.align_to_idmap(idmap)
    assert fvec2.idmap.lst == idmap.lst
    assert fvec2.mat.tolist() == [[1, 2], [0, 0], [2, 3]]


def test_align_right(data):
    fvec1, fvec2 = data.fvec1.copy(), data.fvec2.copy()
    fvec1.align(fvec2, join="right", update=True)

    assert fvec1.idmap.lst == data.ids2
    assert fvec2.idmap.lst == data.ids2

    assert fvec1.mat.tolist() == [[2, 3, 4], [1, 2, 3], [0, 1, 2], [0, 0, 0], [0, 0, 0]]
    assert fvec2.mat.tolist() == data.mat2.tolist()


def test_align_left(subtests, data):
    with subtests.test(update=False):
        fvec1, fvec2 = data.fvec1.copy(), data.fvec2.copy()
        fvec1.align(fvec2, join="left", update=False)

        assert fvec1.idmap.lst == data.ids1
        assert fvec2.idmap.lst == data.ids2

    with subtests.test(update=True):
        fvec1, fvec2 = data.fvec1.copy(), data.fvec2.copy()
        fvec1.align(fvec2, join="left", update=True)

        assert fvec1.idmap.lst == data.ids1
        assert fvec2.idmap.lst == data.ids1

        assert fvec1.mat.tolist() == data.mat1.tolist()
        assert fvec2.mat.tolist() == [[2, 3], [1, 2], [0, 1], [0, 0]]


def test_align_intersection(subtests, data):
    with subtests.test(update=False):
        fvec1, fvec2 = data.fvec1.copy(), data.fvec2.copy()
        fvec1.align(fvec2, join="intersection", update=False)

        assert fvec1.idmap.lst == data.ids_intersection
        assert fvec2.idmap.lst == data.ids2

        assert fvec1.mat.tolist() == [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
        assert fvec2.mat.tolist() == data.mat2.tolist()

    with subtests.test(update=True):
        fvec1, fvec2 = data.fvec1.copy(), data.fvec2.copy()
        fvec1.align(fvec2, join="intersection", update=True)

        assert fvec1.idmap.lst == data.ids_intersection
        assert fvec2.idmap.lst == data.ids_intersection

        assert fvec1.mat.tolist() == [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
        assert fvec2.mat.tolist() == [[2, 3], [1, 2], [0, 1]]


def test_align_union(subtests, data):
    with subtests.test(update=False):
        fvec1, fvec2 = data.fvec1.copy(), data.fvec2.copy()
        fvec1.align(fvec2, join="union", update=False)

        assert fvec1.idmap.lst == data.ids_union
        assert fvec2.idmap.lst == data.ids2

        assert fvec1.mat.tolist() == [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [0, 0, 0],
            [0, 0, 0],
        ]
        assert fvec2.mat.tolist() == data.mat2.tolist()

    with subtests.test(update=True):
        fvec1, fvec2 = data.fvec1.copy(), data.fvec2.copy()
        fvec1.align(fvec2, join="union", update=True)

        assert fvec1.idmap.lst == data.ids_union
        assert fvec2.idmap.lst == data.ids_union

        assert fvec1.mat.tolist() == [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [0, 0, 0],
            [0, 0, 0],
        ]
        assert fvec2.mat.tolist() == [[2, 3], [1, 2], [0, 1], [0, 0], [3, 4], [4, 5]]
