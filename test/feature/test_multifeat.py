import numpy as np
import pytest

from obnb.feature import MultiFeatureVec


class Data:
    rng = np.random.default_rng(0)
    dims = [3, 2, 4]
    indptr = np.array([0, 3, 5, 9])
    mat1 = rng.random((5, dims[0]))
    mat2 = rng.random((5, dims[1]))
    mat3 = rng.random((5, dims[2]))
    mats = [mat1, mat2, mat3]
    mat = np.hstack(mats)
    ids = ["a", "b", "c", "d", "e"]
    fset_ids = ["Features1", "Features2", "Features3"]


@pytest.fixture
def data():
    return Data()


def test_from_mat(data):
    mfv = MultiFeatureVec.from_mat(
        data.mat,
        data.ids,
        indptr=data.indptr,
        fset_ids=data.fset_ids,
    )
    assert mfv.mat.tolist() == data.mat.tolist()
    assert mfv.indptr.tolist() == data.indptr.tolist()
    assert mfv.idmap.lst == data.ids
    assert mfv.fset_idmap.lst == data.fset_ids

    # Implicit indptr setting
    fset_ids = list(map(str, range(data.mat.shape[1])))
    mfv = MultiFeatureVec.from_mat(data.mat, fset_ids=fset_ids)
    assert mfv.mat.tolist() == data.mat.tolist()
    assert mfv.indptr.tolist() == list(range(data.mat.shape[1] + 1))
    assert mfv.idmap.lst == list(map(str, range(data.mat.shape[0])))
    assert mfv.fset_idmap.lst == fset_ids

    # Cannot have both fset_ids and indptr set to None
    with pytest.raises(ValueError):
        MultiFeatureVec.from_mat(data.mat)

    # Mismatch between fset_ids dimensiona and matrix columns number
    with pytest.raises(ValueError):
        MultiFeatureVec.from_mat(data.mat, fset_ids=list(map(str, range(10))))


def test_from_mats(data):
    mfv = MultiFeatureVec.from_mats(data.mats, data.ids, fset_ids=data.fset_ids)
    assert mfv.mat.tolist() == data.mat.tolist()
    assert mfv.indptr.tolist() == data.indptr.tolist()
    assert mfv.idmap.lst == data.ids
    assert mfv.fset_idmap.lst == data.fset_ids


def test_get_features(subtests, data):
    mfv = MultiFeatureVec.from_mats(data.mats, data.ids, fset_ids=data.fset_ids)

    with subtests.test(ids="a", fset_ids="Features1"):
        assert mfv.get_features("a", "Features1").tolist() == [data.mat1[0].tolist()]

    with subtests.test(ids=["a"], fset_ids="Features1"):
        assert mfv.get_features(["a"], "Features1").tolist() == [data.mat1[0].tolist()]

    with subtests.test(ids=["a", "c"], fset_ids="Features1"):
        assert (
            mfv.get_features(["a", "c"], "Features1").tolist()
            == data.mat1[[0, 2]].tolist()
        )

    with subtests.test(ids=["a", "c"], fset_ids="Features3"):
        assert (
            mfv.get_features(["a", "c"], "Features3").tolist()
            == data.mat3[[0, 2]].tolist()
        )

    with subtests.test(ids="a", fset_ids=["Features3", "Features1"]):
        assert mfv.get_features("a", ["Features3", "Features1"]).tolist() == [
            data.mat[0, [5, 6, 7, 8, 0, 1, 2]].tolist(),
        ]

    with subtests.test(ids=["a", "c"], fset_ids=["Features3", "Features1"]):
        assert (
            mfv.get_features(["a", "c"], ["Features3", "Features1"]).tolist()
            == data.mat[[0, 2]][:, [5, 6, 7, 8, 0, 1, 2]].tolist()
        )

    with subtests.test(ids=None, fset_ids=["Features3", "Features1"]):
        assert (
            mfv.get_features(fset_ids=["Features3", "Features1"]).tolist()
            == data.mat[:, [5, 6, 7, 8, 0, 1, 2]].tolist()
        )

    with subtests.test(ids=None, fset_ids="Features3"):
        assert (
            mfv.get_features(fset_ids="Features3").tolist()
            == data.mat[:, [5, 6, 7, 8]].tolist()
        )

    with subtests.test(ids=["a", "c"], fset_ids=None):
        assert mfv.get_features(["a", "c"]).tolist() == data.mat[[0, 2]].tolist()

    with subtests.test(ids="c", fset_ids=None):
        assert mfv.get_features("c").tolist() == data.mat[[2]].tolist()
