import os.path as osp

import numpy as np
import pytest

from obnb.feature import FeatureVec


def test_dim():
    feat = FeatureVec()

    feat.dim = int(10)
    assert feat.dim == 10

    feat.dim = int(9)
    assert feat.dim == 9

    feat.dim = np.int64(10)
    assert feat.dim == 10

    with pytest.raises(ValueError):
        feat.dim = 0

    with pytest.raises(ValueError):
        feat.dim = int(-10)

    with pytest.raises(TypeError):
        feat.dim = float(5)
    assert feat.dim == 10

    with pytest.raises(TypeError):
        feat.dim = "5"
    assert feat.dim == 10

    with pytest.raises(TypeError):
        feat.dim = float(5)
    assert feat.dim == 10


def test_mat():
    feat = FeatureVec()
    feat.idmap.add_id("a")
    feat.idmap.add_id("b")
    feat.idmap.add_id("c")
    mat1 = np.random.random((3, 5))
    mat2 = np.random.random((5, 7))
    mat3 = np.random.random((5, 5))

    # Test if dim set automatically
    assert feat.dim is None
    feat.mat = mat1
    assert feat.dim == 5

    # Test if mat must match dim
    feat.idmap.add_id("d")
    feat.idmap.add_id("e")
    with pytest.raises(ValueError):
        feat.mat = mat2

    # Test if matrix recovered if exception raised due to size inconsistency
    assert np.all(feat.mat == mat1)
    feat.mat = mat3


def test_add_featvec():
    feat = FeatureVec(dim=4)

    vec1 = np.array([1, 2, 3])
    vec2 = np.array([2, 4, 5])
    vec3 = np.array([3, 5, 6])

    # Test if input vec must match preset dim
    with pytest.raises(ValueError):
        feat.add_featvec("a", vec1)

    # Test if only add ID when vec constructed successfully
    assert feat.idmap.size == 0

    feat.dim = 3
    feat.add_featvec("a", vec1)
    feat.add_featvec("b", vec2)
    feat.add_featvec("c", vec3)
    assert feat.idmap.lst == ["a", "b", "c"]

    # Test if input vec must be numeric
    with pytest.raises(TypeError):
        feat.add_featvec("str", np.array(["1", "2", "3"]))

    # Test if only add_id when vec append to mat successfully
    assert feat.idmap.lst == ["a", "b", "c"]

    feat = FeatureVec()
    assert feat.dim is None
    feat.add_featvec("a", vec1)

    # Test if automatically set dim correctly
    assert feat.dim == 3

    # Test if captures inconsistency between number of IDs and number matrix entries
    feat.idmap.add_id("d")
    with pytest.raises(ValueError):
        feat.add_featvec("e", vec1)


def test_from_emd(tmpdir, commonvar):
    toy_emd_path = osp.join(commonvar.sample_data_dir, "toy1.emd")
    feat = FeatureVec.from_emd(toy_emd_path)
    temd_data = np.loadtxt(toy_emd_path, delimiter=" ", skiprows=1)[:, 1:]
    assert np.all(feat.mat == temd_data)
