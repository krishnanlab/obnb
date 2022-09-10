import numpy as np
import pytest
import sklearn.metrics

from nleval.metric.standard import auroc, prior


@pytest.fixture
def case1():
    y_pred = np.array([1, 2, 3, 4])
    y_true = np.array([0, 0, 1, 1])
    return y_true, y_pred


@pytest.fixture
def case2():
    y_pred = np.array([1, 2, 3, 4])
    y_true = np.array([1, 1, 0, 0])
    return y_true, y_pred


@pytest.fixture
def rand_cases():
    rand_pairs = []
    for i in range(10):
        y_pred = np.random.random(100)
        y_true = np.random.random(100) > (i + 1) / 10
        y_true[0], y_true[1] = 0, 1  # make sure have two classes
        rand_pairs.append((y_true, y_pred))

    return rand_pairs


@pytest.fixture
def rand_cases_multi():
    rand_pairs = []
    for i in range(10):
        y_pred = np.random.random((100, 5))
        y_true = np.random.random((100, 5)) > (i + 1) / 10
        y_true[0], y_true[1] = 0, 1  # make sure have two classes
        rand_pairs.append((y_true, y_pred))

    return rand_pairs


def test_prior():
    assert prior(np.array([0, 0, 0, 0])) == 0
    assert prior(np.array([0, 1, 0, 0])) == 1 / 4
    assert prior(np.array([0, 1, 0, 0, 1, 1])) == 3 / 6
    assert prior(np.array([1, 1, 1, 1])) == 1


def test_auroc(case1, case2, rand_cases):
    assert auroc(*case1) == 1
    assert auroc(*case2) == 0

    for case in rand_cases:
        assert auroc(*case) == sklearn.metrics.roc_auc_score(*case)


def test_auroc_multi(rand_cases_multi):
    for y_true, y_pred in rand_cases_multi:
        skl_aurocs = [
            sklearn.metrics.roc_auc_score(i, j) for i, j in zip(y_true.T, y_pred.T)
        ]
        assert auroc(y_true, y_pred, reduce="none").tolist() == skl_aurocs
        assert auroc(y_true, y_pred, reduce="mean") == np.mean(skl_aurocs)
        assert auroc(y_true, y_pred, reduce="median") == np.median(skl_aurocs)
