import pytest

from nleval.label import LabelsetCollection
from nleval.label.filters.nonred import LabelsetNonRedFilter

# TODO: add test for construct_labelset_graph
# TODO: add test for get_nonred_label_ids
# TODO: pass log level from cli?

LOG_LEVEL = "INFO"


@pytest.fixture(scope="function")
def case1():
    r"""Labelset collection test case 1.

    When threshold is set to 0.49, and use overlap (or threshold set to 0.3
    and use jaccard):

        lb0
        |   \
        lb1  lb2
        |
        lb3

    The expected outcome is thus: ['lb0', 'lb3']

    """
    lsc = LabelsetCollection()

    lsc.add_labelset(["a", "b"], "lb0")
    lsc.add_labelset(["a", "c"], "lb1")
    lsc.add_labelset(["b", "d"], "lb2")
    lsc.add_labelset(["c", "d"], "lb3")

    return lsc


def test_nonred_jaccard(case1):
    filter_ = LabelsetNonRedFilter(0.3, 0.0, log_level=LOG_LEVEL)
    assert set(case1.apply(filter_).label_ids) == {"lb0", "lb3"}

    filter_ = LabelsetNonRedFilter(0.5, 0.0, log_level=LOG_LEVEL)
    assert set(case1.apply(filter_).label_ids) == {"lb0", "lb1", "lb2", "lb3"}


def test_nonred_overlap(case1):
    filter_ = LabelsetNonRedFilter(0.0, 0.49, log_level=LOG_LEVEL)
    assert set(case1.apply(filter_).label_ids) == {"lb0", "lb3"}

    filter_ = LabelsetNonRedFilter(0.0, 0.5, log_level=LOG_LEVEL)
    assert set(case1.apply(filter_).label_ids) == {"lb0", "lb1", "lb2", "lb3"}


def test_nonred_redratio(case1):
    filter_ = LabelsetNonRedFilter(0.5, 0.5, log_level=LOG_LEVEL)

    labelsets = list(map(case1.get_labelset, ["lb0", "lb1", "lb2", "lb3"]))
    assert filter_._get_repr_score(labelsets, 0) == 1
    assert filter_._get_repr_score(labelsets, 1) == 1
    assert filter_._get_repr_score(labelsets, 2) == 1
    assert filter_._get_repr_score(labelsets, 3) == 1

    labelsets = list(map(case1.get_labelset, ["lb0", "lb1"]))
    assert filter_._get_repr_score(labelsets, 0) == 1 / 2
    assert filter_._get_repr_score(labelsets, 1) == 1 / 2

    labelsets = list(map(case1.get_labelset, ["lb0", "lb2", "lb3"]))
    assert filter_._get_repr_score(labelsets, 0) == 1 / 2
    assert filter_._get_repr_score(labelsets, 1) == 1
    assert filter_._get_repr_score(labelsets, 2) == 1 / 2
