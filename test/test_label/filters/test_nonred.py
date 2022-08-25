import pytest

from NLEval.label import LabelsetCollection
from NLEval.label.filters import (
    LabelsetNonRedFilterJaccard,
    LabelsetNonRedFilterOverlap,
)

# TODO: add test for construct_labelset_graph
# TODO: add test for get_nonred_label_ids


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
    filter_ = LabelsetNonRedFilterJaccard(threshold=0.3)
    assert set(case1.apply(filter_).label_ids) == {"lb0", "lb3"}

    filter_ = LabelsetNonRedFilterJaccard(threshold=0.5)
    assert set(case1.apply(filter_).label_ids) == {"lb0", "lb1", "lb2", "lb3"}


def test_nonred_overlap(case1):
    filter_ = LabelsetNonRedFilterOverlap(threshold=0.49)
    assert set(case1.apply(filter_).label_ids) == {"lb0", "lb3"}

    filter_ = LabelsetNonRedFilterOverlap(threshold=0.5)
    assert set(case1.apply(filter_).label_ids) == {"lb0", "lb1", "lb2", "lb3"}
