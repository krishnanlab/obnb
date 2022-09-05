import pytest

from nleval.typing import Literal
from nleval.util.checkers import checkLiteral


def test_check_literal():
    literals = Literal["a", "b", "abc"]

    checkLiteral("a", literals, "a")
    checkLiteral("b", literals, "b")
    checkLiteral("abc", literals, "abc")

    with pytest.raises(ValueError):
        checkLiteral("c", literals, "c")
