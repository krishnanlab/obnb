from NLEval.label.filters.base import BaseFilter
from NLEval.typing import List


class BaseValueFilter(BaseFilter):
    """Filter based on certain values.

    Attributes:
        val: target value
        remove(bool): if true, remove any ID with matched value,
            else remove any ID with mismatched value

    """

    def __init__(self, val, remove=True, **kwargs):
        """Initialize BaseValueFilter object."""
        super().__init__(**kwargs)
        self.val = val
        self.remove = remove

    @property
    def params(self) -> List[str]:
        """Parameter list."""
        return ["val", "remove"]

    def criterion(self, val):
        return True if (val == self.val) is self.remove else False
