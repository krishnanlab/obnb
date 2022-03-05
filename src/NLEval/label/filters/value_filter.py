from .base import BaseFilter


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

    def __repr__(self):
        """Return name of the ValueFilter and its parameters."""
        val, remove = self.val, self.remove
        return f"{super().__repr__()}({val=}, {remove=})"

    def criterion(self, val):
        return True if (val == self.val) is self.remove else False
