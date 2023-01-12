import warnings
from functools import wraps


class Deprecated:
    """Deprecation decorator."""

    def __init__(
        self,
        msg: str,
        category=DeprecationWarning,
    ) -> None:
        """Initialize Deprecated object.

        Args:
            msg: Deprecation warning message.
            category: Warning category.

        """
        self.msg = msg
        self.category = category

    def __call__(self, func):
        """Wraps the function with deprecation warning."""

        @wraps(func)
        def wrapped(*args, **kwargs):
            warnings.simplefilter("always", self.category)
            warnings.warn(self.msg, category=self.category, stacklevel=2)
            warnings.simplefilter("default", self.category)
            return func(*args, **kwargs)

        return wrapped
