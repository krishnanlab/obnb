import time
import typing

from NLEval.util import checkers


class TimeIt:
    """Timing function call."""

    def __init__(self, verbose: bool = True) -> None:
        """Initialize timer.

        Args:
            verbose: whether or not to print timing info
        """
        self.verbose = verbose

    @typing.no_type_check
    def __call__(self, func):
        """Return function wrapped with timer."""

        def wrapper(*args):
            start = time.perf_counter()
            func(*args)
            elapsed = time.perf_counter() - start
            print(f"Took {elapsed:.2f} seconds to run function {repr(func)}")

        return wrapper if self.verbose else func

    @property
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, val: bool) -> None:
        checkers.checkType("verbose", bool, val)
        self._verbose = val
