import signal
import time

from .checkers import checkType


class Timeit:
    """Timing function call."""

    def __init__(self, verbose: bool = True) -> None:
        """Initialize timer.

        Args:
            verbose: whether or not to print timing info
        """
        self.verbose = verbose

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
        checkType("verbose", bool, val)
        self._verbose = val


class Timeout:
    """Timeout decorator.

    https://stackoverflow.com/a/22348885/12519564

    """

    def __init__(self, seconds: int = 10, error_message: str = "Timeout"):
        """Initialize timeout decorator.

        Args:
            seconds (int): Maximum allowed time in seconds.
            error_message (str): Error message.

        """
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        """Raising timeout error."""
        raise TimeoutError(f"({self.seconds} secs) {self.error_message}")

    def __enter__(self):
        """Entering context."""
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        """Leaving context."""
        signal.alarm(0)
