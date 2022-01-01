class IDNotExistError(Exception):
    """Raised when query ID not exist."""


class IDExistsError(Exception):
    """Raised when try to add new ID that already exists."""


class NotConvergedWarning(RuntimeWarning):
    """Warn when model not converged."""
