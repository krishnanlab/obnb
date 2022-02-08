class IDNotExistError(Exception):
    """Raised when query ID not exist."""


class IDExistsError(Exception):
    """Raised when try to add new ID that already exists."""


class NotConvergedWarning(RuntimeWarning):
    """Warn when model not converged."""


class OboTermIncompleteError(Exception):
    """Raised when the obo term do not have an ID or a name."""
