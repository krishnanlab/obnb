class DataNotFoundError(Exception):
    """Raised when an particular version of arvhival data is unavailable."""


class ExceededMaxNumRetries(Exception):
    """Raised when the number of download retries exceeds the limit."""


class IDNotExistError(Exception):
    """Raised when query ID not exist."""


class IDExistsError(Exception):
    """Raised when try to add new ID that already exists."""


class EdgeNotExistError(Exception):
    """Raised when the edge being accessed does not exist."""


class NotConvergedWarning(RuntimeWarning):
    """Warn when model not converged."""


class OboTermIncompleteError(Exception):
    """Raised when the obo term do not have an ID or a name."""
