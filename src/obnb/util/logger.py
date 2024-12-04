"""Logger utils."""

import logging
import logging.config
import os
from contextlib import contextmanager

from obnb.alltypes import List, LogLevel, Optional, Union
from obnb.config.logger_config import LOGGER_CONFIG


def verbose(level: Union[int, str], threshold="INFO") -> bool:
    """Determines verbosity based on the given level and the threshold."""
    level_int = level if isinstance(level, int) else logging.getLevelName(level)
    threshold_int = logging.getLevelName(threshold)
    return level_int <= threshold_int


def display_pbar(level: Union[int, str], threshold="INFO") -> bool:
    """Determines whether to display progress bar."""
    return verbose(level, threshold)


def log(_round: int = 3, _fill: str = ": ", **kwargs):
    """Simple keyword-value logger."""

    def _format(key: str) -> str:
        value = kwargs[key]
        if isinstance(value, int):
            out = f"{key}{_fill}{value:,}"
        elif isinstance(value, float):
            out = f"{key}{_fill}{value:.{_round}f}"
        else:
            out = f"{key}{_fill}{value}"
        return out

    print(", ".join(map(_format, kwargs)))


def config_logger():
    """Configure logger used by obnb."""
    logging.config.dictConfig(LOGGER_CONFIG)


def get_logger(
    name: Optional[str],
    base_logger: str = "obnb",
    log_level: LogLevel = "WARNING",
    verbose: bool = False,
) -> logging.Logger:
    """Obtain logger.

    Args:
        name: Name of the logger. If set to None, then directly use the
            base logger.
        base_logger: Name of the base logger to inherit from.
        log_level: Logging level.
        verbose: If set to True and the log_level is more restrictive than
            WARNING, then set the log_level to WARNING. Otherwise leave the
            log level unchanged.

    """
    # logger = logging.getLogger(f"{base_logger}.{name}")
    logger_name = base_logger if name is None else f"{base_logger}.{name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.getLevelName(log_level))
    if verbose and logger.getEffectiveLevel() > 20:
        logger.setLevel(logging.INFO)
    return logger


def attach_file_handler(
    logger: logging.Logger,
    log_path: str,
    *,
    formatter: Optional[logging.Formatter] = None,
) -> logging.FileHandler:
    """Attach a file handler to a logger.

    Use the format of the first handler to format the file handler.

    Args:
        logger: The logger to which the file handler is attached.
        log_path: Path of the logged file.
        formatter: Formatter for the file handler to use. Use that of the
            logger's parent if not set.

    """
    # Make sure directory to the log file exists
    os.makedirs(os.path.split(log_path)[0], exist_ok=True)

    # Create file handler and use parent's formatter if formatter not set
    file_handler = logging.FileHandler(log_path)
    if formatter is None:
        formatter = _get_eff_handlers(logger)[0].formatter
    file_handler.setFormatter(formatter)

    # Attach file handler
    logger.addHandler(file_handler)

    return file_handler


def _get_eff_handlers(logger: Optional[logging.Logger]) -> List[logging.Handler]:
    while logger is not None:
        if logger.handlers:
            return logger.handlers
        logger = logger.parent
    else:
        raise ValueError("No handler available.")


@contextmanager
def log_file_context(logger: logging.Logger, log_path: str):
    """File log context."""
    file_handler = attach_file_handler(logger, log_path)
    try:
        yield
    finally:
        logger.removeHandler(file_handler)
