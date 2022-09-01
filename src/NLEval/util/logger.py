"""Logger utils."""
import logging
import logging.config
from contextlib import contextmanager

from NLEval._config.logger_config import LOGGER_CONFIG
from NLEval.typing import LogLevel, Optional, Union


def display_pbar(level: Union[int, str], threshold="INFO") -> bool:
    """Determines whether to display progress bar."""
    level_int = level if isinstance(level, int) else logging.getLevelName(level)
    threshold_int = logging.getLevelName(threshold)
    return level_int <= threshold_int


def config_logger():
    """Configure logger used by NLEval."""
    logging.config.dictConfig(LOGGER_CONFIG)


def get_logger(
    name: Optional[str],
    base_logger: str = "NLEval",
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
) -> logging.FileHandler:
    """Attach a file handler to a logger.

    Use the format of the first handler to formate the file handler.

    Args:
        logger: The logger to which the file handler is attached.
        log_path: Path of the logged file.

    """
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logger.handlers[0].formatter)
    logger.addHandler(file_handler)
    return file_handler


@contextmanager
def log_file_context(logger: logging.Logger, log_path: str):
    """File log context."""
    file_handler = attach_file_handler(logger, log_path)
    try:
        yield
    finally:
        logger.removeHandler(file_handler)
