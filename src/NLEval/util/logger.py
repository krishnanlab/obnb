"""Logger utils."""
import logging.config
import os.path as osp
import pathlib

import yaml

from ..typing import LogLevel


def config_logger():
    """Configure logger used by NLEval."""
    homedir = pathlib.Path(__file__).parent.parent.absolute()
    with open(osp.join(homedir, "_config", "logging.yaml"), "r") as f:
        logging.config.dictConfig(yaml.safe_load(f.read()))


def get_logger(
    name: str,
    logger_name: str = "NLEval",
    log_level: LogLevel = "WARNING",
    verbose: bool = False,
) -> logging.Logger:
    """Obtain logger."""
    logger = logging.getLogger(f"{logger_name}.{name}")
    logger.setLevel(getattr(logging, log_level))
    if verbose and logger.getEffectiveLevel() > 20:
        logger.setLevel(logging.INFO)
    return logger
