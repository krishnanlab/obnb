"""Collection of network learning evaluation utilities."""
import logging

from . import graph, label, model_trainer
from .util.logger import config_logger

__all__ = ["graph", "label", "model_trainer"]

# Configure logger setting and create global logger
config_logger()
logger = logging.getLogger("NLEval_brief")
logger.setLevel(logging.INFO)
