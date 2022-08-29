"""Collection of network learning evaluation utilities."""
import logging

from NLEval import graph, label, model_trainer
from NLEval.util.logger import config_logger

__version__ = "0.1.0-dev.0"
__all__ = ["graph", "label", "model_trainer"]

# Configure logger setting and create global logger
config_logger()
logger = logging.getLogger("NLEval_brief")
logger.setLevel(logging.INFO)
