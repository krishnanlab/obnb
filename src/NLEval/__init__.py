"""Collection of network learning evaluation utilities."""
from NLEval import graph, label, model_trainer
from NLEval.dataset import Dataset
from NLEval.util.logger import config_logger, get_logger

__version__ = "0.1.0-dev1"
__data_version__ = "nledata-v0.1.0-dev"
__all__ = ["Dataset", "graph", "label", "model_trainer"]

# Configure logger setting and create global logger
config_logger()
logger = get_logger(None, "NLEval", log_level="INFO")
