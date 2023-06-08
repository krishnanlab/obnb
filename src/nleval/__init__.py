"""Collection of network learning evaluation utilities."""
from obnb import graph, label, model_trainer
from obnb.dataset import Dataset
from obnb.util.logger import config_logger, get_logger

__version__ = "0.1.0-dev8"
__data_version__ = "nledata-v0.1.0-dev6"
__all__ = ["Dataset", "graph", "label", "model_trainer"]

# Configure logger setting and create global logger
config_logger()
logger = get_logger(None, "obnb", log_level="INFO")
