"""Collection of network learning evaluation utilities."""
from nleval import graph, label, model_trainer
from nleval.dataset import Dataset
from nleval.util.logger import config_logger, get_logger

__version__ = "0.1.0-dev4"
__data_version__ = "nledata-v0.1.0-dev1"
__all__ = ["Dataset", "graph", "label", "model_trainer"]

# Configure logger setting and create global logger
config_logger()
logger = get_logger(None, "nleval", log_level="INFO")
