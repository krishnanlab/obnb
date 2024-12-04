"""Collection of network learning evaluation utilities."""

from obnb import graph, label, model_trainer
from obnb.dataset import Dataset, OpenBiomedNetBench
from obnb.util.checkers import checkVersion
from obnb.util.logger import config_logger, get_logger

__version__ = "0.1.1-dev"
__data_version__ = "obnbdata-0.1.0"
__all__ = [
    "Dataset",
    "OpenBiomedNetBench",
    "REGISTRIES",
    "graph",
    "label",
    "model_trainer",
]

checkVersion(__version__)

# Configure logger setting and create global logger
config_logger()
logger = get_logger(None, "obnb", log_level="INFO")


# Register modules to REGISTRIES
from obnb.registry import REGISTRIES  # noqa: E402
from obnb.transform import node_feature  # noqa: E402, F401
