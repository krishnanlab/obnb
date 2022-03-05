"""Collection of network learning evaluation utilities."""
import logging.config
import os.path as osp
import pathlib

import yaml

from . import graph
from . import label
from . import model_trainer

__all__ = ["graph", "label", "model_trainer"]

HOMEDIR = pathlib.Path(__file__).parent.parent.parent.absolute()
with open(osp.join(HOMEDIR, "logging.yaml"), "r") as f:
    logging.config.dictConfig(yaml.safe_load(f.read()))
