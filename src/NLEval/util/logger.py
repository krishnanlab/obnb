"""Logger utils."""
import logging.config
import os.path as osp
import pathlib

import yaml


def config_logger():
    """Configure logger used by NLEval."""
    homedir = pathlib.Path(__file__).parent.parent.absolute()
    with open(osp.join(homedir, "logging.yaml"), "r") as f:
        logging.config.dictConfig(yaml.safe_load(f.read()))
