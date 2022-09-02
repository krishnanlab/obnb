import os.path as osp
import pathlib

import pytest


class CommonVar:
    home_dir = pathlib.Path(__file__).resolve().parent
    sample_data_dir = osp.join(home_dir, "sample_data")


@pytest.fixture
def commonvar():
    return CommonVar()
