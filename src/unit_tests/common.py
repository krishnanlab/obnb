import os
from sys import path

pwd = os.path.dirname(os.path.abspath(__file__))
lst = pwd.split("/")
idx = lst.index("src") + 1
src_pth = "/".join(lst[:idx])
path.append(src_pth)
import unittest
import numpy as np
from NLEval.util.Exceptions import IDNotExistError, IDExistsError

SAMPLE_DATA_PATH = src_pth + "/unit_tests/sample_data/"
