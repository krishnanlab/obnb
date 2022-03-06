"""
A collection of predefined types used for type checking in NLEval.
"""
from collections.abc import Iterable
from typing import Literal

import numpy as np


INT_TYPE = (int, np.int32, np.int64)
FLOAT_TYPE = (float, np.float32, np.float64, np.float128)
NUMERIC_TYPE = INT_TYPE + FLOAT_TYPE
ITERABLE_TYPE = Iterable

LogLevel = Literal[
    "CRITICAL",
    "ERROR",
    "WARNING",
    "INFO",
    "DEBUG",
    "NOTSET",
]
