"""Collection of types used in NLEval."""
from collections.abc import Iterable
from typing import Any
from typing import Callable
from typing import Iterator
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple

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

Metric = Callable[[np.ndarray, np.ndarray], float]
GraphGymMetric = Callable[[List[Tensor], List[Tensor], str], float]
