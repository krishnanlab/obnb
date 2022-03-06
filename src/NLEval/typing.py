"""Collection of types used in NLEval."""
from typing import Any
from typing import Callable
from typing import DefaultDict
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Set
from typing import TextIO
from typing import Tuple
from typing import Union

import numpy as np

try:
    from torch import Tensor
except ModuleNotFoundError:
    import warnings

    warnings.warn("PyTorch not installed, some functionalities may be limited")
    Tensor = Any

INT_TYPE = (int, np.int32, np.int64)
FLOAT_TYPE = (float, np.float32, np.float64, np.float128)
NUMERIC_TYPE = INT_TYPE + FLOAT_TYPE
ITERABLE_TYPE = Iterable

EdgeData = List[Dict[int, float]]

LogLevel = Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]

Metric = Callable[[np.ndarray, np.ndarray], float]
GraphGymMetric = Callable[[List[Tensor], List[Tensor], str], float]
Term = Tuple[str, str, Optional[List[str]], Optional[List[str]]]
Splitter = Callable[[np.ndarray, np.ndarray], Iterator[Tuple[np.ndarray, ...]]]


__all__ = [
    "Any",
    "Callable",
    "DefaultDict",
    "Dict",
    "Iterable",
    "List",
    "Literal",
    "Optional",
    "Sequence",
    "Set",
    "TextIO",
    "Tuple",
    "Union",
    "INT_TYPE",
    "FLOAT_TYPE",
    "NUMERIC_TYPE",
    "ITERABLE_TYPE",
    "LogLevel",
    "Metric",
    "Term",
    "Splitter",
]
