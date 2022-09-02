"""Collection of types used in NLEval."""
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    TextIO,
    Tuple,
    Union,
)

import numpy as np

try:
    from torch import Tensor
except ModuleNotFoundError:
    Tensor = Any

try:
    from torch_geometric.data import Data as PyG_Data
except ModuleNotFoundError:
    PyG_Data = Any

INT_TYPE = (int, np.int32, np.int64)
FLOAT_TYPE = (float, np.float32, np.float64, np.float128)
NUMERIC_TYPE = INT_TYPE + FLOAT_TYPE

EdgeData = List[Dict[int, float]]
EdgeDir = Literal["out", "in", "both"]

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
    "EdgeData",
    "EdgeDir",
    "FLOAT_TYPE",
    "INT_TYPE",
    "Iterable",
    "List",
    "Literal",
    "LogLevel",
    "Mapping",
    "Metric",
    "NUMERIC_TYPE",
    "Optional",
    "PyG_Data",
    "Sequence",
    "Set",
    "Splitter",
    "Term",
    "TextIO",
    "Tuple",
    "Union",
]
