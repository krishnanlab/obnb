"""
Type checking functions.

This module contains a collection of checkers to ensure that the input value
to a function call is valid.
"""
import numpy as np

from ..typing import INT_TYPE
from ..typing import ITERABLE_TYPE
from ..typing import NUMERIC_TYPE

__all__ = [
    "checkValuePositive",
    "checkValueNonnegative",
    "checkType",
    "checkNullableType",
    "checkTypesInIterable",
    "checkTypesInList",
    "checkTypesInSet",
    "checkTypesInNumpyArray",
    "checkTypesInIterableErrEmpty",
    "checkNumpyArrayIsNumeric",
    "checkNumpyArrayNDim",
    "checkNumpyArrayShape",
]


def checkValuePositive(name, val):
    """Check if the input value is positive."""
    if not val > 0:
        raise ValueError(f"{name!r} should be positive, got {val}")


def checkValueNonnegative(name, val):
    """Check if the input value is non-negative."""
    if not val >= 0:
        raise ValueError(f"{name!r} should be non-negative, got {val}")


def checkNullableType(name, targetType, val):
    """Check the type of an input value and allow None.

    Args:
        name(str): name of the value
        targetType(type): desired type of the value
        val(any): value to be checked

    Raises:
        TypeError: if `val` is not an instance of `targetType`

    """
    if not isinstance(val, targetType) and val is not None:
        raise TypeError(
            f"{name!r} should be {targetType!r}, not {type(val)!r}: {val!r}",
        )


def checkType(name, targetType, val):
    """Check the type of an input and raise ValueError if it is None."""
    if val is None:
        raise ValueError(f"Value for {name!r} has not yet been provided")
    else:
        checkNullableType(name, targetType, val)


def checkTypesInIterable(name, targetType, val):
    """Check the types of all elements in an iterable"""
    for idx, i in enumerate(val):
        if not isinstance(i, targetType):
            raise TypeError(
                f"All instances in {name!r} must be type {targetType!r}, "
                f"invalid type {type(i)!r} found at position ({idx}): {i!r}",
            )


def checkTypesInList(name, targetType, val):
    """Check types of all elements in a list."""
    checkType(name, list, val)
    checkTypesInIterable(name, targetType, val)


def checkTypesInSet(name, targetType, val):
    """Check types of all elements in a set."""
    checkType(name, set, val)
    checkTypesInIterable(name, targetType, val)


def checkTypesInNumpyArray(name, targetType, val):
    """Check types of all elements in a numpy array."""
    checkType(name, np.ndarray, val)
    checkTypesInIterable(name, targetType, val)


def checkTypesInIterableErrEmpty(name, targetType, val):
    """Check the types of all elements in an iterable and error if empty."""
    if len(val) == 0:
        raise ValueError(f"No {name} specified")
    checkTypesInIterable(name, targetType, val)


def checkNumpyArrayIsNumeric(name, ary):
    """Check if numpy array is numeric type."""
    checkType(name, np.ndarray, ary)
    if not any([ary.dtype == i for i in NUMERIC_TYPE]):
        raise TypeError(f"{name!r} should be numeric, not type {ary.dtype!r}")


def checkNumpyArrayNDim(name, targetNDim, ary):
    """Check the rank (number of dimension) of a numpy array.

    Args:
        name(str): name of the input array
        targetNDim(int): desired number of dimensions
        ary(:obj:`numpy.ndarray`): numpy array to be checked

    Raises:
        ValueError: if the number of dimensions of the input array is different
            from the target number of dimensions

    """
    checkType("targetNDim", INT_TYPE, targetNDim)
    checkType(name, np.ndarray, ary)
    NDim = len(ary.shape)
    if NDim != targetNDim:
        raise ValueError(
            f"{name!r} should be {targetNDim} dimensional array, "
            f"not {NDim} dimensional",
        )


def checkNumpyArrayShape(name, targetShape, ary):
    """Check the shape of a numpy array.

    Args:
        name(str): name of the input array
        targetShape: desired shape of the array
        ary(:obj:`numpy.ndarray`): numpy array to be checked

    Raises:
        ValueError: if the sape of the input array differ from the target shape

    """
    if isinstance(targetShape, ITERABLE_TYPE):
        checkTypesInIterable("targetShape", INT_TYPE, targetShape)
        NDim = len(targetShape)
    else:
        checkType("targetShape", INT_TYPE, targetShape)
        NDim = 1
        targetShape = (targetShape,)
    checkNumpyArrayNDim(name, NDim, ary)
    shape = ary.shape
    if shape != targetShape:
        raise ValueError(
            f"{name!r} should be in shape {targetShape!r}, not {shape!r}",
        )
