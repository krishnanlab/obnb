"""Type checking functions.

This module contains a collection of checkers to ensure that the input value to
a function call is valid.

"""
import os
import warnings
from typing import get_args

import numpy as np
import requests

from obnb.typing import INT_TYPE, NUMERIC_TYPE, Iterable, List, Optional, Tuple

__all__ = [
    "checkConfig",
    "checkNullableType",
    "checkNumpyArrayIsNumeric",
    "checkNumpyArrayNDim",
    "checkNumpyArrayShape",
    "checkType",
    "checkTypesInIterable",
    "checkTypesInIterableErrEmpty",
    "checkTypesInList",
    "checkTypesInNumpyArray",
    "checkTypesInSet",
    "checkValueNonnegative",
    "checkValuePositive",
]


def checkVersion(version: str):
    """Check if the current version is up to date."""
    if os.environ.get("OBNB_IGNORE_OUTDATE") == "1":
        return

    try:
        r = requests.get("https://pypi.org/pypi/obnb/json")

        versions = list(r.json()["releases"])
        stable_versions = [v for v in versions if "dev" not in v]
        if not stable_versions:
            # XXX: remove once the first stable version is released
            return

        latest_version = stable_versions[-1]
        if latest_version != version:
            warnings.warn(
                f"A new OBNB version {latest_version!r} is available "
                f"(current version: {version!r}).",
                UserWarning,
                stacklevel=2,
            )
    except requests.exceptions.ConnectionError:
        warnings.warn(
            "Failed to check the latest obnb version due to connection errorx",
            RuntimeWarning,
            stacklevel=2,
        )


def checkValuePositive(name, val):
    """Check if the input value is positive."""
    if not val > 0:
        raise ValueError(f"{name!r} should be positive, got {val}")


def checkValueNonnegative(name, val):
    """Check if the input value is non-negative."""
    if not val >= 0:
        raise ValueError(f"{name!r} should be non-negative, got {val}")


def checkLiteral(name, targetLiteral, val):
    """Check if the input value matches one of the literals."""
    if val not in (target_values := get_args(targetLiteral)):
        raise ValueError(
            f"Unknown value {val} for {name!r}, accepted literals are {target_values}",
        )


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
    """Check the types of all elements in an iterable."""
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
    if isinstance(targetShape, Iterable):
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


def checkConfig(
    name: str,
    config,
    /,
    *,
    allowed_types: Tuple[type, ...] = (int, float, str, bool, type(None)),
    max_depth: int = 2,
    white_list: Optional[List[str]] = None,
):
    """Check a configuration dictionary.

    Args:
        name: Name of the input config dict.
        allowed_types: A tuple containing all allowed types.
        max_depth: Maximum number of levels of dictionary allowed. When it is
            set to one, then nested dictionary, i.e., dictionary as parameter,
            is disallowed.
        white_list: List of keys to bypass the checks.

    """
    checkType("name", str, name)
    checkType("config", dict, config)
    checkType("allowed_types", tuple, allowed_types)
    checkType("max_depth", INT_TYPE, max_depth)
    checkValuePositive("max_depth", max_depth)
    checkNullableType("white_list", list, white_list)
    white_list = white_list or []
    checkTypesInIterable("white_list", str, white_list)

    def _check_val_in_config(config_dict, depth):
        if depth > max_depth:
            raise ValueError(f"Max depth ({max_depth}) exceeded")

        for key, val in config_dict.items():
            if key in white_list:
                continue
            elif isinstance(val, dict):
                _check_val_in_config(val, depth + 1)
            elif isinstance(val, Iterable):
                checkTypesInIterable(key, allowed_types, val)
            elif isinstance(val, allowed_types):
                continue
            else:
                raise TypeError(
                    f"The {key!r} parameter type ({type(val)})in the config dict "
                    f"{name!r} is not allowed. Allowed types are: {allowed_types!r}",
                )

    _check_val_in_config(config, 1)
