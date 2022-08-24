"""Genearting data splits from the labelset collection."""
from NLEval.label.split.holdout import (
    AllHoldout,
    RandomRatioHoldout,
    RatioHoldout,
    ThresholdHoldout,
)
from NLEval.label.split.partition import (
    RandomRatioPartition,
    RatioPartition,
    ThresholdPartition,
)

__all__ = [
    "AllHoldout",
    "RandomRatioHoldout",
    "RatioHoldout",
    "ThresholdHoldout",
    "RandomRatioPartition",
    "RatioPartition",
    "ThresholdPartition",
]
