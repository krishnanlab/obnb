"""Genearting data splits from the labelset collection."""
from nleval.label.split.holdout import (
    AllHoldout,
    RandomRatioHoldout,
    RatioHoldout,
    ThresholdHoldout,
)
from nleval.label.split.partition import (
    RandomRatioPartition,
    RatioPartition,
    ThresholdPartition,
)

__all__ = classes = [
    "AllHoldout",
    "RandomRatioHoldout",
    "RatioHoldout",
    "ThresholdHoldout",
    "RandomRatioPartition",
    "RatioPartition",
    "ThresholdPartition",
]
