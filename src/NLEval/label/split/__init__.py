"""Genearting data splits from the labelset collection."""
from .holdout import AllHoldout, RandomRatioHoldout, RatioHoldout, ThresholdHoldout
from .partition import RandomRatioPartition, RatioPartition, ThresholdPartition

__all__ = [
    "AllHoldout",
    "RandomRatioHoldout",
    "RatioHoldout",
    "ThresholdHoldout",
    "RandomRatioPartition",
    "RatioPartition",
    "ThresholdPartition",
]
