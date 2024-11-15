"""Genearting data splits from the labelset collection."""
from obnb.label.split.holdout import (
    AllHoldout,
    RandomRatioHoldout,
    RatioHoldout,
    ThresholdHoldout,
)
from obnb.label.split.partition import (
    RandomRatioPartition,
    RatioPartition,
    ThresholdPartition,
)
from obnb.label.split.explicit import (
    ByTermSplit
)

__all__ = classes = [
    "AllHoldout",
    "RandomRatioHoldout",
    "RatioHoldout",
    "ThresholdHoldout",
    "RandomRatioPartition",
    "RatioPartition",
    "ThresholdPartition",
    "ByTermSplit",
]
