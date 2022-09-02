import numpy as np

from NLEval.feature.base import BaseFeature
from NLEval.typing import Iterable, LogLevel, Optional, Union


class FeatureVec(BaseFeature):
    def __init__(
        self,
        dim: Optional[int] = None,
        log_level: LogLevel = "INFO",
        verbose: bool = False,
    ):
        """Initialize FeatureVec."""
        super().__init__(dim, log_level, verbose)

    def get_featvec_from_idxs(
        self,
        idxs: Optional[Union[int, Iterable[int]]],
    ) -> np.ndarray:
        pass

    def get_featvec(
        self,
        ids: Optional[Union[str, Iterable[str]]],
    ) -> np.ndarray:
        pass
