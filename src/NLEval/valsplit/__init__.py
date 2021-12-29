import warnings

from NLEval.valsplit import Holdout
from NLEval.valsplit import Interface

warnings.simplefilter("always", DeprecationWarning)
warnings.warn(
    "The module valsplit is deprecated",
    category=DeprecationWarning,
    stacklevel=2,
)
warnings.simplefilter("default", DeprecationWarning)

__all__ = ["Holdout", "Interface"]
