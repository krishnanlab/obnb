import warnings

from NLEval.model import LabelPropagation
from NLEval.model import SupervisedLearning

warnings.simplefilter("always", DeprecationWarning)
warnings.warn(
    "The module model is deprecated",
    category=DeprecationWarning,
    stacklevel=2,
)

__all__ = ["LabelPropagation", "SupervisedLearning"]
