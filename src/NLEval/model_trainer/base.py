import numpy as np

from NLEval.typing import Any, Callable, Dict, LogLevel
from NLEval.util.logger import get_logger


class BaseTrainer:
    """The BaseTrainer object.

    Abstract class for trainer objects, which serve as interfaces or shortcuts
    for training specific types of models.

    """

    def __init__(
        self,
        metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
        train_on: str = "train",
        log_level: LogLevel = "INFO",
    ):
        """Initialize BaseTraining.

        Note: "dual" mode only works if the input features is MultiFeatureVec.

        Args:
            metrics: Dictionary of metrics used to train/evaluate the model.
            graph: Optional graph object.
            features: Optional node feature vectors.
            train_on: Which mask to use for training.
            dual (bool): If set to true, predict the label of individual
                feature, i.e.  individual columns (default: :obj:`False`)

        """
        self.metrics = metrics
        self.train_on = train_on
        self.logger = get_logger(
            self.__class__.__name__,
            log_level=log_level,
            base_logger="NLEval_brief",
        )

    @staticmethod
    def get_mask(
        masks: Dict[str, np.ndarray],
        mask_name: str,
        split_idx: int,
    ) -> np.ndarray:
        """Return a specific mask."""
        return masks[mask_name][:, split_idx]

    def train(
        self,
        model: Any,
        dataset,
        split_idx: int = 0,
    ):
        """Train model and return metrics.

        Args:
            model: Model to be trained.
            y: Label array with the shape of (n_tot_samples, n_classes) or
                (n_tot_samples,) if n_classes = 1.
            masks: Masks for splitting data, see the ``split`` method in
                ``label.collection.LabelsetCollection`` for moer info.
            split_idx: Which split to use for training and evaluation.

        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not have functional ``train`` "
            f"method, use a derived class instead.",
        )
