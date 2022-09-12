from functools import partial

import numpy as np

from nleval.typing import Any, Callable, Dict, LogLevel, Optional
from nleval.util.logger import get_logger


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
            base_logger="nleval_brief",
        )

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


class StandardTrainer(BaseTrainer):
    def _setup(self, dataset, split_idx):
        # Initialize y dictionary: mask_name -> y_pred/true (2d arrays)
        # Set up results compute function using the y dicts and the metrics
        y_pred_dict: Dict[str, np.ndarray] = {}
        y_true_dict: Dict[str, np.ndarray] = {}
        for mask_name in dataset.masks:
            num_examples = dataset.masks[mask_name][:, split_idx].sum()
            num_classes = 1 if len(dataset.y.shape) == 1 else dataset.y.shape[1]
            shape = (num_examples, num_classes)
            y_pred_dict[mask_name] = np.zeros(shape)
            y_true_dict[mask_name] = np.zeros(shape)

        compute_results = partial(
            self._compute_results,
            y_true_dict,
            y_pred_dict,
            metrics=self.metrics,
        )

        return y_true_dict, y_pred_dict, compute_results

    @staticmethod
    def _compute_results(
        y_true_dict,
        y_pred_dict,
        masks,
        metrics,
        label_idx: Optional[str] = None,
    ):
        results = {}
        for metric_name, metric_func in metrics.items():
            for mask_name in masks:
                y_true = y_true_dict[mask_name]
                y_pred = y_pred_dict[mask_name]
                if label_idx is not None:
                    y_true, y_pred = y_true[:, label_idx], y_pred[:, label_idx]

                score = metric_func(y_true, y_pred)
                results[f"{mask_name}_{metric_name}"] = score

        return results
