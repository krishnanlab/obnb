import time
from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

import obnb.metric
from obnb.typing import Any, Callable, Dict, LogLevel, Optional
from obnb.util.logger import attach_file_handler, get_logger


class BaseTrainer:
    """The BaseTrainer object.

    Abstract class for trainer objects, which serve as interfaces or shortcuts
    for training specific types of models.

    """

    def __init__(
        self,
        metrics: Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
        train_on: str = "train",
        log_level: LogLevel = "INFO",
        log_path: Optional[str] = None,
    ):
        """Initialize BaseTraining.

        Note: "dual" mode only works if the input features is MultiFeatureVec.

        Args:
            metrics: Dictionary of metrics used to train/evaluate the model. If
                not specified, will use the default selection of APOP and AUROC.
            train_on: Which mask to use for training.
            log_level: Log level.
            log_path: Log file path. If not set, then do not log to file.

        """
        self._tic: Optional[float] = None

        if not metrics:
            metrics = {
                "apop": obnb.metric.log2_auprc_prior,
                "auroc": obnb.metric.auroc,
            }
        self.metrics = metrics

        self.train_on = train_on
        self.logger = get_logger(
            self.__class__.__name__,
            log_level=log_level,
            base_logger="obnb_brief",
        )

        if log_path:
            attach_file_handler(self.logger, log_path)

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

    def _elapse(self) -> float:
        """Record the time difference between two consecutive calls.

        Note:
            The first call will return elapsed time of 0.

        """
        now = time.time()
        elapsed = 0.0 if self._tic is None else now - self._tic
        self._tic = now
        return elapsed


class StandardTrainer(BaseTrainer):
    def train(
        self,
        model: Any,
        dataset,
        split_idx: int = 0,
    ) -> Dict[str, float]:
        """Train a supervised learning model.

        The ``model`` in this case is a  upervised learning model that has a
        ``fit`` method for training the model, and a ``decision_function`` that
        returns the predict confidence scores given some features. See
        ``sklearn.linear_model.LogisticRegression`` for example.

        """
        g = dataset.graph
        x = None if dataset.feature is None else dataset.feature.mat
        y = dataset.y

        # TODO: log time and other useful stats (maybe use the decorator?)
        train_mask = dataset.masks[self.train_on][:, split_idx]
        self._model_train(model, g, x, y, train_mask)

        _, _, get_predictions, compute_results = self._setup(dataset, split_idx)
        get_predictions(model, x, y, dataset.masks)
        results = compute_results(dataset.masks)

        return results

    def fit_and_eval(
        self,
        model: Any,
        dataset,
        split_idx: int = 0,
        consider_negative: bool = False,
        reduce: str = "none",
        progress: bool = True,
    ) -> Dict[str, float]:
        """Fit model and evaluate.

        Note:
            The original model is not trained. For each task, a deep copy of
            the model is created and it is evaluated via one-vs-rest.

        """
        g = dataset.graph
        x = None if dataset.feature is None else dataset.feature.mat

        _, _, get_predictions, compute_results = self._setup(dataset, split_idx)
        pbar = tqdm(dataset.label.label_ids, disable=not progress)
        for i, label_id in enumerate(pbar):
            y, masks = dataset.label.split(
                splitter=dataset.splitter,
                target_ids=tuple(dataset.idmap.lst),
                labelset_name=label_id,
                consider_negative=consider_negative,
            )

            train_mask = masks[self.train_on][:, split_idx]
            model_copy = deepcopy(model)
            self._model_train(model_copy, g, x, y, train_mask)

            get_predictions(model_copy, x, y, masks, i)
            intermediate_results = compute_results(masks, label_idx=i)
            self.logger.info(f"{label_id}\t{intermediate_results}")

        results = compute_results(dataset.masks, reduce=reduce)

        return results

    def _setup(self, dataset, split_idx: int):
        # Initialize y dictionary: mask_name -> y_pred/true (2d arrays)
        y_pred_dict: Dict[str, np.ndarray] = {}
        y_true_dict: Dict[str, np.ndarray] = {}
        num_classes = 1 if len(dataset.y.shape) == 1 else dataset.y.shape[1]
        for mask_name in dataset.masks:
            num_examples = dataset.masks[mask_name][:, split_idx].sum()
            shape = (num_examples, num_classes)
            y_pred_dict[mask_name] = np.zeros(shape)
            y_true_dict[mask_name] = np.zeros(shape)

        def compute_results(
            masks,
            label_idx: Optional[int] = None,
            reduce: str = "mean",
        ) -> Dict[str, float]:
            # Set up results compute function using the y dicts and the metrics
            results = {}
            for metric_name, metric_func in self.metrics.items():
                for mask_name in masks:
                    y_true = y_true_dict[mask_name]
                    y_pred = y_pred_dict[mask_name]
                    if label_idx is not None:
                        y_true = y_true[:, label_idx]
                        y_pred = y_pred[:, label_idx]

                    score = metric_func(y_true, y_pred, reduce=reduce)  # type: ignore
                    results[f"{mask_name}_{metric_name}"] = score

            return results

        def get_predictions(model, x, y, masks, label_idx: Optional[int] = None):
            # Function to fill in y_pred_dict and y_true_dict given trained model
            for mask_name in masks:
                mask = masks[mask_name][:, split_idx]
                y_true = y[mask]
                y_pred = self._model_predict(model, x, mask)

                if label_idx is None:
                    y_true_dict[mask_name] = y_true
                    y_pred_dict[mask_name] = y_pred
                else:  # only fill in the column that corresponds to the task
                    y_true_dict[mask_name][:, label_idx] = y_true
                    y_pred_dict[mask_name][:, label_idx] = y_pred

        return y_true_dict, y_pred_dict, get_predictions, compute_results

    @staticmethod
    def _model_predict(model, x, mask):
        raise NotImplementedError

    @staticmethod
    def _model_train(model, g, x, y, mask):
        raise NotImplementedError
