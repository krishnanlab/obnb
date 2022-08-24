import numpy as np

from NLEval.typing import Any, Dict, LogLevel, Optional
from NLEval.util.checkers import checkNumpyArrayShape
from NLEval.model_trainer.base import BaseTrainer


class SupervisedLearningTrainer(BaseTrainer):
    """Trainer for supervised learning model.

    Example:
        Given a dictionary ``metrics`` of metric functions, and a ``features``
        that contains the features for each data point, we can train a
        logistic regression model as follows.

        >>> from sklearn.linear_model import LogisticRegression
        >>> trainer = SupervisedLearningTrainer(metrics, features)
        >>> results = trainer.train(LogisticRegression(), y, masks)

        See the ``split`` method in ``label.collection.LabelsetCollection`` for
        generating ``y`` and ``masks``.

    """

    def __init__(
        self,
        metrics,
        features,
        train_on="train",
        dual=False,
        log_level: LogLevel = "WARNING",
    ):
        """Initialize SupervisedLearningTrainer.

        Note:
            Only takes features as input. However, one could pass the graph
            object as features to use the rows of the adjaceny matrix as the
            node features.

        """
        super().__init__(
            metrics,
            features=features,
            train_on=train_on,
            dual=dual,
            log_level=log_level,
        )

    def train(
        self,
        model: Any,
        y: np.ndarray,
        masks: Dict[str, np.ndarray],
        split_idx: int = 0,
    ) -> Dict[str, float]:
        """Train a supervised learning model.

        The ``model`` in this case is a  upervised learning model that has a
        ``fit`` method for training the model, and a ``decision_function`` that
        returns the predict confidence scores given some features. See
        ``sklearn.linear_model.LogisticRegression`` for example.

        """
        # Train model using the training set
        train_mask = self.get_mask(masks, self.train_on, split_idx)
        # TODO: log time and other useful stats (maybe use the decorator?)
        model.fit(self.get_x_from_mask(train_mask), y[train_mask])

        # Evaluate the trained model using the specified metrics
        results = {}
        for metric_name, metric_func in self.metrics.items():
            for mask_name in masks:
                mask = self.get_mask(masks, mask_name, split_idx)
                y_pred = model.decision_function(self.get_x_from_mask(mask))
                score = metric_func(y[mask], y_pred)
                results[f"{mask_name}_{metric_name}"] = score

        return results


class MultiSupervisedLearningTrainer(SupervisedLearningTrainer):
    """Supervised learning model trainer with multiple feature sets.

    Used primarily for auto hyperparameter selection.

    """

    def __init__(
        self,
        metrics,
        features,
        train_on="train",
        val_on: str = "val",
        metric_best: Optional[str] = None,
        log_level: LogLevel = "WARNING",
    ):
        """Initialize MultiSupervisedLearningTrainer."""
        super().__init__(
            metrics,
            features=features,
            train_on=train_on,
            log_level=log_level,
        )

        self.val_on = val_on
        if metric_best is None:
            self.metric_best = list(metrics)[0]
        else:
            self.metric_best = metric_best

    def get_x_from_mask(self, mask):
        """Obtain features of specific nodes from a specific feature set.

        In each iteraction, use one single feature set, indicated by
        ``self._curr_fset_name``, which updated within the for loop in the
        ``train`` method below.

        """
        checkNumpyArrayShape("mask", len(self.idmap), mask)
        idx = np.where(mask)[0]
        fset_idx = self.features.fset_idmap[self._curr_fset_name]
        return self.features.get_features_from_idx(idx, fset_idx)

    def train(self, model, y, masks, split_idx=0):
        """Train a supervised learning mode and select based on validation."""
        best_results = None
        best_val_score = 0
        val_mask_name = f"{self.val_on}_{self.metric_best}"

        val_scores = []
        for fset_name in self.features.fset_idmap.lst:
            self._curr_fset_name = fset_name
            results = super().train(model, y, masks, split_idx)
            val_score = results[val_mask_name]
            val_scores.append(val_score)

            if val_score > best_val_score:
                best_results = results
                best_val_score = val_score
                best_fset_name = self._curr_fset_name

        score_str = ", ".join([f"{i:.3f}" for i in val_scores])
        self.logger.info(
            f"Best val score: {best_val_score:.3f} (via {best_fset_name}) "
            f"Other val scores: [{score_str}]",
        )

        return best_results
