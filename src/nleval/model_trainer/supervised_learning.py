from nleval.model_trainer.base import StandardTrainer
from nleval.typing import LogLevel, Optional


class SupervisedLearningTrainer(StandardTrainer):
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
        train_on="train",
        log_level: LogLevel = "WARNING",
        log_path: Optional[str] = None,
    ):
        """Initialize SupervisedLearningTrainer.

        Note:
            Only takes features as input. However, one could pass the graph
            object as features to use the rows of the adjaceny matrix as the
            node features.

        """
        super().__init__(
            metrics,
            train_on=train_on,
            log_level=log_level,
            log_path=log_path,
        )

    @staticmethod
    def _model_predict(model, x, mask):
        return model.decision_function(x[mask])

    @staticmethod
    def _model_train(model, g, x, y, mask):
        model.fit(x[mask], y[mask])


class MultiSupervisedLearningTrainer(SupervisedLearningTrainer):
    """Supervised learning model trainer with multiple feature sets.

    Used primarily for auto hyperparameter selection.

    """

    def __init__(
        self,
        metrics,
        train_on="train",
        val_on: str = "val",
        metric_best: Optional[str] = None,
        log_level: LogLevel = "WARNING",
        log_path: Optional[str] = None,
    ):
        """Initialize MultiSupervisedLearningTrainer."""
        super().__init__(
            metrics,
            train_on=train_on,
            log_level=log_level,
            log_path=log_path,
        )

        self.val_on = val_on
        if metric_best is None:
            self.metric_best = list(metrics)[0]
        else:
            self.metric_best = metric_best

    def train(self, model, dataset, y, masks, split_idx=0):
        """Train a supervised learning mode and select based on validation."""
        best_results = None
        best_val_score = 0
        val_mask_name = f"{self.val_on}_{self.metric_best}"

        val_scores = []
        for fset_name in dataset.feature.fset_idmap.lst:
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
