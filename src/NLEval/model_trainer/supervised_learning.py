from typing import Any
from typing import Dict

import numpy as np

from .base import BaseTrainer


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

    def __init__(self, metrics, features, train_on="train"):
        """Initialize SupervisedLearningTrainer.

        Note:
            Only takes features as input. However, one could pass the graph
            object as features to use the rows of the adjaceny matrix as the
            node features.

        """
        super().__init__(metrics, features=features, train_on=train_on)

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
        # TODO: log time and other useful stats
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
