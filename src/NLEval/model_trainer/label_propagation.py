import numpy as np

from NLEval.model_trainer.base import BaseTrainer
from NLEval.typing import Any, Dict


class LabelPropagationTrainer(BaseTrainer):
    """Label propagation trainer."""

    def __init__(self, metrics, train_on="train"):
        """Initialize LabelPropagationTrainer."""
        super().__init__(metrics, train_on=train_on)

    def train(
        self,
        model: Any,
        dataset,
        y: np.ndarray,
        masks: Dict[str, np.ndarray],
        split_idx: int = 0,
    ) -> Dict[str, float]:
        """Propagate labels.

        Note:
            No need to specify ``model`` in this case. The label propagation
            scheme simply propagate the seed nodes across the network.

        """
        # Train model using the training set
        train_mask = self.get_mask(masks, self.train_on, split_idx)
        y_pred = model(dataset.graph, y * train_mask)

        # Evaluate the prediction using the specified metrics
        results = {}
        for metric_name, metric_func in self.metrics.items():
            for mask_name in masks:
                mask = self.get_mask(masks, mask_name, split_idx)
                score = metric_func(y[mask], y_pred[mask])
                results[f"{mask_name}_{metric_name}"] = score

        return results
