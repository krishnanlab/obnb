from typing import Any
from typing import Dict

import numpy as np

from .base import BaseTrainer


class LabelPropagationTrainer(BaseTrainer):
    """Label propagation trainer."""

    def __init__(self, metrics, graph):
        """Initialize LabelPropagationTrainer.

        Note:
            Only takes graph as input.

        """
        super().__init__(metrics, graph=graph)

    def train(
        self,
        model: Any,
        y: np.ndarray,
        masks: Dict[str, np.ndarray],
        split_idx: int = 0,
        train_on: str = "train",
    ) -> Dict[str, float]:
        """Propagate labels.

        Note:
            No need to specify ``model`` in this case. The label propagation
            scheme simply propagate the seed nodes across the network.

        """
        # Train model using the training set
        train_mask = self.get_mask(masks, train_on, split_idx)
        seed = y * train_mask
        y_pred = model(self.graph, seed)

        # Evaluate the prediction using the specified metrics
        results = {}
        for metric_name, metric_func in self.metrics.items():
            for mask_name in masks:
                mask = self.get_mask(masks, mask_name, split_idx)
                score = metric_func(y[mask], y_pred[mask])
                results[f"{mask_name}_{metric_name}"] = score

        return results