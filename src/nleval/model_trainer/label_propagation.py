from nleval.model_trainer.base import BaseTrainer
from nleval.typing import Any, Dict


class LabelPropagationTrainer(BaseTrainer):
    """Label propagation trainer."""

    def __init__(self, metrics, train_on="train"):
        """Initialize LabelPropagationTrainer."""
        super().__init__(metrics, train_on=train_on)

    def train(
        self,
        model: Any,
        dataset,
        split_idx: int = 0,
    ) -> Dict[str, float]:
        """Propagate labels.

        Note:
            No need to specify ``model`` in this case. The label propagation
            scheme simply propagate the seed nodes across the network.

        """
        # Train model using the training set
        train_mask = dataset.get_mask(self.train_on, split_idx)
        y_pred = model(dataset.graph, dataset.y * train_mask)

        y_true_dict, y_pred_dict, compute_results = self._setup(dataset, split_idx)
        for mask_name in dataset.masks:
            mask = dataset.get_mask(mask_name, split_idx)
            y_true_dict[mask_name] = dataset.y[mask]
            y_pred_dict[mask_name] = y_pred[mask]

        results = compute_results(dataset.masks)

        return results
