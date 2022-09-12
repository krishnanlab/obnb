from nleval.model_trainer.base import StandardTrainer
from nleval.typing import Any, Dict, LogLevel


class LabelPropagationTrainer(StandardTrainer):
    """Label propagation trainer."""

    def __init__(
        self,
        metrics,
        train_on="train",
        log_level: LogLevel = "WARNING",
    ):
        """Initialize LabelPropagationTrainer."""
        super().__init__(metrics, train_on=train_on, log_level=log_level)

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

    def eval_multi_ovr(
        self,
        model: Any,
        dataset,
        split_idx: int = 0,
        consider_negative: bool = False,
    ) -> Dict[str, float]:
        """Evaluate the model in a multiclass setting.

        Note:
            The original model is not trained. For each task, a deep copy of
            the model is craeted and it is evaluted via one-vs-rest.

        """
        y_true_dict, y_pred_dict, compute_results = self._setup(dataset, split_idx)
        for i, label_id in enumerate(dataset.label.label_ids):
            y, masks = dataset.label.split(
                splitter=dataset.splitter,
                target_ids=tuple(dataset.idmap.lst),
                labelset_name=label_id,
                consider_negative=consider_negative,
            )

            train_mask = masks[self.train_on][:, split_idx]
            y_pred = model(dataset.graph, y * train_mask)

            for mask_name in masks:
                mask = masks[mask_name][:, split_idx]
                y_pred_dict[mask_name][:, i] = y_pred[mask]
                y_true_dict[mask_name][:, i] = y[mask]

            intermediate_results = compute_results(masks, label_idx=i)
            self.logger.info(f"{label_id}\t{intermediate_results}")

        results = compute_results(dataset.masks)

        return results
