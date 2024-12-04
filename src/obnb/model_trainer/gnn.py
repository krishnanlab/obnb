from copy import deepcopy

import numpy as np
import torch

from obnb.model_trainer.base import BaseTrainer
from obnb.alltypes import Any, Callable, Dict, List, LogLevel, Optional, Tuple


class GNNTrainer(BaseTrainer):
    """Trainner for GNN models."""

    def __init__(
        self,
        metrics: Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
        train_on="train",
        val_on: str = "val",
        mask_suffix: str = "_mask",
        device: str = "cpu",
        metric_best: Optional[str] = None,
        lr: float = 0.01,
        epochs: int = 100,
        eval_steps: int = 10,
        use_negative: bool = False,
        log_level: LogLevel = "INFO",
        log_path: Optional[str] = None,
    ):
        """Initialize GNNTrainer.

        Args:
            val_on (str): Validation mask name (default: :obj:`"train"`).
            device (str): Training device (default: :obj:`"cpu"`).
            metric_best (str): Metric used for determining the best epoch.
            lr (float): Learning rate (default: :obj:`0.01`)
            epochs (int): Total epochs (default: :obj:`100`)
            eval_steps (int): Interval for evaluation (default: :obj:`10`)
            use_negative: If set to True, then try to restrict calculation of
                the loss function to only the positive and negative examples,
                and exclude those that are neutral. This will be indicated in
                the :obj:`y_mask` attribute of the data object, where the
                entries corresponding to positives or negatives are set to
                :obj:`True`.

        """
        super().__init__(
            metrics,
            train_on=train_on,
            log_level=log_level,
            log_path=log_path,
        )

        self.val_on = val_on
        self.mask_suffix = mask_suffix
        self.metric_best = metric_best
        self.lr = lr
        self.epochs = epochs
        self.eval_steps = eval_steps
        self.use_negative = use_negative
        self.device = device

    @property
    def metric_best(self):
        """Str: Metric used for determining the best model."""
        return self._metric_best

    @metric_best.setter
    def metric_best(self, metric_best):
        """Setter for :attr:`metric_best`.

        Raises:
            ValueError: More than one metrics is available but did not specify
                metric_best.
            KeyError: metric_best did not match any of the specified metrics.

        """
        if metric_best is None or metric_best == "auto":
            if "apop" in self.metrics:  # default best metric
                self._metric_best = "apop"
            elif len(self.metrics) != 1:
                raise ValueError(
                    "Multiple metrics found but did not specify metric_best",
                )
            else:
                self._metric_best = list(self.metrics)[0]
        elif metric_best not in self.metrics:
            raise KeyError(f"No metrics named {metric_best!r}")
        else:
            self._metric_best = metric_best

    def new_stats(
        self,
        masks: List[str],
    ) -> Tuple[Dict[str, List], Dict[str, float], Dict[str, torch.Tensor]]:
        """Create new stats for tracking model performance."""
        stats: Dict[str, List] = {"epoch": [], "loss": [], "time_per_epoch": []}
        best_stats: Dict[str, float] = {"epoch": 0, "loss": 1e12, "time_per_epoch": 0.0}
        best_model_state: Dict[str, torch.Tensor] = {}

        for mask_name in masks:
            for metric_name in self.metrics:
                score_name = f"{mask_name.split(self.mask_suffix)[0]}_{metric_name}"
                stats[score_name] = []
                best_stats[score_name] = 0.0

        return stats, best_stats, best_model_state

    def update_stats(
        self,
        model: Any,
        stats: Dict[str, List],
        best_stats: Dict[str, float],
        best_model_state: Dict[str, torch.Tensor],
        new_results: Dict[str, float],
        epoch: int,
        loss: float,
    ) -> None:
        """Update model performance stats using the new evaluation results.

        Args:
            model: GNN model.
            stats: Full performance history to be updated.
            best_stats: Current performance.
            best_model_state: State information of the current best model.
            new_results: New evaluation results.
            epoch: Current epoch.
            loss: Current loss.

        """
        new_results["epoch"] = epoch
        new_results["loss"] = loss
        name = f"{self.val_on}_{self.metric_best}"
        if new_results[name] > best_stats[name]:
            best_stats.update(new_results)
            best_model_state.update(deepcopy(model.state_dict()))
        for i, j in new_results.items():
            stats[i].append(j)

    def is_eval_epoch(self, cur_epoch: int) -> bool:
        """Return true if current epoch is eval epoch."""
        return cur_epoch % self.eval_steps == 0


class SimpleGNNTrainer(GNNTrainer):
    """Simple GNN trainer using Adam with fixed learning rate.

    Note:
        Do not take into account of edge weights/attrs.

    """

    def train_epoch(self, model, data, split_idx, optimizer):
        """Train a single epoch."""
        model.train()
        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

        train_mask = data[self.train_on + self.mask_suffix][:, split_idx]
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])

        y_mask = data.y_mask[train_mask]
        if self.use_negative:
            # Average of column(task)-wise mean
            loss = (loss / y_mask.float().sum(0))[y_mask].sum()
        else:
            loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, model, data, split_idx):
        """Evaluate current model."""
        model.eval()
        y_pred = model(data.x, data.edge_index).detach().cpu().numpy()
        y_true = data.y.detach().cpu().numpy()

        results = {}
        for metric_name, metric_func in self.metrics.items():
            for mask_name in data.masks:
                mask = data[mask_name][:, split_idx].detach().cpu().numpy()
                y_mask = data.y_mask[mask].detach().cpu().numpy()
                score_name = f"{mask_name.split(self.mask_suffix)[0]}_{metric_name}"
                score = metric_func(y_true[mask], y_pred[mask], y_mask=y_mask)
                results[score_name] = score

        results["time_per_epoch"] = self._elapse() / self.eval_steps

        return results

    def train(self, model, dataset, split_idx: int = 0):
        """Train the GNN model."""
        model.to(self.device)
        data = dataset.to_pyg_data(device=self.device, mask_suffix=self.mask_suffix)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        stats, best_stats, best_model_state = self.new_stats(data.masks)
        for cur_epoch in range(self.epochs):
            loss = self.train_epoch(model, data, split_idx, optimizer)

            if self.is_eval_epoch(cur_epoch):
                new_results = self.evaluate(model, data, split_idx)
                self.update_stats(
                    model,
                    stats,
                    best_stats,
                    best_model_state,
                    new_results,
                    cur_epoch,
                    loss,
                )
                self.logger.info(new_results)

        # Rewind back to best model
        model.load_state_dict(best_model_state)

        return best_stats
